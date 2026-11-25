"""六个修复点的行为测试（契约测试之外的正确性验证）。

覆盖：
1. crossfade_kernel / FadeSession —— 相位对齐的交叉淡化，消除切换咔哒；
2. unified_coord —— 方位/仰角统一笛卡尔坐标后再插值，斜上方不跳位；
3. recompute_itd —— ITD 随头围/方向重算并与卷积核一致；
4. load_before_swap / swap / ReferenceHolder —— 加载完成才发布引用；
5. rebuild_filter / ResampleFilter —— 采样率变化重建重采样抗混叠滤波；
6. hold_old_kernel —— 旧核强引用保留到淡出结束。
另含端到端引擎与 metrics/audit/fallback/determinism 的稳定性检查。
"""

from __future__ import annotations

import math
import threading
import time

import pytest

import media.hrtf_kernel_switch as m
from media.hrtf_kernel_switch import (
    ConvolutionStream,
    FadeSession,
    HRTFEngine,
    HRTFDataset,
    ReferenceHolder,
    ResampleFilter,
    discontinuity,
    load_before_swap,
    recompute_itd,
    unified_coord,
)


# ---------------------------------------------------------------------------
# 1. 坐标系统一
# ---------------------------------------------------------------------------


def test_az_el_roundtrip_and_known_axes():
    assert unified_coord(0, 0).vector == pytest.approx((0, 0, 1), abs=1e-9)   # 正前
    assert unified_coord(90, 0).vector == pytest.approx((1, 0, 0), abs=1e-9)  # 正右
    assert unified_coord(0, 90).vector == pytest.approx((0, 1, 0), abs=1e-9)   # 头顶
    u = unified_coord(37.0, -21.0)
    assert (u.azimuth, u.elevation) == pytest.approx((37.0, -21.0), abs=1e-8)


def test_binaural_and_vertical_conventions_map_to_same_vector():
    # 斜后上方声源
    u = unified_coord(-120.0, 55.0)
    binaural = unified_coord(u.lateral, u.polar, convention="binaural")
    assert binaural.vector == pytest.approx(u.vector, abs=1e-9)

    # vertical-polar 的第二角是从头顶量的极角 = 90 - elevation
    vertical = unified_coord(u.azimuth, 90.0 - u.elevation, convention="vertical")
    assert vertical.vector == pytest.approx(u.vector, abs=1e-9)


def test_unified_interpolation_between_conventions_does_not_jump():
    az, el = -130.0, 62.0
    std = unified_coord(az, el)
    other = unified_coord(std.lateral, std.polar, convention="binaural")
    # 同一个物理方向，两种坐标表示；统一后插值必须稳定在该方向
    mid = unified_coord(std.vector, other.vector, t=0.5)
    assert mid.vector == pytest.approx(std.vector, abs=1e-9)
    assert sum(v * v for v in mid.vector) == pytest.approx(1.0, abs=1e-9)

    # 反例：把双耳横向角直接当方位角、仰角沿用另一套坐标，会产生明显跳位
    naive = m.az_el_to_xyz(std.lateral, el)
    jump = math.dist(naive, std.vector)
    assert jump > 0.3  # 复现题目所说的“斜上方跳位”


# ---------------------------------------------------------------------------
# 2. ITD 随头围重算
# ---------------------------------------------------------------------------


def test_itd_front_is_zero_and_scales_linearly_with_circumference():
    assert recompute_itd(57.0, 0.0, 0.0).itd_seconds == pytest.approx(0.0, abs=1e-12)
    small = recompute_itd(40.0, 90.0, 0.0)
    large = recompute_itd(80.0, 90.0, 0.0)
    assert large.itd_seconds / small.itd_seconds == pytest.approx(2.0, rel=1e-9)


def test_itd_value_matches_woodworth_and_realistic_range():
    r = recompute_itd(57.0, 90.0, 0.0, sample_rate=48000)
    assert r.itd_ms == pytest.approx(0.68, abs=0.03)
    assert 30 <= r.itd_samples <= 35
    # 米/厘米两种传法一致
    r_m = recompute_itd(0.57, 90.0, 0.0, sample_rate=48000)
    assert r_m.itd_samples == r.itd_samples


def test_itd_ear_assignment_flips_with_side_and_rate():
    right = recompute_itd(57.0, 90.0, 0.0, sample_rate=48000)
    assert right.left_delay_samples == right.itd_samples  # 右声源 -> 左耳远
    assert right.right_delay_samples == 0
    left = recompute_itd(57.0, -90.0, 0.0, sample_rate=48000)
    assert left.right_delay_samples == left.itd_samples
    assert left.left_delay_samples == 0
    hi = recompute_itd(57.0, 90.0, sample_rate=96000)
    assert hi.itd_samples == pytest.approx(2 * right.itd_samples, abs=1)


# ---------------------------------------------------------------------------
# 3. 相位对齐的核交叉淡化
# ---------------------------------------------------------------------------


def test_delay_alignment_detects_onset_offset_and_polarity():
    old = [0.0] * 20
    old[4] = 1.0
    new = [0.0] * 20
    new[9] = 0.8
    al = m._best_delay_alignment(old, new, 12)
    # aligned[n] = pol * new[n - shift]；new 晚 5 个采样 -> shift = -5
    assert al.shift == -5
    assert al.polarity == 1

    new_inv = [-v for v in new]
    al2 = m._best_delay_alignment(old, new_inv, 12)
    assert al2.polarity == -1


def test_phase_align_moves_new_pulse_on_top_of_old():
    old = [0.0] * 20
    old[5] = 1.0
    new = [0.0] * 20
    new[11] = 1.0  # 晚 6 个采样

    aligned = m.crossfade_kernel(old, new, 12, curve="linear", phase_align=True)
    gain_only = m.crossfade_kernel(old, new, 12, curve="linear", phase_align=False)

    assert aligned.phase_aligned is True
    assert aligned.shifts[0] == -6
    # 对齐后两个脉冲叠加在 bin 5，线性淡化增益和为 1 -> 相干峰值 ~1
    assert aligned.kernel[5] == pytest.approx(1.0, abs=1e-9)
    assert abs(aligned.kernel[11]) < 1e-9
    # 只做增益不做相位：新脉冲仍留在 bin 11，形成两个分离瞬态
    assert gain_only.kernel[11] == pytest.approx(1.0, abs=1e-9)
    assert 0.4 < gain_only.kernel[5] < 0.7


def test_fade_session_reduces_switch_click_vs_hard_replace():
    def kernel(seed: int, shift: int) -> list[float]:
        _, rnd = m._splitmix64(seed, 64)
        h = [0.0] * 64
        p = 8 + shift
        h[p] = 1.0
        h[p + 1] = -0.6
        for k in range(p + 2, 64):
            h[k] = 0.25 * math.exp(-(k - p) / 10.0) * (rnd[k] * 2 - 1)
        return h

    old_l, old_r = kernel(1, 0), kernel(2, 0)
    new_l, new_r = kernel(3, 5), kernel(4, 5)
    x = [0.3 * math.sin(2 * math.pi * 220 * i / 48000) for i in range(256)]
    start, fade_n = 64, 48

    # 硬替换基线：输入历史连续，但核在 start 处瞬间换掉
    sl, sr = ConvolutionStream(old_l), ConvolutionStream(old_r)
    hard_l = sl.process(x[:start])
    hard_r = sr.process(x[:start])
    nl, nr = ConvolutionStream(new_l), ConvolutionStream(new_r)
    nl.history, nr.history = list(sl.history), list(sr.history)
    hard_l += nl.process(x[start:])
    hard_r += nr.process(x[start:])
    hard_jump = max(
        discontinuity(hard_l[start - 3:start + 4]),
        discontinuity(hard_r[start - 3:start + 4]),
    )

    sess = FadeSession((old_l, old_r), (new_l, new_r), fade_n, start_sample=start)
    head = sess.process(x)
    tail = sess.flush()
    fade_l = list(head.left) + list(tail.left)
    fade_r = list(head.right) + list(tail.right)
    fade_jump = max(
        discontinuity(fade_l[start - 3:start + fade_n + 4]),
        discontinuity(fade_r[start - 3:start + fade_n + 4]),
    )
    assert hard_jump > 0.02
    assert fade_jump < 0.5 * hard_jump  # 实测约为硬切换的 5%


# ---------------------------------------------------------------------------
# 4. 旧核保留到淡出结束
# ---------------------------------------------------------------------------


def test_old_kernel_is_held_until_fade_completes():
    old, new = [1.0, 0.5], [0.8, 0.2]
    sess = FadeSession(old, new, 16)
    assert sess.old_kernel is not None
    assert sess.active and not sess.released_old

    sess.process([0.1] * 8)
    assert not sess.released_old
    assert sess.old_kernel is not None  # 淡出未结束，旧核仍被强引用

    sess.process([0.1] * 8)
    assert sess.released_old
    assert sess.old_kernel is None


def test_hold_old_kernel_entry_reports_retention():
    rep = m.hold_old_kernel([1.0, 0.0], [0.5, 0.5], 8)
    assert rep.ok and rep.retained and not rep.released
    sess = rep.session
    out = m.hold_old_kernel(sess, [0.25] * 8)
    assert out.released is True
    assert out.session is sess


# ---------------------------------------------------------------------------
# 5. 加载完成再换引用（原子发布）
# ---------------------------------------------------------------------------


def test_holder_refuses_incomplete_dataset():
    h = ReferenceHolder()
    half = HRTFDataset("half")  # 未 freeze -> complete=False
    with pytest.raises(RuntimeError):
        h.publish(half)
    assert h.get() is None


def test_load_before_swap_publishes_only_complete_and_keeps_old_on_failure():
    engine = HRTFEngine()
    before = engine.current.name

    ok = load_before_swap(lambda: m.make_dataset("ari"), engine, fade_samples=4)
    assert ok.ok and ok.complete and ok.swapped
    assert engine.current.name == "ari"

    def boom():
        raise OSError("simulated disk error")

    bad = load_before_swap(boom, engine)
    assert not bad.ok and bad.kept_previous
    assert engine.current.name == "ari"  # 旧引用原封不动
    assert before == "generic"


def test_audio_thread_never_reads_half_loaded_kernel():
    holder = ReferenceHolder(m.make_dataset("generic"))
    violations: list[str] = []
    stop = threading.Event()

    def audio_reader():
        while not stop.is_set():
            ref = holder.snapshot()  # 音频线程每块只快照一次
            if ref is not None and not ref.complete:
                violations.append("read incomplete dataset")
            if ref is not None:
                ref.direction_kernel(15.0, 10.0)

    reader = threading.Thread(target=audio_reader)
    reader.start()
    try:
        for k in range(4):
            ds = HRTFDataset(f"threadset{k}", sample_rate=48000)
            for i in range(400):  # 模拟主线程慢慢填充半成品
                ds.metadata[f"p{i}"] = i
            time.sleep(0.002)
            ds.freeze()
            holder.publish(ds)
            time.sleep(0.002)
    finally:
        stop.set()
        reader.join(timeout=2)

    assert violations == []
    assert holder.get().complete


# ---------------------------------------------------------------------------
# 6. 采样率变化重建滤波器
# ---------------------------------------------------------------------------


def test_rebuild_filter_only_on_rate_change():
    f = ResampleFilter(44100, 48000)
    same = m.rebuild_filter(f, source_rate=44100, target_rate=48000)
    assert same.rebuilt is False

    changed = m.rebuild_filter(f, source_rate=96000, target_rate=48000)
    assert changed.rebuilt and changed.rate_changed
    assert changed.source_rate == 96000
    # 截止频率 = 0.45 * min(96000, 48000)
    assert changed.cutoff_hz == pytest.approx(21600.0)
    assert len(changed.coefficients) == changed.taps


def test_engine_rebuilds_filter_when_dataset_rate_differs():
    engine = HRTFEngine()  # 输出 48k，generic 48k
    assert engine.resample_filter.source_rate == 48000
    rep = engine.select_set("cipic")  # 44.1k
    assert rep.filter.rebuilt is True
    assert engine.resample_filter.source_rate == 44100

    again = engine.select_set("cipic")
    assert again.filter.rebuilt is False  # 同采样率不重复重建

    hi = engine.select_set("highrate")  # 96k
    assert hi.filter.rebuilt is True
    assert engine.resample_filter.source_rate == 96000


def test_rebuilt_filter_preserves_high_frequency_that_stale_filter_cuts():
    wide = ResampleFilter(96000, 96000)
    assert wide.frequency_gain(30000) > 0.95   # 重建到 96k 后 30k 保留
    stale_48 = ResampleFilter(48000, 48000)
    assert stale_48.frequency_gain(23000) < 0.5  # 旧 48k 滤波在 23k 已截断
    assert stale_48.frequency_gain(12000) > 0.95  # 带内正常


# ---------------------------------------------------------------------------
# 端到端：ITD 与卷积核一致、声像方向正确
# ---------------------------------------------------------------------------


def test_engine_itd_matches_kernel_onset_after_update():
    engine = HRTFEngine()  # generic 48k / 57cm
    engine.update_source(90, 0)
    y = engine.convolve([1.0])
    peak_l = max(range(len(y.left)), key=lambda i: abs(y.left[i]))
    peak_r = max(range(len(y.right)), key=lambda i: abs(y.right[i]))
    assert peak_l - peak_r == engine.itd.itd_samples  # 严格一致，不偏移
    # 右声源：右耳（近）更响
    assert max(abs(v) for v in y.right) > max(abs(v) for v in y.left)

    engine.set_head_circumference(80.0)
    engine.update_source(90, 0)
    y2 = engine.convolve([1.0])
    pl = max(range(len(y2.left)), key=lambda i: abs(y2.left[i]))
    pr = max(range(len(y2.right)), key=lambda i: abs(y2.right[i]))
    assert pl - pr == engine.itd.itd_samples
    assert engine.itd.itd_samples > 40


def test_select_then_update_still_matches_itd():
    # 待播淡化尚未出声时更新方向，淡化核必须按新方向重建
    engine = HRTFEngine()
    engine.select_set("ari", fade_samples=32)
    engine.update_source(90, 0)
    y = engine.convolve([1.0])
    pl = max(range(len(y.left)), key=lambda i: abs(y.left[i]))
    pr = max(range(len(y.right)), key=lambda i: abs(y.right[i]))
    assert pl - pr == engine.itd.itd_samples
    assert max(abs(v) for v in y.right) > max(abs(v) for v in y.left)


def test_engine_switch_set_renders_finite_stereo_and_validates():
    engine = HRTFEngine()
    engine.select_set("ari", fade_samples=32)
    engine.update_source(40, 30, t=0.5)
    sig = [0.2 * math.sin(2 * math.pi * 440 * i / 48000) for i in range(256)]
    y = engine.convolve(sig)
    assert len(y.left) == len(y.right)
    assert all(math.isfinite(v) for v in y.left + y.right)
    report = engine.validate()
    assert report.ok, report.errors


def test_streaming_process_keeps_equal_block_sizes():
    engine = HRTFEngine()
    engine.select_set("cipic", fade_samples=16)
    for _ in range(6):
        y = engine.process([0.1] * 64)
        assert len(y.left) == 64 and len(y.right) == 64
    assert engine.validate().ok


# ---------------------------------------------------------------------------
# 通用入口：metrics / audit / fallback / determinism / validate
# ---------------------------------------------------------------------------


def test_determinism_is_stable_and_seed_sensitive():
    a = m.determinism(123, count=16)
    b = m.determinism(123, count=16)
    c = m.determinism(124, count=16)
    assert a.integers == b.integers and a["values"] == b["values"]
    assert a.sequence == b.sequence
    assert a.reproducible is True
    assert a.integers != c.integers


def test_fallback_equal_power_stereo():
    f = m.fallback("hrtf missing", azimuth=90)
    assert f.ok and f.stereo
    assert f.left_gain ** 2 + f.right_gain ** 2 == pytest.approx(1.0, abs=1e-9)
    y = f.process([1.0, 1.0])
    assert len(y.left) == 2 and len(y.right) == 2


def test_metrics_and_audit_cover_switch_lifecycle():
    engine = HRTFEngine()
    engine.select_set("cipic")
    engine.update_source(20, 10)
    engine.convolve([0.5, -0.5, 0.25])
    stats = engine.metrics([0.1, 0.2, -0.1])
    assert stats.rms > 0 and stats.peak == pytest.approx(0.2)
    trail = engine.audit()
    assert trail.ok
    types = {r.get("type") for r in trail.records}
    assert "swap" in types and "filter_rebuild" in types and "update_source" in types
    assert trail.filter_rebuilds >= 1


def test_validate_spatial_accepts_scalar_and_engine():
    assert m.validate_spatial(1).ok is True
    engine = HRTFEngine()
    assert m.validate_spatial(engine).ok is True
    assert m.validate_spatial({"sample_rate": 48000, "azimuth": 30}).ok is True


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("crossfade_kernel", dict(old=[1.0, 0.0], new=[0.0, 1.0], fade_samples=2)),
        ("unified_coord", dict(azimuth=30, elevation=15)),
        ("recompute_itd", dict(head_circumference_cm=57, azimuth=60)),
        ("load_before_swap", dict(spec="ari")),
        ("rebuild_filter", dict(source_rate=44100)),
        ("hold_old_kernel", dict(old=[1.0], new=[0.5], fade_samples=1)),
        ("select_set", dict(name="generic")),
        ("update_source", dict(azimuth=10, elevation=5)),
        ("convolve", dict(signal=[1.0, 0.5], kernel=[1.0])),
        ("swap", dict()),
        ("validate_spatial", dict()),
        ("metrics", dict(signal=[1.0, -1.0])),
        ("audit", dict()),
        ("fallback", dict(reason="test")),
        ("determinism", dict(seed=7)),
    ],
)
def test_keyword_calls_return_reports(name, kwargs):
    fn = getattr(m, name)
    assert fn(**kwargs) is not None
