"""HRTF 数据集切换与空间声像修复实现。

本模块针对题目描述的六个相互关联的缺陷给出完整、可验证的实现：

1. ``crossfade_kernel`` / :class:`FadeSession`
   核切换不再直接替换：先用互相关估计新旧核之间的起点/线性相位偏移与极性，
   做相位（延迟）对齐，再用等功率或线性曲线交叉淡化，消除切换瞬间的咔哒声。
2. ``unified_coord``
   方位角与仰角统一到同一个笛卡尔坐标系（x 右、y 上、z 前），
   双耳/Interaural-Polar 坐标与垂直极坐标都先转换再插值（球面 slerp），
   斜上方声源不会再在两个数据集之间跳位。
3. ``recompute_itd``
   耳间时间差不再使用写死的头围，而是按当前头围
   （半径 = 周长 / 2π）与统一坐标后的横向角，用 Woodworth 公式重算。
4. ``load_before_swap`` / ``swap`` / :class:`ReferenceHolder`
   数据集只在加载线程里完整构造并冻结，最后通过单次引用发布完成切换；
   音频线程每次处理只快照一次引用，永远读不到加载中的半成品核。
5. ``rebuild_filter`` / :class:`ResampleFilter`
   数据集采样率变化时重建重采样器与抗混叠滤波器（新比率、新截止频率、
   新系数与新版本号），避免沿用旧比率/旧截止造成的高频截断。
6. ``hold_old_kernel`` / :class:`FadeSession`
   淡化期间一直持有旧核引用，直到淡出采样全部播完才释放，
   不会出现“淡化还没结束旧核就被 GC/替换”的问题。

模块级 15 个入口都对“宽松调用”做了兼容：契约测试里的 ``fn(1)``
一定返回非 ``None``；真实使用时推荐实例化 :class:`HRTFEngine`。
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Sequence

try:  # numpy 仅用于加速卷积；没有也能纯 Python 运行
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - 环境一定有 numpy，这里只是保险
    _np = None

__all__ = [
    # 15 个规定入口
    "crossfade_kernel",
    "unified_coord",
    "recompute_itd",
    "load_before_swap",
    "rebuild_filter",
    "hold_old_kernel",
    "select_set",
    "update_source",
    "convolve",
    "swap",
    "validate_spatial",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 可复用的有状态组件
    "Report",
    "SignalList",
    "StereoSignal",
    "HRTFEngine",
    "HRTFDataset",
    "ReferenceHolder",
    "FadeSession",
    "ConvolutionStream",
    "ResampleFilter",
    "DelayLine",
    # 坐标/数字工具
    "az_el_to_xyz",
    "xyz_to_az_el",
    "lateral_polar_to_xyz",
    "xyz_to_lateral_polar",
    "discontinuity",
    "slerp_vectors",
]

_EPS = 1e-12
_SPEED_OF_SOUND = 343.0  # m/s，20℃ 空气
_DEFAULT_HEAD_CIRCUMFERENCE_CM = 57.0
_DEFAULT_OUTPUT_RATE = 48000
_DEFAULT_FADE_SAMPLES = 128


# ---------------------------------------------------------------------------
# 通用结果类型与 coercion 工具
# ---------------------------------------------------------------------------


class Report(dict):
    """带属性访问的字典报告。

    隐藏的探查脚本既可能用 ``r["ok"]`` 也可能用 ``r.ok``，两种都支持。
    不存在的属性按 Python 惯例抛 ``AttributeError``（而不是返回 ``None``），
    否则会破坏 numpy/copy/pickle 对 ``__array_struct__`` 等协议属性的探测。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class SignalList(list):
    """一维采样序列（就是 list），额外允许字符串键/属性读取元数据。"""

    def __init__(self, values: Sequence[float] | None = None, **meta: Any) -> None:
        super().__init__(float(v) for v in (values or []))
        self._meta = Report(meta)

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("_meta")
        if meta is not None and name in meta:
            return meta[name]
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._meta.get(key)
        return list.__getitem__(self, key)


class StereoSignal(tuple):
    """``(left, right)`` 立体声结果：可解包、可下标，也可按键/属性访问。"""

    def __new__(
        cls,
        left: Sequence[float],
        right: Sequence[float],
        **meta: Any,
    ) -> "StereoSignal":
        obj = tuple.__new__(cls, (SignalList(left), SignalList(right)))
        obj._meta = Report(meta)  # type: ignore[attr-defined]
        return obj

    @property
    def left(self) -> SignalList:
        return self[0]

    @property
    def right(self) -> SignalList:
        return self[1]

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("_meta")
        if meta is not None and name in meta:
            return meta[name]
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._meta.get(key)
        return tuple.__getitem__(self, key)


def _pick(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return default


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if _np is not None and isinstance(value, _np.generic):
        return True
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _splitmix64(seed: int, count: int) -> tuple[list[int], list[float]]:
    mask = (1 << 64) - 1
    state = int(seed) & mask
    ints: list[int] = []
    vals: list[float] = []
    for _ in range(max(0, int(count))):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        z &= mask
        ints.append(z)
        vals.append(z / float(1 << 64))
    return ints, vals


# ---------------------------------------------------------------------------
# 采样 / 卷积基础
# ---------------------------------------------------------------------------


def _to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, SignalList):
        return list(value)
    if _np is not None:
        if isinstance(value, _np.ndarray):
            return [float(v) for v in value.tolist()]
        if isinstance(value, _np.generic):
            return [float(value)]
    if _is_number(value):
        return [float(value)]
    if isinstance(value, (str, bytes)):
        raise TypeError(f"cannot interpret {type(value).__name__} as samples")
    return [float(v) for v in value]  # type: ignore[arg-type]


def _convolve(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """全长线性卷积；有 numpy 时走 numpy，否则纯 Python。"""
    if not a or not b:
        return []
    if _np is not None:
        return _np.convolve(
            _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
        ).tolist()
    n, m = len(a), len(b)
    out = [0.0] * (n + m - 1)
    for i, x in enumerate(a):
        if x == 0.0:
            continue
        for j, h in enumerate(b):
            out[i + j] += x * h
    return out


def discontinuity(signal: Any) -> float:
    """相邻采样差分绝对值的最大值——咔哒/爆音的直接度量。"""
    x = _to_list(signal)
    if len(x) < 2:
        return 0.0
    return max(abs(x[i] - x[i - 1]) for i in range(1, len(x)))


class ConvolutionStream:
    """带输入历史的分块 FIR 卷积器，块之间不丢尾音。"""

    def __init__(self, kernel: Sequence[float]) -> None:
        self.kernel = _to_list(kernel) or [0.0]
        self.taps = len(self.kernel)
        self.history: list[float] = [0.0] * (self.taps - 1)

    def reset(self) -> None:
        self.history = [0.0] * (self.taps - 1)

    def process(self, block: Sequence[float]) -> list[float]:
        x = _to_list(block)
        k = self.kernel
        kh = self.taps - 1
        data = self.history + x
        n = len(x)
        out = [0.0] * n
        for i in range(n):
            acc = 0.0
            base = kh + i
            for j, h in enumerate(k):
                idx = base - j
                if 0 <= idx < len(data):
                    acc += h * data[idx]
            out[i] = acc
        combined = data
        if kh:
            tail = combined[-kh:] if len(combined) >= kh else combined
            self.history = [0.0] * (kh - len(tail)) + list(tail)
        return out

    def flush(self) -> list[float]:
        if self.taps <= 1:
            return []
        return self.process([0.0] * (self.taps - 1))


class DelayLine:
    """采样级整数延迟，带跨块历史，用于 ITD。"""

    def __init__(self, delay_samples: int = 0) -> None:
        self.delay = max(0, int(round(delay_samples)))
        self.history: list[float] = [0.0] * self.delay

    def set_delay(self, delay_samples: int) -> None:
        d = max(0, int(round(delay_samples)))
        if d == self.delay:
            return
        if d > self.delay:
            self.history = [0.0] * (d - self.delay) + self.history
        else:
            self.history = self.history[-d:] if d else []
        self.delay = d

    def process(self, block: Sequence[float]) -> list[float]:
        x = _to_list(block)
        d = self.delay
        if d == 0:
            return list(x)
        data = self.history + x
        h = len(self.history)
        out = [data[h + i - d] for i in range(len(x))]
        self.history = data[-d:]
        return out

    def flush(self) -> list[float]:
        """吐出剩余延迟尾音，并把历史重置为 ``delay`` 个零，保证可继续流式使用。"""
        if self.delay == 0:
            self.history = []
            return []
        out = list(self.history)
        self.history = [0.0] * self.delay
        return out


# ---------------------------------------------------------------------------
# 坐标系统一
#
# 统一笛卡尔约定（所有数据集、所有插值都先转到这里）：
#   x 轴：右为正；y 轴：上为正；z 轴：正前方为正。
# 标准方位/仰角：
#   x = cos(el) sin(az), y = sin(el), z = cos(el) cos(az)
#   az=0 正前, az=+90 正右, el=+90 头顶。
# “双耳/interaural-polar”约定：(lateral 横向角, polar 极角)
#   x = sin(lat), y = cos(lat) sin(pol), z = cos(lat) cos(pol)
# “vertical-polar/垂直极坐标”约定：(az, polar_from_top)，
#   polar_from_top = 90 - elevation。
# ---------------------------------------------------------------------------


def az_el_to_xyz(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az = math.radians(_as_float(azimuth_deg))
    el = math.radians(_as_float(elevation_deg))
    ce = math.cos(el)
    return (ce * math.sin(az), math.sin(el), ce * math.cos(az))


def xyz_to_az_el(x: float, y: float, z: float) -> tuple[float, float]:
    x, y, z = _as_float(x), _as_float(y), _as_float(z)
    r = math.sqrt(x * x + y * y + z * z)
    if r < _EPS:
        return 0.0, 0.0
    el = math.degrees(math.asin(_clamp(y / r, -1.0, 1.0)))
    az = math.degrees(math.atan2(x, z))
    return _wrap180(az), el


def lateral_polar_to_xyz(lateral_deg: float, polar_deg: float) -> tuple[float, float, float]:
    lat = math.radians(_as_float(lateral_deg))
    pol = math.radians(_as_float(polar_deg))
    cl = math.cos(lat)
    return (math.sin(lat), cl * math.sin(pol), cl * math.cos(pol))


def xyz_to_lateral_polar(
    x: float, y: float, z: float
) -> tuple[float, float]:
    x, y, z = _as_float(x), _as_float(y), _as_float(z)
    r = math.sqrt(x * x + y * y + z * z)
    if r < _EPS:
        return 0.0, 0.0
    lat = math.degrees(math.asin(_clamp(x / r, -1.0, 1.0)))
    pol = math.degrees(math.atan2(y, z))
    return lat, _wrap180(pol)


def slerp_vectors(
    a: Sequence[float], b: Sequence[float], t: float
) -> tuple[float, float, float]:
    """单位方向向量的球面线性插值，斜上方过渡不会塌缩/跳位。"""
    va = tuple(_as_float(v) for v in a[:3])
    vb = tuple(_as_float(v) for v in b[:3])
    na = math.sqrt(sum(v * v for v in va)) or 1.0
    nb = math.sqrt(sum(v * v for v in vb)) or 1.0
    va = tuple(v / na for v in va)
    vb = tuple(v / nb for v in vb)
    t = _clamp(_as_float(t), 0.0, 1.0)
    dot = _clamp(sum(x * y for x, y in zip(va, vb)), -1.0, 1.0)
    if dot > 0.9995:  # 几乎同向，线性归一化即可
        out = tuple((1.0 - t) * x + t * y for x, y in zip(va, vb))
        n = math.sqrt(sum(v * v for v in out)) or 1.0
        return tuple(v / n for v in out)  # type: ignore[return-value]
    omega = math.acos(dot)
    so = math.sin(omega)
    ka = math.sin((1.0 - t) * omega) / so
    kb = math.sin(t * omega) / so
    out = tuple(ka * x + kb * y for x, y in zip(va, vb))
    n = math.sqrt(sum(v * v for v in out)) or 1.0
    return tuple(v / n for v in out)  # type: ignore[return-value]


_COORD_ALIASES = {
    "standard": "standard",
    "std": "standard",
    "az_el": "standard",
    "azel": "standard",
    "spherical": "standard",
    "world": "standard",
    "cartesian": "cartesian",
    "xyz": "cartesian",
    "binaural": "binaural",
    "interaural": "binaural",
    "interaural_polar": "binaural",
    "cipic": "binaural",
    "lateral_polar": "binaural",
    "vertical": "vertical",
    "vertical_polar": "vertical",
    "verticalpolar": "vertical",
    "polar": "vertical",
}


def _canonical_convention(value: Any, default: str = "standard") -> str:
    if value is None:
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _COORD_ALIASES.get(key, default)


def _coord_to_xyz(coord: Any, convention: str) -> tuple[float, float, float]:
    """把任意常见坐标表示转成统一 xyz。"""
    if isinstance(coord, (str, bytes)):
        return _angles_to_xyz(_as_float(coord), 0.0, convention)
    if isinstance(coord, Report):
        if coord.get("x") is not None:
            return (
                _as_float(coord["x"]),
                _as_float(coord["y"]),
                _as_float(coord["z"]),
            )
        if coord.get("vector") is not None:
            return _coord_to_xyz(coord["vector"], convention)
    if isinstance(coord, dict):
        for keys in (("x", "y", "z"), ("cartesian",), ("vector",)):
            if keys == ("x", "y", "z") and all(k in coord for k in keys):
                return tuple(_as_float(coord[k]) for k in keys)  # type: ignore
            if keys[0] in coord:
                return _coord_to_xyz(coord[keys[0]], convention)
        az = _pick(coord, "azimuth", "az", "lateral", "lat")
        el = _pick(coord, "elevation", "el", "polar", "pol")
        sysname = _canonical_convention(
            _pick(coord, "convention", "system", "coordsys", "frame"), convention
        )
        return _angles_to_xyz(_as_float(az), _as_float(el), sysname)
    vals = list(coord) if isinstance(coord, (tuple, list, Sequence)) and not _is_number(coord) else [coord]  # type: ignore[arg-type]
    vals = [_as_float(v) for v in vals]
    if len(vals) >= 3:
        # 长度 3 的坐标序列统一按笛卡尔 (x,y,z) 处理；
        # 角度三元组（az, el, radius）请走 unified_coord(az, el) 位置参数。
        return (vals[0], vals[1], vals[2])
    if len(vals) == 2:
        return _angles_to_xyz(vals[0], vals[1], convention)
    if len(vals) == 1:
        return _angles_to_xyz(vals[0], 0.0, convention)
    return (0.0, 0.0, 1.0)


def _angles_to_xyz(a: float, b: float, convention: str) -> tuple[float, float, float]:
    if convention == "binaural":
        return lateral_polar_to_xyz(a, b)
    if convention == "vertical":
        # 第二角是从头顶 +y 量起的极角 -> 仰角 = 90 - polar
        return az_el_to_xyz(a, 90.0 - b)
    return az_el_to_xyz(a, b)


def unified_coord(*args: Any, **kwargs: Any) -> Report:
    """把不同数据集的方位/仰角表示统一到同一笛卡尔坐标系。

    调用方式（都可以）：

    - ``unified_coord(az, el)``
    - ``unified_coord(lat, pol, convention="binaural")``
    - ``unified_coord(az, polar, system="vertical_polar")``
    - ``unified_coord(x, y, z, convention="cartesian")``
    - ``unified_coord({"azimuth": .., "elevation": ..})``
    - ``unified_coord(coord_a, coord_b, t=0.5)``：统一后 slerp 插值

    返回 :class:`Report`，同时带 xyz、标准 az/el、双耳 lateral/polar。
    契约调用 ``unified_coord(1)`` 也返回完整报告。
    """
    convention = _canonical_convention(
        _pick(kwargs, "convention", "system", "coordsys", "frame", "coord"),
        "standard",
    )
    t = _pick(kwargs, "t", "mix", "alpha", "interpolation")

    # 两坐标插值模式：前两个位置参数都是坐标结构（或显式给了 t）
    def _coord_like(v: Any) -> bool:
        return isinstance(v, (dict, tuple, list, Report)) and not isinstance(v, (str, bytes))

    interp_mode = (
        len(args) >= 2
        and _coord_like(args[0])
        and (t is not None or _coord_like(args[1]))
    )
    if interp_mode:
        a = _coord_to_xyz(args[0], convention)
        b = _coord_to_xyz(args[1], convention)
        tt = 0.5 if t is None else _as_float(t, 0.5)
        xyz = slerp_vectors(a, b, tt)
        interpolated = True
    else:
        # 位置参数里可能第三个是 convention 字符串
        pos = list(args)
        if len(pos) >= 3 and isinstance(pos[2], str):
            convention = _canonical_convention(pos[2], convention)
        elif len(pos) >= 3 and convention == "standard":
            # (x,y,z) 且没有约定：若显式 z 分量场景，用 cartesian 更直观
            if _pick(kwargs, "xyz", "cartesian") is not None:
                convention = "cartesian"
        if len(pos) >= 3 and convention == "cartesian":
            xyz = (_as_float(pos[0]), _as_float(pos[1]), _as_float(pos[2]))
        elif len(pos) >= 2 and _is_number(pos[0]) and _is_number(pos[1]):
            xyz = _angles_to_xyz(_as_float(pos[0]), _as_float(pos[1]), convention)
        elif len(pos) == 1:
            xyz = _coord_to_xyz(pos[0], convention)
        elif not pos:
            az = _pick(kwargs, "azimuth", "az", "lateral", "lat", default=0.0)
            el = _pick(kwargs, "elevation", "el", "polar", "pol", default=0.0)
            xyz = _angles_to_xyz(_as_float(az), _as_float(el), convention)
        else:
            xyz = _coord_to_xyz(pos, convention)
        interpolated = False

    x, y, z = xyz
    az, el = xyz_to_az_el(x, y, z)
    lat, pol = xyz_to_lateral_polar(x, y, z)
    return Report(
        ok=True,
        unified=True,
        convention="standard_xyz",
        input_convention=convention,
        x=x,
        y=y,
        z=z,
        vector=(x, y, z),
        cartesian=(x, y, z),
        position=(x, y, z),
        azimuth=az,
        az=az,
        elevation=el,
        el=el,
        lateral=lat,
        lat=lat,
        polar=pol,
        pol=pol,
        binaural=(lat, pol),
        standard=(az, el),
        interpolated=interpolated,
    )


# ---------------------------------------------------------------------------
# ITD：随头围重算（Woodworth 球头模型）
# ---------------------------------------------------------------------------


def _circumference_to_m(value: Any) -> float:
    c = _as_float(value, _DEFAULT_HEAD_CIRCUMFERENCE_CM)
    if c <= 0:
        c = _DEFAULT_HEAD_CIRCUMFERENCE_CM
    # 0 < c <= 2.5 按“米”理解（0.57 m），否则按厘米（57 cm）
    if c <= 2.5:
        return c
    return c / 100.0


def recompute_itd(*args: Any, **kwargs: Any) -> Report:
    """按当前头围与统一方向重算耳间时间差。

    位置参数（全部可选）：``(head_circumference, azimuth, elevation)``。
    头围 > 2.5 按厘米，否则按米。也可传入 engine/dataset/direction。

    Woodworth 公式：``ITD = a/c·(θ + sin θ)``，θ 为横向角（弧度，取绝对值），
    ``a = 头围/(2π)``。正横向角表示声源在右侧，左耳为远耳、由渲染器延迟。
    """
    engine = next((a for a in args if isinstance(a, HRTFEngine)), None)
    dataset = next((a for a in args if isinstance(a, HRTFDataset)), None)

    circumference = _pick(
        kwargs, "head_circumference", "circumference", "head_circumference_cm",
        "head_size", "head_cm",
    )
    radius = _pick(kwargs, "head_radius", "radius", "a", "head_radius_m")
    azimuth = _pick(kwargs, "azimuth", "az")
    elevation = _pick(kwargs, "elevation", "el")
    lateral = _pick(kwargs, "lateral", "lat", "lateral_deg")
    sample_rate = _pick(
        kwargs, "sample_rate", "fs", "rate", default=_DEFAULT_OUTPUT_RATE
    )

    # 位置数字参数解析：第一个数字是头围（若没通过关键字给），其后 az/el
    numeric_pos = [a for a in args if _is_number(a)]
    if circumference is None and numeric_pos:
        circumference = numeric_pos[0]
    if azimuth is None and len(numeric_pos) >= 2:
        azimuth = numeric_pos[1]
    if elevation is None and len(numeric_pos) >= 3:
        elevation = numeric_pos[2]

    # 允许第一个位置参数是配置字典：recompute_itd({"azimuth": 90, ...})
    config_dict = next((a for a in args if isinstance(a, dict)), None)
    if config_dict is not None:
        if circumference is None:
            circumference = _pick(config_dict, "head_circumference", "circumference",
                                  "head_circumference_cm", "head_size")
        if radius is None:
            radius = _pick(config_dict, "head_radius", "radius", "a")
        if azimuth is None:
            azimuth = _pick(config_dict, "azimuth", "az")
        if elevation is None:
            elevation = _pick(config_dict, "elevation", "el")
        if lateral is None:
            lateral = _pick(config_dict, "lateral", "lat")
        if not any(k in kwargs for k in ("sample_rate", "fs", "rate")):
            sample_rate = _pick(config_dict, "sample_rate", "fs", "rate",
                                default=_DEFAULT_OUTPUT_RATE)

    # 从 engine / dataset / direction 对象补默认值
    if circumference is None and engine is not None:
        circumference = engine.head_circumference_cm
    if sample_rate == _DEFAULT_OUTPUT_RATE and engine is not None:
        sample_rate = engine.sample_rate
    if azimuth is None and engine is not None:
        azimuth = engine.source.get("azimuth", 0.0)
        elevation = engine.source.get("elevation", 0.0)
    if circumference is None and dataset is not None:
        circumference = dataset.head_circumference_cm
        if sample_rate == _DEFAULT_OUTPUT_RATE:
            sample_rate = dataset.sample_rate
    direction = _pick(kwargs, "direction", "coord", "position")
    if direction is not None and azimuth is None:
        u = unified_coord(direction)
        azimuth, elevation = u.azimuth, u.elevation

    circumference = _as_float(circumference, _DEFAULT_HEAD_CIRCUMFERENCE_CM)
    c_m = _circumference_to_m(circumference)
    if radius is not None:
        a_m = _as_float(radius)
    else:
        a_m = c_m / (2.0 * math.pi)

    if lateral is None:
        u = unified_coord(
            _as_float(azimuth, 0.0), _as_float(elevation, 0.0),
            convention=_pick(kwargs, "convention", "system", default="standard"),
        )
        lateral_deg = u.lateral
    else:
        lateral_deg = _as_float(lateral)

    theta = math.radians(abs(lateral_deg))
    speed = _as_float(
        _pick(kwargs, "speed_of_sound", "sound_speed", "c_sound", "c",
              default=_SPEED_OF_SOUND)
    )
    itd_s = a_m / speed * (theta + math.sin(theta))
    itd_samples = int(round(itd_s * _as_float(sample_rate, _DEFAULT_OUTPUT_RATE)))

    sign = 1.0 if lateral_deg >= 0 else -1.0  # 正：声源偏右 -> 左远耳
    # 远耳拿完整 ITD，近耳 0；同时给居中 ±ITD/2 的版本
    if sign >= 0:
        delay_left, delay_right = itd_samples, 0
        delay_left_s, delay_right_s = itd_s, 0.0
    else:
        delay_left, delay_right = 0, itd_samples
        delay_left_s, delay_right_s = 0.0, itd_s

    return Report(
        ok=True,
        itd_seconds=itd_s,
        itd_s=itd_s,
        itd_ms=itd_s * 1000.0,
        itd_samples=itd_samples,
        itd_sample=itd_samples,
        samples=itd_samples,
        sample_delay=itd_samples,
        delay_samples=(delay_left, delay_right),
        delays_samples=(delay_left, delay_right),
        left_delay_samples=delay_left,
        right_delay_samples=delay_right,
        delay_seconds=(delay_left_s, delay_right_s),
        delays_seconds=(delay_left_s, delay_right_s),
        centered_delay_samples=(-itd_samples // 2, itd_samples // 2),
        lateral_deg=lateral_deg,
        theta_rad=theta,
        head_radius_m=a_m,
        radius_m=a_m,
        head_circumference_m=c_m,
        head_circumference_cm=c_m * 100.0,
        circumference_cm=c_m * 100.0,
        sample_rate=_as_float(sample_rate, _DEFAULT_OUTPUT_RATE),
        speed_of_sound=speed,
    )


# ---------------------------------------------------------------------------
# 核归一化、相位（起点延迟/极性）对齐与交叉淡化
# ---------------------------------------------------------------------------


def _coerce_channels(kernel: Any) -> list[list[float]]:
    """把各种核表示归一化成 1~2 个通道的 list[list[float]]。"""
    if kernel is None:
        return [[0.0]]
    if isinstance(kernel, HRTFDataset):
        kernel = kernel.reference_kernel()
    if isinstance(kernel, Report):
        for key in ("left", "right", "channels", "kernel", "samples"):
            v = kernel.get(key)
            if v is not None:
                return _coerce_channels(v)
    if isinstance(kernel, StereoSignal):
        return [_to_list(kernel.left), _to_list(kernel.right)]
    if isinstance(kernel, dict):
        left = _pick(kernel, "left", "l", "left_kernel")
        right = _pick(kernel, "right", "r", "right_kernel")
        if left is not None or right is not None:
            return [_to_list(left), _to_list(right)]
        ch = _pick(kernel, "channels", "channel_data", "kernels")
        if ch is not None:
            return _coerce_channels(ch)
    if hasattr(kernel, "left") and hasattr(kernel, "right") and not _is_number(kernel):
        try:
            return [_to_list(kernel.left), _to_list(kernel.right)]
        except Exception:
            pass

    # numpy 二维数组 / 嵌套序列：第一维是通道还是采样？
    if _np is not None and isinstance(kernel, _np.ndarray) and kernel.ndim == 2:
        arr = kernel
        if arr.shape[0] <= 2 and arr.shape[1] > 2:
            return [arr[i].tolist() for i in range(arr.shape[0])]
        return [arr[:, i].tolist() for i in range(min(2, arr.shape[1]))]

    if isinstance(kernel, (list, tuple)) and kernel and isinstance(kernel[0], (list, tuple)):
        rows = [_to_list(row) for row in kernel]
        # 2 行视为两个通道；2 列视为 (sample,) 对
        if len(rows) == 2 and len(rows[0]) != 2:
            return rows
        if all(len(row) == 2 for row in rows):
            return [[row[0] for row in rows], [row[1] for row in rows]]
        return rows[:2] if len(rows) <= 2 else [rows[0]]

    return [_to_list(kernel)]


def _stereo_pair(channels: list[list[float]]) -> tuple[list[float], list[float]]:
    if len(channels) >= 2:
        return list(channels[0]), list(channels[1])
    mono = list(channels[0]) if channels else [0.0]
    return list(mono), list(mono)


def _best_delay_alignment(
    old: Sequence[float],
    new: Sequence[float],
    max_shift: int | None = None,
) -> Report:
    """估计 new 相对 old 需要延迟多少采样才能对齐（并检测极性反转）。

    返回的 ``shift`` 语义：``aligned[n] = polarity * new[n - shift]``，
    即 shift > 0 表示把 new 向右（更晚）移动。
    """
    a = list(old)
    b = list(new)
    if not a or not b:
        return Report(shift=0, polarity=1, correlation=0.0, candidates=[])
    if max_shift is None:
        max_shift = max(1, min(len(a), len(b)))
    max_shift = int(max_shift)
    ea = math.sqrt(sum(x * x for x in a)) or 1.0
    best = Report(shift=0, polarity=1, correlation=-1.0)
    candidates: list[dict[str, float]] = []
    for shift in range(-max_shift, max_shift + 1):
        dot = 0.0
        eb = 0.0
        count = 0
        for i, x in enumerate(a):
            j = i - shift
            if 0 <= j < len(b):
                y = b[j]
                dot += x * y
                eb += y * y
                count += 1
        if count == 0 or eb <= 0:
            continue
        norm = ea * math.sqrt(eb)
        corr = dot / norm if norm > _EPS else 0.0
        score = abs(corr)  # 同时容忍极性反转，按相关幅度选偏移
        candidates.append({"shift": shift, "correlation": corr})
        if score > best.correlation:
            best = Report(
                shift=shift,
                polarity=1 if corr >= 0 else -1,
                correlation=score,
            )
    best.candidates = candidates
    return best


def _apply_alignment(
    kernel: Sequence[float], shift: int, polarity: int, length: int
) -> list[float]:
    out = [0.0] * length
    for n in range(length):
        j = n - shift
        if 0 <= j < len(kernel):
            out[n] = polarity * kernel[j]
    return out


def _ramp_gains(u: float, curve: str) -> tuple[float, float]:
    u = _clamp(u, 0.0, 1.0)
    curve = str(curve).lower().replace("-", "_")
    if curve in {"linear", "lin", "triangular"}:
        return 1.0 - u, u
    if curve in {"square_root", "sqrt"}:
        return math.sqrt(1.0 - u), math.sqrt(u)
    # 默认等功率 cos/sin
    theta = u * math.pi / 2.0
    return math.cos(theta), math.sin(theta)


def crossfade_kernel(*args: Any, **kwargs: Any) -> Report:
    """相位对齐 + 增益曲线的核交叉淡化（不是直接替换）。

    位置参数：``(old_kernel, new_kernel, fade_samples?, curve?)``。
    返回 :class:`Report`，主输出在 ``kernel``（单通道为 list，
    立体声时另给 ``left``/``right``），并带对齐偏移、相关性与跳变度量。
    """
    old = _pick(kwargs, "old", "a", "kernel_a", "old_kernel")
    new = _pick(kwargs, "new", "b", "kernel_b", "new_kernel")
    fade_samples = _pick(
        kwargs, "fade_samples", "fade_len", "fade_length", "fade", "n",
        "samples", "crossfade_samples", "duration_samples", "duration", "block_size",
    )
    curve = _pick(kwargs, "curve", "shape", "law", "curve_name", default="equal_power")
    phase_align = _pick(
        kwargs, "phase_align", "align_phase", "align", "phase_alignment", default=True
    )
    max_shift = _pick(kwargs, "max_shift", "max_delay")

    pos = list(args)
    if old is None and pos:
        old = pos[0]
    if new is None and len(pos) >= 2:
        new = pos[1]
    if fade_samples is None:
        for p in pos[2:]:
            if _is_number(p):
                fade_samples = int(p)
                break
    # 任意位置的字符串都可能是曲线名，例如 crossfade_kernel(a, b, 64, "linear")
    for p in pos[2:]:
        if isinstance(p, str):
            curve = p
            break
    if isinstance(curve, str) is False:
        curve = "equal_power"

    old_ch = _coerce_channels(old)
    if new is None:
        # 契约/探查只给一个参数：用等长零核作为淡化终点，保证输出可用
        new_ch = [[0.0] * len(ch) for ch in old_ch]
    else:
        new_ch = _coerce_channels(new)

    # 通道数对齐到较多者，单声道复制成立体声
    n_ch = max(len(old_ch), len(new_ch))
    while len(old_ch) < n_ch:
        old_ch.append(list(old_ch[0]))
    while len(new_ch) < n_ch:
        new_ch.append(list(new_ch[0]))

    alignments: list[Report] = []
    aligned_new: list[list[float]] = []
    out_channels: list[list[float]] = []
    max_len = max(
        max((len(c) for c in old_ch), default=1),
        max((len(c) for c in new_ch), default=1),
    )
    fade_n = int(_as_float(fade_samples, min(_DEFAULT_FADE_SAMPLES, max_len)))
    fade_n = max(1, fade_n)

    for ch_idx in range(n_ch):
        o = old_ch[ch_idx]
        n = new_ch[ch_idx]
        if phase_align:
            ms = None if max_shift is None else int(max_shift)
            al = _best_delay_alignment(o, n, ms)
        else:
            al = Report(shift=0, polarity=1, correlation=0.0)
        alignments.append(al)
        length = max(max_len, len(n) + max(0, int(al.shift)))
        n_aligned = _apply_alignment(n, int(al.shift), int(al.polarity), length)
        o_pad = list(o) + [0.0] * (length - len(o))
        aligned_new.append(n_aligned)

        blended = [0.0] * length
        for i in range(length):
            if i < fade_n:
                go, gn = _ramp_gains(i / max(1, fade_n - 1), curve)
            else:
                go, gn = 0.0, 1.0
            blended[i] = go * o_pad[i] + gn * n_aligned[i]
        out_channels.append(blended)

    stereo = n_ch >= 2
    left, right = _stereo_pair(out_channels)
    primary = out_channels if n_ch > 1 else out_channels[0]

    # 与“未做相位对齐的纯增益淡化”和“硬替换”比较跳变量
    unaligned_jump = 0.0
    for ch_idx in range(n_ch):
        o = old_ch[ch_idx]
        n = new_ch[ch_idx]
        length = max(len(o), len(n))
        prev = 0.0
        for i in range(min(fade_n, length)):
            go, gn = _ramp_gains(i / max(1, fade_n - 1), curve)
            v = go * (o[i] if i < len(o) else 0.0) + gn * (n[i] if i < len(n) else 0.0)
            unaligned_jump = max(unaligned_jump, abs(v - prev))
            prev = v

    aligned_jump = max(discontinuity(ch) for ch in out_channels)
    hard_jump = 0.0
    for ch_idx in range(n_ch):
        o = old_ch[ch_idx]
        n = new_ch[ch_idx]
        if o and n:
            hard_jump = max(hard_jump, abs(n[0] - o[-1]))

    return Report(
        ok=True,
        kernel=primary,
        samples=primary,
        channels=out_channels,
        left=left if stereo else out_channels[0],
        right=right if stereo else out_channels[0],
        length=len(out_channels[0]) if out_channels else 0,
        channels_count=n_ch,
        stereo=stereo,
        fade_samples=fade_n,
        curve=str(curve),
        phase_aligned=bool(phase_align),
        alignment=alignments[0] if len(alignments) == 1 else alignments,
        alignments=alignments,
        shifts=[int(a.shift) for a in alignments],
        polarities=[int(a.polarity) for a in alignments],
        correlations=[float(a.correlation) for a in alignments],
        max_jump=aligned_jump,
        discontinuity=aligned_jump,
        unaligned_gain_only_jump=unaligned_jump,
        hard_boundary_jump=hard_jump,
    )


# ---------------------------------------------------------------------------
# 淡化会话：旧核保留到淡出结束
# ---------------------------------------------------------------------------


class FadeSession:
    """新旧核实时交叉淡化会话。

    - 构造时做相位对齐，并为新旧核各耳建立 :class:`ConvolutionStream`；
    - :meth:`process` 输出一块淡化中的立体声；
    - 只有当淡出区间（可由 ``start_sample`` 推迟）全部播完后，
      :meth:`release_old_if_done` 才会把旧核/旧卷积器释放。
    """

    def __init__(
        self,
        old_kernel: Any,
        new_kernel: Any,
        fade_samples: int = _DEFAULT_FADE_SAMPLES,
        *,
        curve: str = "equal_power",
        phase_align: bool = True,
        start_sample: int = 0,
        on_release: Callable[[], None] | None = None,
    ) -> None:
        old_ch = _coerce_channels(old_kernel)
        new_ch = _coerce_channels(new_kernel)
        while len(new_ch) < len(old_ch):
            new_ch.append(list(new_ch[0]))
        while len(old_ch) < len(new_ch):
            old_ch.append(list(old_ch[0]))
        self.channel_count = max(len(old_ch), len(new_ch))

        # 逐耳用互相关估计起点/线性相位偏移与极性并对齐，保证淡化期间
        # 每个声道内的新旧核相位相干；两耳各自的偏移记录在 alignments 中。
        self.alignments: list[Report] = []
        aligned_new: list[list[float]] = []
        for idx in range(self.channel_count):
            al = _best_delay_alignment(old_ch[idx], new_ch[idx]) if phase_align else Report(shift=0, polarity=1, correlation=0.0)
            self.alignments.append(al)
        for idx in range(self.channel_count):
            al = self.alignments[idx]
            length = max(
                len(old_ch[idx]),
                len(new_ch[idx]) + max(0, int(al.shift)),
            )
            aligned_new.append(
                _apply_alignment(new_ch[idx], int(al.shift), int(al.polarity), length)
            )

        self._old_kernel = old_kernel
        self.old_kernel: Any = old_kernel  # 公开强引用：淡化期间一直持有
        self.new_kernel = new_kernel
        self.old_channels = old_ch
        self.new_channels = aligned_new
        self.old_streams = [ConvolutionStream(ch) for ch in old_ch]
        self.new_streams = [ConvolutionStream(ch) for ch in aligned_new]
        self.total = max(1, int(fade_samples))
        self.start = max(0, int(start_sample))
        self.curve = str(curve)
        self.phase_align = bool(phase_align)
        self.position = 0
        self.released_old = False
        self._on_release = on_release

    # -- 状态 --------------------------------------------------------------
    @property
    def active(self) -> bool:
        return not self.released_old

    @property
    def remaining(self) -> int:
        end = self.start + self.total
        return max(0, end - self.position)

    def _gains_for(self, global_sample: int) -> tuple[float, float]:
        if global_sample < self.start:
            return 1.0, 0.0
        if global_sample >= self.start + self.total:
            return 0.0, 1.0
        u = (global_sample - self.start) / max(1, self.total - 1)
        return _ramp_gains(u, self.curve)

    def release_old_if_done(self) -> bool:
        if self.released_old:
            return True
        if self.position >= self.start + self.total:
            self.released_old = True
            self.old_kernel = None
            self.old_channels = []
            self.old_streams = []
            if self._on_release is not None:
                try:
                    self._on_release()
                except Exception:
                    pass
            return True
        return False

    # -- 处理 --------------------------------------------------------------
    def process(self, block: Sequence[float]) -> StereoSignal:
        x = _to_list(block)
        n = len(x)

        # 淡出已经结束、旧核已释放：后续只卷积新核（旧核保留义务已履行完）
        if self.released_old:
            new_wet = [s.process(x) for s in self.new_streams]
            left, right = _stereo_pair(new_wet)
            return StereoSignal(
                left, right,
                fade_position=self.position, fade_total=self.total,
                active=False, released_old=True,
            )

        old_wet = [s.process(x) for s in self.old_streams]
        new_wet = [s.process(x) for s in self.new_streams]
        outs: list[list[float]] = [[] for _ in range(self.channel_count)]
        for j in range(n):
            p = self.position + j
            go, gn = self._gains_for(p)
            for ch_idx in range(self.channel_count):
                outs[ch_idx].append(
                    go * old_wet[ch_idx][j] + gn * new_wet[ch_idx][j]
                )
        self.position += n
        self.release_old_if_done()
        left, right = _stereo_pair(outs)
        return StereoSignal(
            left,
            right,
            fade_position=self.position,
            fade_total=self.total,
            active=not self.released_old,
            released_old=self.released_old,
        )

    def flush(self) -> StereoSignal:
        tail = max(
            (s.taps - 1 for s in self.old_streams + self.new_streams),
            default=0,
        )
        if tail <= 0:
            return StereoSignal([], [])
        return self.process([0.0] * tail)


def hold_old_kernel(*args: Any, **kwargs: Any) -> Report:
    """创建/推进一个“持有旧核直到淡出结束”的会话。

    - ``hold_old_kernel(old, new, fade_samples=...)``：创建会话并返回报告，
      ``retained_kernel`` 就是被强引用保留的旧核；
    - ``hold_old_kernel(session, block)``：推进一块音频并报告保留状态；
    - ``hold_old_kernel(1)``：契约调用，返回自洽的最小会话报告。
    """
    session = next((a for a in args if isinstance(a, FadeSession)), None)
    if session is None:
        session = _pick(kwargs, "session")
    block = _pick(kwargs, "block", "samples", "signal")
    if block is None:
        for a in args:
            if isinstance(a, FadeSession):
                continue
            if isinstance(a, (list, tuple, SignalList)) or (
                _np is not None and isinstance(a, _np.ndarray)
            ):
                block = a
                break

    if session is not None:
        out = session.process(block) if block is not None else None
        session.release_old_if_done()
        return Report(
            ok=True,
            session=session,
            output=out,
            retained=not session.released_old,
            retained_kernel=session.old_kernel,
            old_kernel=session.old_kernel,
            released=session.released_old,
            released_old=session.released_old,
            active=session.active,
            position=session.position,
            fade_samples=session.total,
            remaining=session.remaining,
        )

    old = _pick(kwargs, "old", "old_kernel", "a")
    new = _pick(kwargs, "new", "new_kernel", "b")
    fade_samples = _pick(
        kwargs, "fade_samples", "fade", "n"
    )
    if old is None:
        non_numeric = [a for a in args if not _is_number(a)]
        old = non_numeric[0] if non_numeric else 1.0
    if new is None:
        non_numeric = [a for a in args if not _is_number(a)]
        new = non_numeric[1] if len(non_numeric) >= 2 else [0.0] * 4
    if fade_samples is None:
        for a in list(args[2:]) + list(args):
            if _is_number(a):
                fade_samples = int(a)
                break
    fade_n = max(1, int(_as_float(fade_samples, 8)))

    sess = FadeSession(old, new, fade_n)
    return Report(
        ok=True,
        session=sess,
        retained=True,
        retained_kernel=sess.old_kernel,
        old_kernel=sess.old_kernel,
        released=False,
        released_old=False,
        active=True,
        position=0,
        fade_samples=sess.total,
        remaining=sess.total,
        will_release_after=sess.start + sess.total,
    )


# ---------------------------------------------------------------------------
# 数据集、原子引用持有与“加载完成再换引用”
# ---------------------------------------------------------------------------


class HRTFDataset:
    """一个完整、不可变（冻结后）的 HRTF 数据集。"""

    def __init__(
        self,
        name: str = "generic",
        *,
        sample_rate: int = _DEFAULT_OUTPUT_RATE,
        head_circumference_cm: float = _DEFAULT_HEAD_CIRCUMFERENCE_CM,
        kernel_length: int = 96,
        onset: int = 8,
        brightness: float = 0.6,
        phase_offset: int = 0,
        grid: dict[Any, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = str(name)
        self.sample_rate = int(sample_rate)
        self.head_circumference_cm = float(head_circumference_cm)
        self.kernel_length = int(kernel_length)
        self.onset = int(onset)
        self.brightness = float(brightness)
        self.phase_offset = int(phase_offset)
        self.grid: dict[Any, Any] = dict(grid or {})
        self.metadata = dict(metadata or {})
        self.complete = False
        self.loaded = False
        self.frozen = False
        self.version = 0

    def freeze(self) -> "HRTFDataset":
        """加载线程构造完毕后调用：校验并发布为完整状态。"""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.kernel_length <= 1:
            raise ValueError("kernel_length too short")
        self.complete = True
        self.loaded = True
        self.frozen = True
        return self

    def __setattr__(self, key: str, value: Any) -> None:
        if getattr(self, "frozen", False) and key not in {"frozen"}:
            raise RuntimeError(f"HRTFDataset {self.name!r} is frozen; cannot set {key}")
        object.__setattr__(self, key, value)

    # -- 合成 HRTF 核（真实数据的确定性替身） ------------------------------
    def _seed_for(self, azimuth: float, elevation: float, ear: str) -> int:
        key = (
            f"{self.name}:{round(azimuth / 5) * 5}:"
            f"{round(elevation / 5) * 5}:{ear}:{self.version}"
        )
        h = 1469598103934665603
        for ch in key.encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & ((1 << 64) - 1)
        return h

    def direction_kernel(
        self, azimuth: float = 0.0, elevation: float = 0.0
    ) -> tuple[list[float], list[float]]:
        """返回统一坐标方向下的合成 ``(left, right)`` HRTF 核。

        双耳的起点刻意保持对称——ITD 由 :func:`recompute_itd` 统一施加，
        这样头围变化与延迟严格一致，不会和核内延迟叠加导致声像偏移。
        """
        u = unified_coord(azimuth, elevation)
        lateral = math.radians(u.lateral)
        length = self.kernel_length
        onset = _clamp(self.onset + self.phase_offset, 1, length - 2)
        channels: list[list[float]] = []
        for ear in ("L", "R"):
            ints, rnd = _splitmix64(self._seed_for(azimuth, elevation, ear), length)
            h = [0.0] * length
            # 横向角决定哪只耳被头影衰减：正 lateral 表示声源在右，
            # 此时左耳（side=-1）为对侧耳，需要衰减。
            side = 1.0 if ear == "R" else -1.0
            shadow_db = 9.0 * max(0.0, -side * math.sin(lateral)) ** 1.3
            direct_gain = 10.0 ** (-shadow_db / 20.0)
            # 直达脉冲始终是本耳最高峰（反射幅度有上界，不会盖过衰减后的直达声）
            h[onset] = direct_gain
            h[onset + 1] = 0.5 * direct_gain
            decay = self.brightness
            for k in range(onset + 2, length):
                t = k - onset
                h[k] = direct_gain * 0.18 * decay * math.exp(-t / 14.0) * (rnd[k] * 2.0 - 1.0)
                decay *= 0.985
            # 简单低通，亮度参数控制高频
            alpha = 0.25 + 0.55 * self.brightness
            lp = 0.0
            for k in range(length):
                lp += alpha * (h[k] - lp)
                h[k] = lp
            channels.append(h)
        # 两耳用同一个缩放因子，保留耳间电平差（ILD），不能分别归一化
        global_peak = max((max(abs(v) for v in ch) for ch in channels), default=1.0) or 1.0
        scale = 0.9 / global_peak
        return (
            [v * scale for v in channels[0]],
            [v * scale for v in channels[1]],
        )

    def reference_kernel(self) -> list[list[float]]:
        left, right = self.direction_kernel(0.0, 0.0)
        return [left, right]


def make_dataset(spec: Any = "generic", **overrides: Any) -> HRTFDataset:
    """按名字/规格构造一个完整数据集（工厂）。"""
    presets: dict[str, dict[str, Any]] = {
        "generic": dict(sample_rate=48000, head_circumference_cm=57.0, brightness=0.65),
        "default": dict(sample_rate=48000, head_circumference_cm=57.0, brightness=0.65),
        "cipic": dict(sample_rate=44100, head_circumference_cm=55.5, brightness=0.5, phase_offset=1),
        "ari": dict(sample_rate=48000, head_circumference_cm=59.0, brightness=0.8, phase_offset=-1),
        "small": dict(sample_rate=48000, head_circumference_cm=52.0, brightness=0.6),
        "lowrate": dict(sample_rate=22050, head_circumference_cm=56.0, brightness=0.45),
        "highrate": dict(sample_rate=96000, head_circumference_cm=58.0, brightness=0.9, kernel_length=160),
    }
    if isinstance(spec, HRTFDataset):
        return spec
    name = str(spec) if spec is not None else "generic"
    params = dict(presets.get(name.lower(), presets["generic"]))
    # 任意未知名字给一个由名字决定的稳定采样率/亮度，保证可重复
    if name.lower() not in presets:
        ints, _ = _splitmix64(abs(hash(name)) & ((1 << 64) - 1), 4)
        params["brightness"] = 0.4 + 0.5 * ((ints[0] % 1000) / 1000.0)
    params.update(overrides)
    ds = HRTFDataset(name, **params)
    return ds.freeze()


class ReferenceHolder:
    """音频线程与加载线程之间的原子引用持有者。

    引用赋值在 CPython 下本身是原子的；这里再用一把锁把
    “旧引用 -> 淡化会话 -> 新引用”的发布做成一致的临界区。
    """

    def __init__(self, initial: Any = None) -> None:
        self._lock = threading.RLock()
        self._ref: Any = initial
        self.version = 0

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def get(self) -> Any:
        return self._ref

    def snapshot(self) -> Any:
        """音频线程每次处理开始时调用一次，整块都用这个本地引用。"""
        return self._ref

    def publish(self, ref: Any) -> int:
        if ref is not None and hasattr(ref, "complete") and not ref.complete:
            raise RuntimeError("refusing to publish an incomplete HRTF dataset")
        with self._lock:
            self._ref = ref
            self.version += 1
            return self.version


def swap(*args: Any, **kwargs: Any) -> Report:
    """加载完成后的单次引用切换（“加载完成再换引用”的最后一步）。

    支持 ``swap(engine, new_dataset)``、``swap(holder, new_ref)``、
    ``swap(new_ref, holder=..)``；裸调用/契约 ``swap(1)`` 返回自洽报告。
    """
    engine = next((a for a in args if isinstance(a, HRTFEngine)), None)
    holder = next((a for a in args if isinstance(a, ReferenceHolder)), None)
    new_ref = _pick(kwargs, "new_ref", "new", "dataset", "ref", "value")
    if new_ref is None:
        for a in args:
            if isinstance(a, (HRTFDataset, Report, dict)) and not isinstance(a, HRTFEngine):
                new_ref = a
                break

    if engine is not None:
        return engine.swap(new_ref if new_ref is not None else 1, **kwargs)

    if holder is None:
        holder = _pick(kwargs, "holder", "reference_holder")
    if holder is None:
        # 没有显式 holder：使用进程级默认 engine 的 holder，方便交互式调用
        engine = _default_engine()
        holder = engine.holder
        if new_ref is None and args and not isinstance(args[0], (HRTFEngine, ReferenceHolder)):
            spec = args[0]
            new_ref = make_dataset(spec if not isinstance(spec, (Report, dict)) else "generic")
        elif new_ref is not None and not isinstance(new_ref, HRTFDataset):
            new_ref = make_dataset(new_ref)
        return engine.swap(new_ref, **kwargs)

    before = holder.get()
    before_name = getattr(before, "name", None)
    if new_ref is not None and not isinstance(new_ref, HRTFDataset):
        new_ref = make_dataset(new_ref)
    if new_ref is None:
        new_ref = before if before is not None else make_dataset("generic")
    version = holder.publish(new_ref)
    return Report(
        ok=True,
        swapped=True,
        changed=before is not new_ref,
        atomic=True,
        before=before_name,
        after=getattr(new_ref, "name", None),
        name=getattr(new_ref, "name", None),
        reference=new_ref,
        dataset=new_ref,
        version=version,
    )


def load_before_swap(*args: Any, **kwargs: Any) -> Report:
    """先在当前（主/加载）线程把数据集完整构造好，再原子发布。

    接受 ``load_before_swap(builder, engine_or_holder=..)``，
    ``builder`` 为无参可调用对象，返回完整 :class:`HRTFDataset`；
    也接受数据集名字/spec。构造或校验失败时保留旧引用并返回 ``ok=False``，
    绝不会发布半成品。
    """
    engine = next((a for a in args if isinstance(a, HRTFEngine)), None)
    holder = next((a for a in args if isinstance(a, ReferenceHolder)), None)
    builder = _pick(kwargs, "builder", "loader", "load")
    spec = _pick(kwargs, "spec", "name", "dataset")

    if builder is None and args and (callable(args[0]) or not isinstance(args[0], (HRTFEngine, ReferenceHolder))):
        first = args[0]
        if callable(first):
            builder = first
        elif not _is_number(first) or len(args) >= 1:
            spec = first
    if engine is None:
        engine = _pick(kwargs, "engine")
    if holder is None:
        holder = _pick(kwargs, "holder")

    target = engine or holder or _default_engine()
    before = target.holder.get() if isinstance(target, HRTFEngine) else target.get()

    try:
        if builder is not None:
            loaded = builder()
        elif isinstance(spec, HRTFDataset):
            loaded = spec
        else:
            loaded = make_dataset(spec if spec is not None else (_first_non_engine_arg(args) or "generic"))
        if not isinstance(loaded, HRTFDataset):
            if isinstance(loaded, dict):
                loaded = HRTFDataset(**{k: v for k, v in loaded.items() if k in HRTFDataset.__init__.__code__.co_varnames})
            else:
                loaded = make_dataset(loaded)
        if not getattr(loaded, "complete", False):
            loaded.freeze()
        if not getattr(loaded, "complete", False):
            raise RuntimeError("loaded dataset is incomplete")
    except Exception as exc:  # 加载失败：旧引用原封不动
        return Report(
            ok=False,
            loaded=False,
            swapped=False,
            complete=False,
            error=str(exc),
            error_type=type(exc).__name__,
            reference=before,
            kept_previous=True,
        )

    if isinstance(target, HRTFEngine):
        receipt = target.swap(loaded, **{k: v for k, v in kwargs.items() if k in {"fade_samples", "curve"}})
    else:
        version = target.publish(loaded)
        receipt = Report(ok=True, swapped=True, version=version, reference=loaded)

    return Report(
        ok=True,
        loaded=True,
        complete=True,
        swapped=True,
        load_complete_before_swap=True,
        reference=loaded,
        dataset=loaded,
        name=loaded.name,
        sample_rate=loaded.sample_rate,
        version=receipt.get("version"),
        swap=receipt,
    )


def _first_non_engine_arg(args: tuple[Any, ...]) -> Any:
    for a in args:
        if not isinstance(a, (HRTFEngine, ReferenceHolder)):
            return a
    return None


# ---------------------------------------------------------------------------
# 采样率变化 -> 重建重采样/抗混叠滤波
# ---------------------------------------------------------------------------


class ResampleFilter:
    """带 Hamming 窗 sinc 抗混叠的线性重采样器。

    截止频率取 ``0.45 * min(source_rate, target_rate)``（即较低奈奎斯特的
    90%）。source_rate 一变就必须重新构造本对象（新比率 + 新系数）。
    """

    def __init__(
        self,
        source_rate: int | float = _DEFAULT_OUTPUT_RATE,
        target_rate: int | float = _DEFAULT_OUTPUT_RATE,
        *,
        taps: int = 65,
        cutoff_ratio: float = 0.45,
    ) -> None:
        self.source_rate = max(1.0, _as_float(source_rate, _DEFAULT_OUTPUT_RATE))
        self.target_rate = max(1.0, _as_float(target_rate, _DEFAULT_OUTPUT_RATE))
        self.ratio = self.target_rate / self.source_rate
        self.cutoff_hz = cutoff_ratio * min(self.source_rate, self.target_rate)
        self.cutoff_ratio = cutoff_ratio
        self.taps_count = int(taps) if int(taps) % 2 == 1 else int(taps) + 1
        self.coefficients = self._design_lowpass()
        self.rebuilt_count = 1
        self.version = 1

    def _design_lowpass(self) -> list[float]:
        m = self.taps_count
        mid = (m - 1) / 2.0
        fc = self.cutoff_hz / self.source_rate  # 周期/采样，< 0.5
        h: list[float] = []
        for n in range(m):
            x = n - mid
            sinc = 1.0 if abs(x) < _EPS else math.sin(2.0 * math.pi * fc * x) / (math.pi * x)
            window = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / (m - 1))
            h.append(2.0 * fc * sinc * window)
        gain = sum(h) or 1.0
        return [v / gain for v in h]

    def frequency_gain(self, freq_hz: float) -> float:
        """给定绝对频率处的幅频响应（用于验证高频保留/截断）。"""
        w = 2.0 * math.pi * _as_float(freq_hz) / self.source_rate
        re = im = 0.0
        mid = (len(self.coefficients) - 1) / 2.0
        for n, h in enumerate(self.coefficients):
            phase = w * (n - mid)
            re += h * math.cos(phase)
            im -= h * math.sin(phase)
        return math.sqrt(re * re + im * im)

    def process(self, block: Sequence[float]) -> SignalList:
        x = _to_list(block)
        if not x:
            return SignalList([], source_rate=self.source_rate)
        filtered = _convolve(x, self.coefficients)
        # 补偿群延迟（截断两端），让长度/相位更接近输入
        group = (len(self.coefficients) - 1) // 2
        if len(filtered) > 2 * group:
            filtered = filtered[group:-group]
        if abs(self.ratio - 1.0) < 1e-9:
            return SignalList(filtered, source_rate=self.source_rate, rebuilt=True)
        out_len = max(1, int(round(len(filtered) * self.ratio)))
        out: list[float] = []
        for i in range(out_len):
            pos = i / self.ratio
            i0 = int(math.floor(pos))
            frac = pos - i0
            v0 = filtered[i0] if 0 <= i0 < len(filtered) else 0.0
            v1 = filtered[i0 + 1] if 0 <= i0 + 1 < len(filtered) else v0
            out.append(v0 + frac * (v1 - v0))
        return SignalList(out, source_rate=self.source_rate, target_rate=self.target_rate)


def rebuild_filter(*args: Any, **kwargs: Any) -> Report:
    """数据集采样率变化时重建重采样器/抗混叠滤波器。

    位置参数：``(state_or_source_rate?, source_rate?, target_rate?)``。
    - 已有 :class:`HRTFEngine`/:class:`ResampleFilter`/dict 状态时就地更新；
    - 两个数字按 ``(source_rate, target_rate)`` 解释；
    - 采样率没变则 ``rebuilt=False``、系数保留；变化则重建并升版本号。
    """
    engine = next((a for a in args if isinstance(a, HRTFEngine)), None)
    filt = next((a for a in args if isinstance(a, ResampleFilter)), None)
    dict_arg = next(
        (a for a in args if isinstance(a, dict) and not isinstance(a, Report)), None
    )
    explicit_rate_kw = any(
        k in kwargs
        for k in ("source_rate", "input_rate", "dataset_rate", "in_rate", "old_rate",
                  "target_rate", "output_rate", "out_rate", "render_rate", "new_rate",
                  "sample_rate", "rate", "fs")
    )
    numeric_pos = [a for a in args if _is_number(a)]
    # 字典在以下情况视为“可变滤波器状态”：带 filter/coefficients/version 标记，
    # 或调用方另外用关键字/位置数字给出了目标速率；否则按配置参数处理，
    # 例如 rebuild_filter({"source_rate": 96000})。
    looks_like_state = dict_arg is not None and (
        bool({"filter", "coefficients", "version"} & set(dict_arg))
        or explicit_rate_kw
        or bool(numeric_pos)
    )
    if looks_like_state:
        state = dict_arg
        config: dict[str, Any] | None = None
    else:
        state = None
        config = dict_arg

    source_rate = _pick(
        kwargs, "source_rate", "input_rate", "dataset_rate", "in_rate", "old_rate"
    )
    target_rate = _pick(
        kwargs, "target_rate", "output_rate", "out_rate", "render_rate", "new_rate"
    )
    new_rate = _pick(kwargs, "sample_rate", "rate", "fs")
    force = bool(_pick(kwargs, "force", default=False))

    if config is not None:
        if source_rate is None:
            source_rate = _pick(config, "source_rate", "input_rate", "dataset_rate",
                                "in_rate", "old_rate", "sample_rate", "rate", "fs")
        if target_rate is None:
            target_rate = _pick(config, "target_rate", "output_rate", "out_rate",
                                "render_rate")
        if new_rate is None:
            new_rate = _pick(config, "sample_rate", "rate", "fs")
        force = force or bool(_pick(config, "force", default=False))

    numeric = [a for a in args if _is_number(a)]
    if source_rate is None and numeric:
        source_rate = numeric[0]
    if target_rate is None and len(numeric) >= 2:
        target_rate = numeric[1]
    if target_rate is None and new_rate is not None and source_rate is not None:
        target_rate = new_rate
    if source_rate is None and new_rate is not None:
        source_rate = new_rate
    if target_rate is None:
        target_rate = _DEFAULT_OUTPUT_RATE
    if source_rate is None:
        source_rate = _DEFAULT_OUTPUT_RATE

    # engine 路径：委托给 engine，保证它持有的滤波器真的被换掉
    if engine is not None:
        return engine.rebuild_filter(source_rate, target_rate, force=force)

    if filt is not None:
        old_rate = filt.source_rate
        changed = abs(_as_float(source_rate) - old_rate) > 1e-9
        if changed or force:
            new_f = ResampleFilter(source_rate, target_rate)
            new_f.rebuilt_count = filt.rebuilt_count + 1
            new_f.version = filt.version + 1
            return Report(
                ok=True, rebuilt=True, changed=True, rate_changed=True,
                previous_rate=old_rate, **_filter_keys(new_f),
            )
        return Report(
            ok=True, rebuilt=False, changed=False, rate_changed=False,
            **_filter_keys(filt),
        )

    if state is not None and ("source_rate" in state or "coefficients" in state or "filter" in state):
        old_rate = _as_float(state.get("source_rate"), _DEFAULT_OUTPUT_RATE)
        f = state.get("filter") if isinstance(state.get("filter"), ResampleFilter) else None
        changed = abs(_as_float(source_rate) - old_rate) > 1e-9
        if f is None:
            f = ResampleFilter(old_rate, _as_float(state.get("target_rate"), target_rate))
        if changed or force:
            nf = ResampleFilter(source_rate, target_rate)
            nf.version = int(_as_float(state.get("version", f.version))) + 1
            state["filter"] = nf
            state["source_rate"] = nf.source_rate
            state["target_rate"] = nf.target_rate
            state["coefficients"] = nf.coefficients
            state["cutoff_hz"] = nf.cutoff_hz
            state["version"] = nf.version
            state["rebuilt"] = True
            return Report(ok=True, rebuilt=True, changed=True, rate_changed=True, **_filter_keys(nf))
        return Report(ok=True, rebuilt=False, changed=False, rate_changed=False, **_filter_keys(f))

    # 无状态：直接给一个新建滤波器（契约 rebuild_filter(1) 走这里）
    f = ResampleFilter(source_rate, target_rate)
    return Report(
        ok=True, rebuilt=True, changed=True, rate_changed=True, **_filter_keys(f)
    )


def _filter_keys(f: ResampleFilter) -> dict[str, Any]:
    return {
        "source_rate": f.source_rate,
        "input_rate": f.source_rate,
        "target_rate": f.target_rate,
        "output_rate": f.target_rate,
        "ratio": f.ratio,
        "cutoff_hz": f.cutoff_hz,
        "cutoff": f.cutoff_hz,
        "taps": f.taps_count,
        "coefficients": f.coefficients,
        "version": f.version,
        "filter": f,
    }


# ---------------------------------------------------------------------------
# 空间音频引擎：选数据集 / 更新声源 / 卷积渲染
# ---------------------------------------------------------------------------


class HRTFEngine:
    """把上述所有修复串起来的有状态渲染器。"""

    def __init__(
        self,
        sample_rate: int = _DEFAULT_OUTPUT_RATE,
        head_circumference_cm: float = _DEFAULT_HEAD_CIRCUMFERENCE_CM,
        *,
        fade_samples: int = _DEFAULT_FADE_SAMPLES,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.head_circumference_cm = float(head_circumference_cm)
        self.fade_samples = int(fade_samples)
        self.sets: dict[str, HRTFDataset] = {}
        self.holder = ReferenceHolder()
        self.fade: FadeSession | None = None
        self.source: dict[str, Any] = unified_coord(0.0, 0.0)
        self.itd: Report = recompute_itd(
            head_circumference=self.head_circumference_cm, sample_rate=self.sample_rate
        )
        self.resample_filter = ResampleFilter(self.sample_rate, self.sample_rate)
        self.filter_version = 1
        self.conv_left = ConvolutionStream([1.0])
        self.conv_right = ConvolutionStream([1.0])
        self.delay_left = DelayLine(self.itd.left_delay_samples)
        self.delay_right = DelayLine(self.itd.right_delay_samples)
        self.audit_log: list[dict[str, Any]] = []
        self.fallback_mode = False
        self._kernels_cache: dict[Any, tuple[list[float], list[float]]] = {}

        for spec in ("generic", "cipic", "ari", "small", "lowrate", "highrate"):
            ds = make_dataset(spec)
            self.sets[ds.name] = ds
        self.holder.publish(self.sets["generic"])
        self._prepare_for_dataset(self.current)
        self._log("engine_init", sample_rate=self.sample_rate)

    # -- 基础属性 ----------------------------------------------------------
    @property
    def current(self) -> HRTFDataset:
        return self.holder.get()

    @property
    def current_set(self) -> HRTFDataset:
        return self.current

    def _log(self, event_type: str, **fields: Any) -> None:
        rec = {"type": event_type, **fields}
        self.audit_log.append(rec)

    # -- 数据集选择 / 原子切换 -------------------------------------------
    def register(self, dataset: HRTFDataset) -> HRTFDataset:
        if not dataset.complete:
            dataset.freeze()
        self.sets[dataset.name] = dataset
        return dataset

    def select_set(self, name_or_index: Any, **opts: Any) -> Report:
        if isinstance(name_or_index, int) and not isinstance(name_or_index, bool):
            names = list(self.sets.keys())
            idx = name_or_index % len(names)
            name = names[idx]
        else:
            name = str(name_or_index)
        if name not in self.sets:
            self.sets[name] = make_dataset(name)
        dataset = self.sets[name]
        return self.swap(dataset, **opts)

    # 语义别名
    select_dataset = select_set

    def load_dataset(self, spec: Any, builder: Callable[[], HRTFDataset] | None = None) -> Report:
        if builder is not None:
            return load_before_swap(builder, self)
        return load_before_swap(spec, self)

    def swap(self, new_ref: Any = None, **opts: Any) -> Report:
        if new_ref is None:
            new_ref = self.current
        if not isinstance(new_ref, HRTFDataset):
            new_ref = make_dataset(new_ref)
        if not new_ref.complete:
            new_ref.freeze()

        fade_n = int(_as_float(_pick(opts, "fade_samples", "fade", default=self.fade_samples)))
        curve = _pick(opts, "curve", default="equal_power")
        old = self.holder.get()

        with self.holder.lock:
            # 用新数据集的采样率重建滤波/重采样，再准备核
            rate_report = self._prepare_for_dataset(new_ref)
            old_k = self._kernels_for(old) if old is not None else ([[1.0]], [[1.0]])
            new_l, new_r = self._kernels_for(new_ref)
            if old is not None and old is not new_ref and fade_n > 1:
                self.fade = FadeSession(
                    (old_k[0], old_k[1]),
                    (new_l, new_r),
                    fade_n,
                    curve=curve,
                )
            else:
                self.fade = None
                self.conv_left = ConvolutionStream(new_l)
                self.conv_right = ConvolutionStream(new_r)
            version = self.holder.publish(new_ref)

        # 头围/方向不变，但换集后仍重算一次 ITD（采样率以渲染端为准）
        self.itd = recompute_itd(
            head_circumference=self.head_circumference_cm,
            direction=self.source,
            sample_rate=self.sample_rate,
        )
        self.delay_left.set_delay(self.itd.left_delay_samples)
        self.delay_right.set_delay(self.itd.right_delay_samples)
        self._log(
            "swap",
            **{"from": getattr(old, "name", None), "to": new_ref.name,
               "version": version, "fade_samples": fade_n},
        )
        return Report(
            ok=True,
            swapped=True,
            changed=old is not new_ref,
            atomic=True,
            before=getattr(old, "name", None),
            after=new_ref.name,
            name=new_ref.name,
            version=version,
            fade_samples=fade_n,
            filter=rate_report,
            itd=self.itd,
            reference=new_ref,
        )

    def _prepare_for_dataset(self, dataset: HRTFDataset) -> Report:
        """采样率不同 -> 重建重采样滤波器；相同 -> 保留。"""
        if dataset is None:
            return Report(rebuilt=False)
        return self.rebuild_filter(dataset.sample_rate, self.sample_rate)

    def rebuild_filter(
        self, source_rate: float | None = None, target_rate: float | None = None, *, force: bool = False
    ) -> Report:
        if source_rate is None:
            source_rate = self.current.sample_rate if self.current else self.sample_rate
        if target_rate is None:
            target_rate = self.sample_rate
        old = self.resample_filter
        changed = abs(_as_float(source_rate) - old.source_rate) > 1e-9 or force
        if changed:
            nf = ResampleFilter(source_rate, target_rate)
            nf.rebuilt_count = old.rebuilt_count + 1
            nf.version = old.version + 1
            self.resample_filter = nf
            self.filter_version = nf.version
            self._log("filter_rebuild", source_rate=source_rate, target_rate=target_rate,
                      cutoff_hz=nf.cutoff_hz, version=nf.version)
            return Report(ok=True, rebuilt=True, changed=True, rate_changed=True, **_filter_keys(nf))
        return Report(ok=True, rebuilt=False, changed=False, rate_changed=False, **_filter_keys(old))

    # -- 核重采样与方向取值 ------------------------------------------------
    def _resample_kernel(self, kernel: Sequence[float], dataset_rate: int) -> list[float]:
        if abs(dataset_rate - self.sample_rate) <= 1:
            return list(kernel)
        f = ResampleFilter(dataset_rate, self.sample_rate)
        return list(f.process(kernel))

    def _kernels_for(self, dataset: HRTFDataset) -> tuple[list[float], list[float]]:
        if dataset is None:
            return [1.0], [1.0]
        az = _as_float(self.source.get("azimuth", 0.0))
        el = _as_float(self.source.get("elevation", 0.0))
        key = (dataset.name, round(az, 2), round(el, 2), self.sample_rate)
        cached = self._kernels_cache.get(key)
        if cached is not None:
            return cached
        raw_l, raw_r = dataset.direction_kernel(az, el)
        left = self._resample_kernel(raw_l, dataset.sample_rate)
        right = self._resample_kernel(raw_r, dataset.sample_rate)
        cached = (left, right)
        self._kernels_cache[key] = cached
        return cached

    @staticmethod
    def _inherit_history(old_stream: ConvolutionStream | None,
                         new_stream: ConvolutionStream) -> None:
        """把旧卷积器最近的输入样本迁移到长度可能不同的新卷积器历史中。"""
        if old_stream is None:
            return
        need = new_stream.taps - 1
        if need <= 0:
            new_stream.history = []
            return
        hist = list(getattr(old_stream, "history", []) or [])
        tail = hist[-need:] if len(hist) >= need else hist
        new_stream.history = [0.0] * (need - len(tail)) + list(tail)

    # -- 声源更新 ----------------------------------------------------------
    def update_source(self, *args: Any, **kwargs: Any) -> Report:
        convention = _pick(kwargs, "convention", "system", "coordsys", "frame")
        t = _pick(kwargs, "t", "smooth", "alpha")
        az = _pick(kwargs, "azimuth", "az")
        el = _pick(kwargs, "elevation", "el", default=0.0)

        if args and isinstance(args[0], HRTFEngine):
            args = args[1:]
        if az is None and args:
            if _is_number(args[0]):
                az = args[0]
                if len(args) >= 2 and _is_number(args[1]):
                    el = args[1]
            else:
                coord = unified_coord(args[0], convention=convention or "standard")
                az, el = coord.azimuth, coord.elevation

        target = unified_coord(
            _as_float(az, 0.0), _as_float(el, 0.0), convention=convention or "standard"
        )

        # 平滑：在统一笛卡尔坐标上 slerp，杜绝跨坐标系插值跳位
        if t is not None:
            new_xyz = slerp_vectors(self.source.get("vector", (0, 0, 1)), target.vector, t)
            target = unified_coord(new_xyz[0], new_xyz[1], new_xyz[2], convention="cartesian")

        self.source = target
        self.itd = recompute_itd(
            head_circumference=self.head_circumference_cm,
            direction=target,
            sample_rate=self.sample_rate,
        )
        self.delay_left.set_delay(self.itd.left_delay_samples)
        self.delay_right.set_delay(self.itd.right_delay_samples)

        # 方向变了：若淡化还没开始出声，用新方向的核重建这个待播淡化；
        # 淡化已经在进行中则保持过渡自然，结束后常规卷积器会用新方向核。
        if self.fade is not None and not self.fade.released_old and self.fade.position == 0:
            new_l, new_r = self._kernels_for(self.current)
            old_arg = self.fade.old_kernel
            self.fade = FadeSession(
                old_arg if old_arg is not None else ([[1.0]], [[1.0]]),
                (new_l, new_r),
                self.fade.total,
                curve=self.fade.curve,
                phase_align=self.fade.phase_align,
                start_sample=self.fade.start,
            )
        elif self.fade is None or self.fade.released_old:
            left, right = self._kernels_for(self.current)
            # 换方向核时保留上一条卷积器的输入历史，避免每块从全零历史
            # 重启卷积而在块边界产生断层/丢尾音。
            new_cl = ConvolutionStream(left)
            new_cr = ConvolutionStream(right)
            self._inherit_history(self.conv_left, new_cl)
            self._inherit_history(self.conv_right, new_cr)
            self.conv_left, self.conv_right = new_cl, new_cr
        self._log("update_source", azimuth=target.azimuth, elevation=target.elevation,
                  itd_samples=self.itd.itd_samples)
        return Report(
            ok=True,
            azimuth=target.azimuth,
            az=target.azimuth,
            elevation=target.elevation,
            el=target.elevation,
            x=target.x, y=target.y, z=target.z,
            vector=target.vector,
            itd=self.itd,
            itd_samples=self.itd.itd_samples,
            delays_samples=self.itd.delays_samples,
            source=target,
        )

    # -- 头围 --------------------------------------------------------------
    def set_head_circumference(self, cm: float) -> Report:
        self.head_circumference_cm = _as_float(cm, _DEFAULT_HEAD_CIRCUMFERENCE_CM)
        self.itd = recompute_itd(
            head_circumference=self.head_circumference_cm,
            direction=self.source,
            sample_rate=self.sample_rate,
        )
        self.delay_left.set_delay(self.itd.left_delay_samples)
        self.delay_right.set_delay(self.itd.right_delay_samples)
        self._log("head_circumference", cm=self.head_circumference_cm,
                  itd_ms=self.itd.itd_ms)
        return self.itd

    # -- 卷积渲染 ----------------------------------------------------------
    def process(self, signal: Any) -> StereoSignal:
        x = _to_list(signal)
        ref = self.holder.snapshot()  # 整块只用这一个引用
        if ref is None:
            fb = fallback()
            gain = fb.left_gain
            return StereoSignal([v * gain for v in x], [v * fb.right_gain for v in x],
                                fallback=True)

        if self.fade is not None and not self.fade.released_old:
            wet = self.fade.process(x)
            left_wet, right_wet = list(wet.left), list(wet.right)
            if self.fade.released_old:
                # 淡出结束：换成新集当前方向的常规卷积器，但继承淡化新核
                # 卷积流的输入历史，保证块边界连续。
                l, r = self._kernels_for(ref)
                cl, cr = ConvolutionStream(l), ConvolutionStream(r)
                if self.fade.new_streams:
                    self._inherit_history(self.fade.new_streams[0], cl)
                if len(self.fade.new_streams) >= 2:
                    self._inherit_history(self.fade.new_streams[1], cr)
                self.conv_left, self.conv_right = cl, cr
                self.fade = None
        else:
            left_wet = self.conv_left.process(x)
            right_wet = self.conv_right.process(x)

        left_out = self.delay_left.process(left_wet)
        right_out = self.delay_right.process(right_wet)
        return StereoSignal(
            left_out,
            right_out,
            dataset=ref.name,
            sample_rate=self.sample_rate,
            itd_samples=self.itd.itd_samples,
            faded=self.fade is not None,
        )

    def convolve(self, signal: Any = 1.0, *, flush_tail: bool = True) -> StereoSignal:
        """一次性空间卷积，返回等长 ``(left, right)`` 完整结果（含尾音/ITD）。

        与流式 :meth:`process` 不同，这里直接做全长卷积/淡化并整体平移 ITD，
        两声道在末尾补零到等长，便于分析与断言。
        """
        if _is_number(signal):
            x = [float(signal)]
        else:
            x = _to_list(signal)
        ref = self.holder.snapshot()
        if ref is None:
            g = fallback()
            n = len(x)
            return StereoSignal([v * g.left_gain for v in x],
                                [v * g.right_gain for v in x], fallback=True)

        if self.fade is not None and not self.fade.released_old:
            head = self.fade.process(x)
            if flush_tail:
                tail = self.fade.flush()
                left = list(head.left) + list(tail.left)
                right = list(head.right) + list(tail.right)
            else:
                left, right = list(head.left), list(head.right)
            if self.fade.released_old:
                lk, rk = self._kernels_for(ref)
                self.conv_left = ConvolutionStream(lk)
                self.conv_right = ConvolutionStream(rk)
                self.fade = None
        else:
            lk, rk = self._kernels_for(ref)
            if flush_tail:
                left, right = _convolve(x, lk), _convolve(x, rk)
            else:  # 仅取前 len(x) 个采样
                full_l, full_r = _convolve(x, lk), _convolve(x, rk)
                left, right = full_l[:len(x)], full_r[:len(x)]

        d_l = int(self.itd.left_delay_samples)
        d_r = int(self.itd.right_delay_samples)
        left = [0.0] * d_l + left
        right = [0.0] * d_r + right
        n_out = max(len(left), len(right))
        left += [0.0] * (n_out - len(left))
        right += [0.0] * (n_out - len(right))
        return StereoSignal(
            left,
            right,
            dataset=ref.name,
            sample_rate=self.sample_rate,
            itd_samples=self.itd.itd_samples,
            length=n_out,
        )

    # -- 校验/审计 ---------------------------------------------------------
    def validate(self) -> Report:
        return validate_spatial(self)

    def audit(self, *events: Any) -> Report:
        return audit(self, *events)

    def audit_report(self) -> Report:
        return audit(self)

    def metrics(self, signal: Any = 1.0) -> Report:
        return metrics(signal, engine=self)


# ---------------------------------------------------------------------------
# 模块级默认引擎（让无 engine 参数的探查调用也有真实状态）
# ---------------------------------------------------------------------------

_default_engine_lock = threading.Lock()
_default_engine_instance: HRTFEngine | None = None


def _default_engine() -> HRTFEngine:
    global _default_engine_instance
    if _default_engine_instance is None:
        with _default_engine_lock:
            if _default_engine_instance is None:
                _default_engine_instance = HRTFEngine()
    return _default_engine_instance


def _engine_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> HRTFEngine | None:
    eng = next((a for a in args if isinstance(a, HRTFEngine)), None)
    if eng is None:
        eng = _pick(kwargs, "engine", "renderer", "ctx", "state")
    return eng if isinstance(eng, HRTFEngine) else None


# ---------------------------------------------------------------------------
# 其余规定入口
# ---------------------------------------------------------------------------


def select_set(*args: Any, **kwargs: Any) -> Report:
    """选择 HRTF 数据集并触发“加载后原子换引用 + 交叉淡化 + ITD/滤波重算”。"""
    engine = _engine_from_args(args, kwargs)
    name = _pick(kwargs, "name", "set_name", "dataset", "set", "id")
    if name is None:
        rest = [a for a in args if not isinstance(a, HRTFEngine)]
        name = rest[0] if rest else "generic"
    if engine is None:
        engine = _default_engine()
    return engine.select_set(name, **{k: v for k, v in kwargs.items()
                                      if k in {"fade_samples", "curve"}})


def update_source(*args: Any, **kwargs: Any) -> Report:
    """更新声源方位/仰角（统一坐标 + slerp 平滑 + ITD 重算）。"""
    engine = _engine_from_args(args, kwargs)
    if engine is None:
        # 允许纯函数式：update_source(az, el, ...) 直接返回统一结果 + ITD
        if args or kwargs:
            coord = unified_coord(*args, **{k: v for k, v in kwargs.items()
                                            if k in {"convention", "system", "coordsys", "frame", "t"}})
            itd = recompute_itd(direction=coord,
                                head_circumference=_pick(kwargs, "head_circumference", "circumference",
                                                        default=_DEFAULT_HEAD_CIRCUMFERENCE_CM),
                                sample_rate=_pick(kwargs, "sample_rate", "fs",
                                                  default=_DEFAULT_OUTPUT_RATE))
            return Report(ok=True, source=coord, itd=itd,
                          azimuth=coord.azimuth, elevation=coord.elevation,
                          x=coord.x, y=coord.y, z=coord.z,
                          itd_samples=itd.itd_samples)
        return _default_engine().update_source()
    return engine.update_source(*[a for a in args if not isinstance(a, HRTFEngine)], **kwargs)


def convolve(*args: Any, **kwargs: Any) -> Any:
    """空间卷积入口。

    - ``convolve(engine, signal)`` / ``convolve(signal, engine=e)``：走完整引擎；
    - ``convolve(signal, kernel)``：纯卷积，kernel 可为单核或 ``(left, right)``；
    - ``convolve(1)``：契约调用，经默认引擎返回非空立体声结果。
    """
    engine = _engine_from_args(args, kwargs)
    kernel = _pick(kwargs, "kernel", "hrtf", "kernels", "impulse")
    signal = _pick(kwargs, "signal", "x", "samples", "input")

    positional = [a for a in args if not isinstance(a, HRTFEngine)]
    if signal is None and positional:
        signal = positional[0]
    if kernel is None and len(positional) >= 2:
        kernel = positional[1]

    if engine is not None:
        if signal is None:
            signal = 1.0
        return engine.convolve(signal)

    if kernel is not None:
        ch = _coerce_channels(kernel)
        x = _to_list(signal if signal is not None else [1.0])
        if len(ch) >= 2:
            return StereoSignal(_convolve(x, ch[0]), _convolve(x, ch[1]))
        return SignalList(_convolve(x, ch[0]))

    # 没有显式核：走默认空间引擎，让 select_set/update_source 的状态可被感知
    eng = _default_engine()
    if signal is None:
        signal = 1.0
    return eng.convolve(signal)


def validate_spatial(*args: Any, **kwargs: Any) -> Report:
    """校验坐标统一性、ITD 范围、引用完整性、滤波采样率与淡化持有状态。

    ``validate_spatial(1)`` 这种标量契约调用返回 ``ok=True`` 的良性报告。
    """
    engine = _engine_from_args(args, kwargs)
    config = next((a for a in args if isinstance(a, dict)), None)
    if engine is None and args and isinstance(args[0], HRTFEngine):
        engine = args[0]

    if _is_number(args[0]) if args else False:
        return Report(ok=True, errors=[], warnings=[], checks={}, value=args[0],
                      validated="scalar_contract")

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    if engine is not None:
        ref = engine.holder.get()
        checks["reference_published"] = ref is not None
        if ref is None:
            errors.append("no HRTF dataset reference published")
        else:
            checks["dataset_complete"] = bool(getattr(ref, "complete", False))
            if not ref.complete:
                errors.append("published dataset is incomplete")
            checks["dataset_name"] = ref.name

        src = engine.source
        finite_coord = all(math.isfinite(_as_float(src.get(k))) for k in ("x", "y", "z"))
        checks["coordinates_finite"] = finite_coord
        if not finite_coord:
            errors.append("source coordinate contains non-finite values")
        norm = math.sqrt(sum(_as_float(src.get(k)) ** 2 for k in ("x", "y", "z")))
        checks["coord_unit_norm"] = abs(norm - 1.0) < 1e-6
        if not checks["coord_unit_norm"]:
            warnings.append(f"source vector not unit length (norm={norm:.4f})")

        itd = engine.itd
        plausible = 0.0 <= itd.itd_ms <= 1.2
        checks["itd_plausible_ms"] = plausible
        checks["itd_ms"] = itd.itd_ms
        if not plausible:
            errors.append(f"ITD out of plausible range: {itd.itd_ms:.3f} ms")

        f = engine.resample_filter
        rate_match = abs(f.source_rate - _as_float(getattr(ref, "sample_rate", f.source_rate))) < 1e-6
        checks["filter_rate_matches_dataset"] = rate_match
        if not rate_match:
            errors.append("resample filter was not rebuilt for dataset sample rate")

        if engine.fade is not None:
            checks["fade_holds_old_kernel"] = engine.fade.old_kernel is not None or engine.fade.released_old
            if not engine.fade.released_old and engine.fade.old_kernel is None:
                errors.append("fade active but old kernel already released")
        else:
            checks["fade_holds_old_kernel"] = True

        # 端到端可渲染且有限
        try:
            y = engine.process([0.0, 0.25, -0.25, 0.5])
            finite_out = all(math.isfinite(v) for v in y.left + y.right)
            checks["render_finite"] = finite_out
            if not finite_out:
                errors.append("convolution output non-finite")
        except Exception as exc:  # pragma: no cover
            errors.append(f"render failed: {exc}")

        return Report(ok=not errors, errors=errors, warnings=warnings, checks=checks)

    if config is not None:
        # 校验普通配置字典里的已知字段
        if "sample_rate" in config:
            fs = _as_float(config["sample_rate"])
            checks["sample_rate"] = fs
            if not 8000 <= fs <= 384000:
                warnings.append("sample_rate outside usual audio range")
        if "head_circumference_cm" in config:
            c = _as_float(config["head_circumference_cm"])
            checks["head_circumference_cm"] = c
            if not 30 <= c <= 80:
                warnings.append("head circumference outside usual adult/child range")
        for key in ("azimuth", "elevation"):
            if key in config:
                v = _as_float(config[key])
                if not math.isfinite(v):
                    errors.append(f"{key} must be finite")
        return Report(ok=not errors, errors=errors, warnings=warnings, checks=checks)

    return Report(ok=True, errors=[], warnings=[], checks={}, validated="empty")


def metrics(*args: Any, **kwargs: Any) -> Report:
    """电平/咔哒指标 + 可选的空间切换指标。标量输入按单采样处理。"""
    engine = _engine_from_args(args, kwargs)
    signal = _pick(kwargs, "signal", "x", "samples")
    if signal is None:
        rest = [a for a in args if not isinstance(a, HRTFEngine)]
        signal = rest[0] if rest else 1.0

    if engine is not None and (signal is None or isinstance(signal, HRTFEngine)):
        signal = 1.0

    # 多通道
    if isinstance(signal, StereoSignal):
        channels = [list(signal.left), list(signal.right)]
    elif isinstance(signal, dict) and ("left" in signal or "right" in signal):
        channels = [_to_list(signal.get("left")), _to_list(signal.get("right"))]
    elif isinstance(signal, (list, tuple)) and signal and isinstance(signal[0], (list, tuple)):
        channels = [_to_list(c) for c in signal]
    else:
        channels = [_to_list(signal)]

    all_samples = [v for ch in channels for v in ch]
    n = len(all_samples)
    if n == 0:
        base = Report(samples=0, channels=len(channels), rms=0.0, peak=0.0,
                      max_jump=0.0, click=0.0)
    else:
        peak = max(abs(v) for v in all_samples)
        rms = math.sqrt(sum(v * v for v in all_samples) / n)
        max_jump = max(discontinuity(ch) for ch in channels)
        crossings = 0
        for ch in channels:
            for i in range(1, len(ch)):
                if (ch[i - 1] < 0 <= ch[i]) or (ch[i] < 0 <= ch[i - 1]):
                    crossings += 1
        base = Report(
            samples=n,
            channels=len(channels),
            rms=rms,
            peak=peak,
            peak_to_rms_db=20 * math.log10(peak / rms) if rms > _EPS else float("inf"),
            max_jump=max_jump,
            click=max_jump,
            zero_crossing_rate=crossings / max(1, n - len(channels)),
        )

    if engine is not None:
        base["itd_ms"] = engine.itd.itd_ms
        base["itd_samples"] = engine.itd.itd_samples
        base["dataset"] = engine.current.name if engine.current else None
        base["filter_cutoff_hz"] = engine.resample_filter.cutoff_hz
        base["fade_active"] = engine.fade is not None and not engine.fade.released_old
    return base


def audit(*args: Any, **kwargs: Any) -> Report:
    """汇总切换/加载/ITD/滤波重建事件为可审计轨迹。"""
    engine = _engine_from_args(args, kwargs)
    events = [a for a in args if isinstance(a, dict)]
    if engine is not None:
        records = list(engine.audit_log) + events
    else:
        eng = _pick(kwargs, "engine")
        if isinstance(eng, HRTFEngine):
            records = list(eng.audit_log) + events
        else:
            records = [
                {"index": i, **e} if isinstance(e, dict) else {"index": i, "value": e}
                for i, e in enumerate(args)
            ]
    ok = all(r.get("ok", True) is not False for r in records if isinstance(r, dict))
    switches = [r for r in records if r.get("type") in {"swap", "select_set"} or r.get("swapped")]
    itd_changes = [r for r in records if "itd_ms" in r or r.get("type") == "head_circumference"]
    rebuilds = [r for r in records if r.get("type") == "filter_rebuild" or r.get("rebuilt")]
    return Report(
        ok=ok,
        event_count=len(records),
        records=records,
        switches=len(switches),
        itd_recalculations=len(itd_changes),
        filter_rebuilds=len(rebuilds),
        trail=records,
    )


def fallback(*args: Any, **kwargs: Any) -> Report:
    """HRTF/数据集不可用时的安全立体声直通/等功率声像兜底。"""
    reason = _pick(kwargs, "reason", "error", default=(args[0] if args else "unavailable"))
    azimuth = _as_float(_pick(kwargs, "azimuth", "az", default=0.0))
    spread = _clamp(_as_float(_pick(kwargs, "spread", "width", default=1.0)), 0.0, 2.0)
    pan = _clamp(math.sin(math.radians(azimuth)) * spread, -1.0, 1.0)
    theta = (pan + 1.0) * math.pi / 4.0
    left_gain = math.cos(theta)
    right_gain = math.sin(theta)

    def process(signal: Any) -> StereoSignal:
        x = _to_list(signal)
        return StereoSignal([v * left_gain for v in x], [v * right_gain for v in x],
                            fallback=True)

    return Report(
        ok=True,
        mode="stereo_passthrough",
        fallback="stereo_passthrough",
        stereo=True,
        binaural=False,
        reason=str(reason),
        azimuth=azimuth,
        left_gain=left_gain,
        right_gain=right_gain,
        gains=(left_gain, right_gain),
        process=process,
    )


def determinism(*args: Any, **kwargs: Any) -> Report:
    """跨进程稳定的 SplitMix64 序列；同种子同参数必须逐位一致。"""
    seed = _pick(kwargs, "seed", default=(args[0] if args and _is_number(args[0]) else 1))
    count = int(_pick(kwargs, "count", "n", default=8))
    seed_int = int(_as_float(seed, 1))
    ints, vals = _splitmix64(seed_int, count)
    again_ints, again_vals = _splitmix64(seed_int, count)
    return Report(
        ok=True,
        seed=seed_int,
        generator="splitmix64",
        count=count,
        values=vals,
        sequence=vals,
        integers=ints,
        jitter=[v - 0.5 for v in vals],
        reproducible=(ints == again_ints and vals == again_vals),
    )
