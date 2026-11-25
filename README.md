# pairwise-hrtf-kernel-switch

## 领域

游戏、图形与多媒体

## 主题

HRTF 数据集切换与空间声像

## 说明

核交叉淡化、统一坐标与延迟重算。

初始状态下 `media/hrtf_kernel_switch.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复与实现对照

实现在 `media/hrtf_kernel_switch.py`，六个问题逐一对应：

| 题目问题 | 入口 / 组件 | 做法 |
| --- | --- | --- |
| 核直接替换、淡化只做增益不做相位，切换有咔哒 | `crossfade_kernel`、`FadeSession` | 互相关估计新旧核的起点/线性相位偏移与极性并对齐，再做等功率/线性交叉淡化；淡化会话输出连续块 |
| 方位按双耳坐标、仰角按另一套坐标，斜上方跳位 | `unified_coord`、`slerp_vectors` | 所有方向先统一到 x 右 / y 上 / z 前的笛卡尔坐标，支持 standard az-el、binaural/interaural-polar、vertical-polar，插值用 slerp |
| ITD 按固定头围，换集后与核不匹配、声像偏移 | `recompute_itd`、`HRTFEngine.set_head_circumference` | `半径=头围/(2π)`，Woodworth 公式 `ITD=a/c·(θ+sinθ)`，按统一坐标后的横向角和渲染采样率重算双耳延迟采样数 |
| 卷积在音频线程、加载在主线程，读到半成品核 | `load_before_swap`、`swap`、`ReferenceHolder` | 加载线程完整构造并 `freeze()` 数据集，最后单次引用发布；拒绝发布 `complete=False` 的对象；音频线程每块只快照一次引用 |
| 采样率不同时重采样器未重建，高频被截断 | `rebuild_filter`、`ResampleFilter` | 采样率变化才重建 Hamming-sinc 抗混叠滤波器与重采样比率，更新截止频率/系数/版本；相同采样率不重建 |
| 旧核要保留到淡出结束 | `hold_old_kernel`、`FadeSession` | 淡化期间强引用旧核与旧卷积流，淡出采样全部播完后才释放，可查询 `retained/released/remaining` |

另有 `select_set` / `update_source` / `convolve` 把上述环节串成
`HRTFEngine`，以及 `validate_spatial` / `metrics` / `audit` /
`fallback` / `determinism` 五个治理入口。所有模块级入口对契约调用
`fn(1)` 均返回非 `None`，并兼容常见的位置/关键字/配置字典写法与
numpy 数组输入（numpy 仅用于加速卷积，缺失时自动回退纯 Python）。

`tests/test_fixes.py` 在契约测试之外验证上述行为（咔哒下降、坐标
一致性、ITD 比例与声像、线程原子发布、滤波重建、旧核保留等）。
