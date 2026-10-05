#!/usr/bin/env python3
"""Render the current CGA+DINOv3 pretraining and PointACT joint-training graph."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 3840, 2400
OUT = Path(__file__).with_name("cga_pretrain_joint_training.png")

COLORS = {
    "bg": "#F5F7FA",
    "ink": "#142033",
    "muted": "#536174",
    "panel": "#FFFFFF",
    "line": "#718096",
    "stage1": "#3F6FCF",
    "stage1_soft": "#EAF1FF",
    "stage2": "#D96A20",
    "stage2_soft": "#FFF1E6",
    "data": "#F2ECFF",
    "data_border": "#7956B3",
    "train": "#E5F7F1",
    "train_border": "#18866A",
    "frozen": "#E9EDF2",
    "frozen_border": "#687485",
    "loss": "#FFE8E5",
    "loss_border": "#C94A41",
    "note": "#FFF8D9",
    "note_border": "#B08A1D",
    "white": "#FFFFFF",
}

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REGULAR)
    return ImageFont.truetype(path, size=size)


img = Image.new("RGB", (W, H), COLORS["bg"])
d = ImageDraw.Draw(img)


def rounded_box(
    xy: tuple[int, int, int, int],
    title: str,
    body: str = "",
    *,
    fill: str = COLORS["panel"],
    outline: str = COLORS["line"],
    title_color: str = COLORS["ink"],
    width: int = 4,
    radius: int = 24,
    title_size: int = 34,
    body_size: int = 27,
    dashed: bool = False,
    align: str = "center",
) -> None:
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=None if dashed else outline, width=width)
    if dashed:
        dash = 20
        gap = 12
        for x in range(x1 + radius, x2 - radius, dash + gap):
            d.line((x, y1, min(x + dash, x2 - radius), y1), fill=outline, width=width)
            d.line((x, y2, min(x + dash, x2 - radius), y2), fill=outline, width=width)
        for y in range(y1 + radius, y2 - radius, dash + gap):
            d.line((x1, y, x1, min(y + dash, y2 - radius)), fill=outline, width=width)
            d.line((x2, y, x2, min(y + dash, y2 - radius)), fill=outline, width=width)
        d.arc((x1, y1, x1 + 2 * radius, y1 + 2 * radius), 180, 270, fill=outline, width=width)
        d.arc((x2 - 2 * radius, y1, x2, y1 + 2 * radius), 270, 360, fill=outline, width=width)
        d.arc((x1, y2 - 2 * radius, x1 + 2 * radius, y2), 90, 180, fill=outline, width=width)
        d.arc((x2 - 2 * radius, y2 - 2 * radius, x2, y2), 0, 90, fill=outline, width=width)
    tf = font(title_size, bold=True)
    bf = font(body_size)
    pad = 24
    title_bbox = d.multiline_textbbox((0, 0), title, font=tf, spacing=5, align=align)
    title_h = title_bbox[3] - title_bbox[1]
    tx = (x1 + x2) // 2 if align == "center" else x1 + pad
    anchor = "ma" if align == "center" else "la"
    d.multiline_text((tx, y1 + pad), title, font=tf, fill=title_color, spacing=5, align=align, anchor=anchor)
    if body:
        by = y1 + pad + title_h + 17
        bx = (x1 + x2) // 2 if align == "center" else x1 + pad
        d.multiline_text(
            (bx, by), body, font=bf, fill=COLORS["ink"], spacing=9, align=align, anchor=anchor
        )


def arrow(
    points: list[tuple[int, int]],
    *,
    color: str = COLORS["line"],
    width: int = 6,
    dashed: bool = False,
    head: int = 18,
    label: str | None = None,
    label_at: tuple[int, int] | None = None,
    label_color: str | None = None,
) -> None:
    if dashed:
        for a, b in zip(points, points[1:]):
            x1, y1 = a
            x2, y2 = b
            length = max(abs(x2 - x1), abs(y2 - y1))
            steps = max(1, length // 28)
            for i in range(steps):
                if i % 2 == 0:
                    t0 = i / steps
                    t1 = min((i + 1) / steps, 1.0)
                    d.line(
                        (x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0,
                         x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1),
                        fill=color, width=width,
                    )
    else:
        d.line(points, fill=color, width=width, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 > x0 else -1
        poly = [(x1, y1), (x1 - sign * head, y1 - head // 2), (x1 - sign * head, y1 + head // 2)]
    else:
        sign = 1 if y1 > y0 else -1
        poly = [(x1, y1), (x1 - head // 2, y1 - sign * head), (x1 + head // 2, y1 - sign * head)]
    d.polygon(poly, fill=color)
    if label and label_at:
        d.rounded_rectangle(
            (label_at[0] - 10, label_at[1] - 5, label_at[0] + 10 + d.textlength(label, font=font(23)), label_at[1] + 31),
            radius=8,
            fill=COLORS["bg"],
        )
        d.text(label_at, label, font=font(23), fill=label_color or color)


def pill(xy: tuple[int, int, int, int], text: str, fill: str, outline: str) -> None:
    d.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    d.text(((xy[0] + xy[2]) // 2, (xy[1] + xy[3]) // 2), text, font=font(24, bold=True), fill=COLORS["ink"], anchor="mm")


# Header
d.text((90, 55), "PointACT 最新 CGA 分支：法向预训练 → 机器人策略联合训练", font=font(58, bold=True), fill=COLORS["ink"])
d.text(
    (92, 132),
    "依据当前工作树 main@c0c8d39 + 未提交的 CGA+DINOv3 normal 实现（2026-10-03）",
    font=font(28),
    fill=COLORS["muted"],
)

# Legend
pill((2870, 65, 3095, 118), "可训练", COLORS["train"], COLORS["train_border"])
pill((3120, 65, 3345, 118), "冻结", COLORS["frozen"], COLORS["frozen_border"])
pill((3370, 65, 3745, 118), "loss / 监督", COLORS["loss"], COLORS["loss_border"])


# ------------------------------- Stage 1 -------------------------------
stage1 = (70, 205, 3770, 1125)
d.rounded_rectangle(stage1, radius=30, fill=COLORS["panel"], outline="#CDD7E5", width=4)
d.rounded_rectangle((70, 205, 570, 275), radius=28, fill=COLORS["stage1"], outline=COLORS["stage1"])
d.text((320, 240), "阶段 1｜表面法向预训练", font=font(36, bold=True), fill=COLORS["white"], anchor="mm")
d.text((610, 238), "Standalone：不加载 VLA / PointACT", font=font(29), fill=COLORS["muted"], anchor="lm")

rounded_box(
    (115, 330, 610, 755),
    "监督数据与分组切分",
    "• I0 / I45 / I90 / I135\n• 对齐 RGB\n• normal_gt + valid mask\n• K / 坐标系元数据\n\n来源：SfPUEL / HAMMER /\nMitsuba-RLBench 等\ntrain/val 的 object/scene/\nepisode group 必须互斥",
    fill=COLORS["data"], outline=COLORS["data_border"], align="left", body_size=25,
)

rounded_box(
    (700, 330, 1190, 790),
    "物理预处理",
    "Stokes → Iun / DoP / AoLP\nFresnel (n=1.5) →\nNs1, Ns2, Nd 三组候选法向\n3×3 specular confidence\n\n输出：\nobservation：native 11ch\n（robot compatibility 为 7ch）\nphysical prior：11ch\n= 9 normal + Iun + spec",
    fill=COLORS["data"], outline=COLORS["data_border"], align="left", body_size=22,
)

# Model compound box
d.rounded_rectangle((1290, 305, 2900, 850), radius=28, fill="#FAFCFF", outline=COLORS["stage1"], width=5)
d.text((1325, 330), "CgaDinoNormalNet", font=font(37, bold=True), fill=COLORS["stage1"])

rounded_box(
    (1330, 400, 1760, 650),
    "CGA 双分支编码器",
    "observation stem\nphysical-prior stem\n→ SA + CA + PA fusion\n→ U-Net E1…E5\n→ bottleneck Transformer ×8",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=21,
)

rounded_box(
    (1330, 690, 1760, 830),
    "RGB 分支",
    "DINOv3 ConvNeXt-Base\nD1…D4｜冻结 / no_grad",
    fill=COLORS["frozen"], outline=COLORS["frozen_border"], dashed=True, body_size=22,
)

rounded_box(
    (1850, 400, 2265, 650),
    "多尺度融合",
    "DINO 1×1 projections\nF3 = fuse(E3, D1)\nF4 = fuse(E4, D2)\nF5 = fuse(E5, D3, ↑D4)\n\n输出 5 层：\n[E1,E2,F3,F4,F5]",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=18,
)

rounded_box(
    (2355, 400, 2835, 650),
    "法向解码器",
    "Up1…Up4 + skip\n3-channel normal head\nL2 normalize\n→ predicted normal [N,3,H,W]",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=24,
)

rounded_box(
    (1850, 690, 2835, 830),
    "预训练阶段的参数更新",
    "训练：CGA、DINO projections、F3/F4/F5 fusion、decoder、normal head\n冻结：DINOv3 backbone（checkpoint 不保存其参数）",
    fill=COLORS["note"], outline=COLORS["note_border"], body_size=22,
)

rounded_box(
    (3000, 330, 3700, 620),
    "监督与选择",
    "masked cosine loss = mean(1 − cos)\n仅在 independent normal GT 有效像素上计算\n\n验证：normal MAE、<11.25°、<22.5°\n多数据集以 macro-average MAE 选 best",
    fill=COLORS["loss"], outline=COLORS["loss_border"], align="left", body_size=22,
)

rounded_box(
    (3000, 680, 3700, 880),
    "阶段产物",
    "best.pt / last.pt\ntrainable_state_dict + optimizer + scheduler\n用于阶段 2 初始化 CGA+DINO 特征编码器",
    fill=COLORS["stage1_soft"], outline=COLORS["stage1"], body_size=23,
)

arrow([(610, 540), (700, 540)], color=COLORS["data_border"])
arrow([(1190, 540), (1250, 540), (1250, 505), (1330, 505)], color=COLORS["stage1"])
arrow([(1190, 620), (1250, 620), (1250, 760), (1330, 760)], color=COLORS["stage1"])
arrow([(1760, 505), (1850, 505)], color=COLORS["train_border"])
arrow([(1760, 760), (1810, 760), (1810, 560), (1850, 560)], color=COLORS["frozen_border"], dashed=True)
arrow([(2265, 505), (2355, 505)], color=COLORS["train_border"])
arrow([(2835, 505), (3000, 455)], color=COLORS["loss_border"])
arrow([(3350, 620), (3350, 680)], color=COLORS["stage1"])

# Stage handoff
arrow([(3350, 880), (3350, 1160), (1930, 1160), (1930, 1325)], color=COLORS["stage1"], width=8, label="加载 best.pt", label_at=(2550, 1138))


# ------------------------------- Stage 2 -------------------------------
stage2 = (70, 1190, 3770, 2290)
d.rounded_rectangle(stage2, radius=30, fill=COLORS["panel"], outline="#CDD7E5", width=4)
d.rounded_rectangle((70, 1190, 620, 1260), radius=28, fill=COLORS["stage2"], outline=COLORS["stage2"])
d.text((345, 1225), "阶段 2｜PointACT 联合策略训练", font=font(36, bold=True), fill=COLORS["white"], anchor="mm")
d.text((660, 1224), "CGA 特征与 3D 点、状态、语言/图像上下文共同优化动作策略", font=font(28), fill=COLORS["muted"], anchor="lm")

rounded_box(
    (115, 1325, 620, 1900),
    "机器人训练 batch",
    "3D 点：xyz + rgb + polar（9ch）\n≤4096 points\nrobot state + GT action\n语言 / front RGB\n\nPolar sidecar：\n• observation 7ch\n• physical prior 11ch\n• aligned polar_rgb 3ch\n• K、T_camera_from_model\n• view / pixel valid mask",
    fill=COLORS["data"], outline=COLORS["data_border"], align="left", body_size=25,
)

rounded_box(
    (710, 1325, 1170, 1545),
    "VLM 上下文",
    "Qwen2.5-VL 3B\nvision tower + LLM + merger\n冻结 → ctx embeddings → ctx_proj",
    fill=COLORS["frozen"], outline=COLORS["frozen_border"], dashed=True, body_size=24,
)

rounded_box(
    (710, 1615, 1170, 1870),
    "动作 / 状态 token",
    "robot state encoder\nlearned action-position token\nchunk size = 1",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=25,
)

rounded_box(
    (1290, 1325, 2140, 1615),
    "预训练 CGA+DINO 特征路径",
    "load cga_dino_normal_checkpoint = best.pt\nobservation 7ch + physical prior 11ch + RGB\n→ CGA encoder + frozen DINOv3 + F3/F4/F5 fusion\n→ [E1,E2,F3,F4,F5]（stride 1/2/4/8/16）",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=23,
)

pill((1325, 1560, 1650, 1605), "DINO 永久冻结", COLORS["frozen"], COLORS["frozen_border"])
pill((1670, 1560, 2105, 1605), "CGA 默认可训练；cga_freeze 可冻结", COLORS["note"], COLORS["note_border"])

rounded_box(
    (1290, 1690, 2140, 1925),
    "标定 Polar Router",
    "用 K + T 将 3D point 投影到特征网格\n按 point group 选 local 邻域 polar tokens\n保留 view / pixel validity 与多尺度对应",
    fill=COLORS["stage2_soft"], outline=COLORS["stage2"], body_size=25,
)

rounded_box(
    (2260, 1370, 3020, 1930),
    "Utonia / PointTransformerV3",
    "Utonia 预训练权重初始化\n9ch 输入（前 6ch 从原权重复制）\npoint serialization + 3D RoPE\n\n每层联合处理：\n① serialized point tokens\n② state / action tokens\n③ calibrated local polar tokens\n④ VLM context cross-attention\n\n输出 point features + action embeddings",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=26,
)

rounded_box(
    (3135, 1405, 3695, 1655),
    "动作回归头",
    "ActionRegressionHead\nposition + rotation(6D) + gripper\n→ predicted action",
    fill=COLORS["train"], outline=COLORS["train_border"], body_size=25,
)

rounded_box(
    (3135, 1740, 3695, 1995),
    "联合训练目标",
    "L_action = masked L2(pred, GT)\npos / rot / open 分项记录\n\n梯度 → action head、Utonia、state/ctx projection\n以及未冻结的 CGA feature path",
    fill=COLORS["loss"], outline=COLORS["loss_border"], body_size=22,
)

rounded_box(
    (710, 2015, 2140, 2205),
    "关键旁路：normal decoder 在策略阶段不运行",
    "PointACT 只调用 forward_features()；Up1…Up4 与 normal_head 被跳过。\n当前配置禁止 CGA+DINO 使用 SfP polar/depth self-supervision，\n因此联合阶段没有 normal loss / depth loss。",
    fill=COLORS["note"], outline=COLORS["note_border"], body_size=25,
)

rounded_box(
    (2260, 2040, 3695, 2205),
    "最终 checkpoint",
    "保存 VLA/PointACT 策略参数与训练状态；CGA 是否随 action loss 微调\n由 cga_freeze 决定；DINOv3 始终冻结并由外部权重重新加载。",
    fill=COLORS["stage2_soft"], outline=COLORS["stage2"], body_size=23,
)

# Stage 2 connections
arrow([(620, 1450), (710, 1450)], color=COLORS["frozen_border"], dashed=True)
arrow([(620, 1760), (710, 1760)], color=COLORS["train_border"])
arrow([(620, 1595), (1240, 1595), (1240, 1470), (1290, 1470)], color=COLORS["stage2"])
arrow([(2140, 1470), (2195, 1470), (2195, 1805), (2140, 1805)], color=COLORS["stage2"])
arrow([(1170, 1435), (1205, 1435), (1205, 1300), (2210, 1300), (2210, 1530), (2260, 1530)], color=COLORS["frozen_border"], dashed=True, label="ctx", label_at=(1840, 1268))
arrow([(1170, 1740), (1220, 1740), (1220, 1655), (2210, 1655), (2210, 1680), (2260, 1680)], color=COLORS["train_border"], label="state/action tokens", label_at=(1530, 1620))
arrow([(2140, 1805), (2260, 1805)], color=COLORS["stage2"], label="polar tokens", label_at=(2115, 1768))
arrow([(620, 1840), (1195, 1840), (1195, 1970), (2205, 1970), (2205, 1875), (2260, 1875)], color=COLORS["data_border"], label="point tokens", label_at=(1570, 1935))
arrow([(620, 1880), (1245, 1880), (1245, 1810), (1290, 1810)], color=COLORS["data_border"], label="K / T / validity", label_at=(900, 1882))
arrow([(3020, 1530), (3135, 1530)], color=COLORS["train_border"])
arrow([(3415, 1655), (3415, 1740)], color=COLORS["loss_border"])
arrow([(3415, 1995), (3415, 2040)], color=COLORS["stage2"])
arrow([(1290, 1585), (650, 1585), (650, 2120), (710, 2120)], color=COLORS["note_border"], dashed=True, label="decoder bypass", label_at=(660, 2075))

# Footer
d.text(
    (90, 2345),
    "实线：前向/数据流    虚线：冻结路径或显式旁路    绿色：可训练    灰色：冻结    红色：监督/loss",
    font=font(27),
    fill=COLORS["muted"],
)
d.text(
    (3750, 2345),
    "source: docs/cga_dinov3_normal.md + current implementation",
    font=font(23),
    fill=COLORS["muted"],
    anchor="ra",
)

img.save(OUT, format="PNG", optimize=True)
print(OUT)
