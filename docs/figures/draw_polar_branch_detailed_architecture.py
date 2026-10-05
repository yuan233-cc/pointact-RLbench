#!/usr/bin/env python3
"""Render the detailed architecture of the current PointACT Polar branch."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 4400, 2850
OUT = Path(__file__).with_name("polar_branch_detailed_architecture.png")

C = {
    "bg": "#F4F7FB",
    "panel": "#FFFFFF",
    "ink": "#122033",
    "muted": "#556579",
    "line": "#7B899B",
    "polar": "#5E4CC4",
    "polar_soft": "#EEEAFE",
    "cga": "#156F83",
    "cga_soft": "#E4F5F7",
    "dino": "#667284",
    "dino_soft": "#E9EDF2",
    "point": "#C66B1C",
    "point_soft": "#FFF0E2",
    "action": "#1C815F",
    "action_soft": "#E5F6EF",
    "vlm": "#2B67B2",
    "vlm_soft": "#E7F0FC",
    "loss": "#BA414A",
    "loss_soft": "#FDE8EA",
    "note": "#A37A15",
    "note_soft": "#FFF7D8",
    "white": "#FFFFFF",
}

REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def ft(size: int, bold: bool = False, mono: bool = False):
    return ImageFont.truetype(MONO if mono else (BOLD if bold else REG), size=size)


img = Image.new("RGB", (W, H), C["bg"])
d = ImageDraw.Draw(img)


def text(x, y, value, size=26, color=None, bold=False, anchor="la", spacing=7, mono=False, align="left"):
    d.multiline_text(
        (x, y), value, font=ft(size, bold=bold, mono=mono), fill=color or C["ink"],
        anchor=anchor, spacing=spacing, align=align,
    )


def box(xy, title, body="", *, fill=None, outline=None, title_color=None,
        title_size=30, body_size=23, align="left", dashed=False, radius=22,
        width=4, pad=22):
    x1, y1, x2, y2 = xy
    fill = fill or C["panel"]
    outline = outline or C["line"]
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=None if dashed else outline, width=width)
    if dashed:
        dash, gap = 20, 12
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
    anchor = "ma" if align == "center" else "la"
    tx = (x1 + x2) // 2 if align == "center" else x1 + pad
    text(tx, y1 + pad, title, title_size, title_color or C["ink"], True, anchor=anchor, align=align)
    if body:
        title_h = d.multiline_textbbox((0, 0), title, font=ft(title_size, True), spacing=5)[3]
        text(tx, y1 + pad + title_h + 14, body, body_size, C["ink"], anchor=anchor, spacing=8, align=align)


def section(xy, label, color):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=30, fill=C["panel"], outline="#CDD7E4", width=4)
    width = int(d.textlength(label, font=ft(34, True))) + 74
    d.rounded_rectangle((x1, y1, x1 + width, y1 + 64), radius=28, fill=color, outline=color)
    text(x1 + width // 2, y1 + 32, label, 34, C["white"], True, anchor="mm")


def arrow(points, *, color=None, width=6, dashed=False, head=18, label=None, label_at=None):
    color = color or C["line"]
    if dashed:
        for p0, p1 in zip(points, points[1:]):
            x0, y0 = p0
            x1, y1 = p1
            length = max(abs(x1 - x0), abs(y1 - y0))
            n = max(1, length // 26)
            for i in range(n):
                if i % 2 == 0:
                    t0, t1 = i / n, min((i + 1) / n, 1)
                    d.line((x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                            x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1), fill=color, width=width)
    else:
        d.line(points, fill=color, width=width, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if abs(x1 - x0) >= abs(y1 - y0):
        s = 1 if x1 > x0 else -1
        tip = [(x1, y1), (x1 - s * head, y1 - head // 2), (x1 - s * head, y1 + head // 2)]
    else:
        s = 1 if y1 > y0 else -1
        tip = [(x1, y1), (x1 - head // 2, y1 - s * head), (x1 + head // 2, y1 - s * head)]
    d.polygon(tip, fill=color)
    if label and label_at:
        tw = d.textlength(label, font=ft(21, True))
        d.rounded_rectangle((label_at[0] - 10, label_at[1] - 4,
                             label_at[0] + tw + 10, label_at[1] + 31), radius=8, fill=C["bg"])
        text(label_at[0], label_at[1], label, 21, color, True)


def pill(xy, label, fill, outline):
    d.rounded_rectangle(xy, radius=17, fill=fill, outline=outline, width=3)
    text((xy[0] + xy[2]) // 2, (xy[1] + xy[3]) // 2, label, 21, C["ink"], True, anchor="mm")


def stage_cell(x, y, w, h, stage, fmap, ptv3, heads):
    d.rounded_rectangle((x, y, x + w, y + h), radius=16, fill="#FBFCFE", outline="#B9C5D4", width=3)
    text(x + 18, y + 16, stage, 24, C["polar"], True)
    text(x + 18, y + 54, fmap, 20, C["ink"], mono=True)
    text(x + 18, y + 86, ptv3, 20, C["point"], mono=True)
    text(x + 18, y + 118, heads, 19, C["muted"])


# Header
text(82, 52, "PointACT 当前 Polar 分支｜详细模型结构", 58, C["ink"], True)
text(84, 128,
     "当前工作树 main@cbd40f4（含本地 Polar 实现）｜CGA+DINOv3 normal backbone｜示例空间尺寸 H=W=256",
     27, C["muted"])
pill((3200, 56, 3440, 108), "可训练", C["action_soft"], C["action"])
pill((3460, 56, 3700, 108), "冻结", C["dino_soft"], C["dino"])
pill((3720, 56, 4060, 108), "仅预训练使用", C["loss_soft"], C["loss"])
pill((4080, 56, 4320, 108), "标定路由", C["polar_soft"], C["polar"])


# Section 1: 2D Polar backbone
section((60, 195, 4340, 1250), "① Polar 2D 特征金字塔：CGA + frozen DINOv3", C["cga"])

box((95, 305, 700, 620), "Polar observation｜7ch",
    "[Iun, DoLP, cos(2AoLP), sin(2AoLP),\n view_x, view_y, view_z]\n[B,V,7,256,256] → flatten B×V",
    fill=C["polar_soft"], outline=C["polar"], body_size=22)
box((95, 650, 700, 985), "Physical prior｜11ch",
    "Ns1 / Ns2 / Nd 三组候选法向 = 9ch\n+ Iun + specular confidence\n[B,V,11,256,256] → flatten B×V",
    fill=C["polar_soft"], outline=C["polar"], body_size=22)
box((95, 1015, 700, 1200), "Aligned RGB｜3ch",
    "[B,V,3,256,256]，范围 [0,1]\n供 DINOv3 分支使用",
    fill=C["vlm_soft"], outline=C["vlm"], body_size=21)

box((800, 320, 1430, 590), "双 stem + CGA content-guided fusion",
    "obs: Conv1×1  7→11｜prior: Identity 11→11\nS = obs + prior\nA₁ = SpatialAttn₇×₇(S) + ChannelAttn(GAP(S))\nG = sigmoid(GroupedConv₇×₇([S,A₁]))\nF = Conv1×1(S + G·obs + (1−G)·prior)",
    fill=C["cga_soft"], outline=C["cga"], body_size=20)

# CGA encoder stages
box((1530, 285, 2580, 675), "CGA U-Net encoder + bottleneck Transformer ×8",
    "E1  DoubleConv(BN)          64 × 256²   stride 1\nE2  MaxPool + DoubleConv(IN)   128 × 128²   stride 2\nE3  MaxPool + DoubleConv(IN)   256 ×  64²   stride 4\nE4  MaxPool + DoubleConv(IN)   512 ×  32²   stride 8\nE5  MaxPool + DoubleConv(IN)   512 ×  16²   stride 16\n\nE5 flatten → 256 tokens × 512\n每块：LN→MHSA(8 heads, head dim 64)→residual\n       LN→MLP(×4)→residual → reshape E5",
    fill=C["cga_soft"], outline=C["cga"], body_size=21)

box((800, 735, 1430, 1135), "DINOv3 ConvNeXt-Base",
    "RGB ImageNet normalize\nget_intermediate_layers [0,1,2,3]\nD1 128×64²｜D2 256×32²\nD3 512×16²｜D4 1024×8²\n\n所有参数 requires_grad=False\nalways eval + no_grad",
    fill=C["dino_soft"], outline=C["dino"], dashed=True, body_size=23)

box((2700, 320, 3360, 760), "DINO 多尺度注入",
    "每层 1×1 projection → 128ch\n\nF3 = Conv3×3+IN+ReLU([E3, P1])\n     → 256 × 64²\nF4 = Conv3×3+IN+ReLU([E4, P2])\n     → 512 × 32²\nF5 = Conv3×3+IN+ReLU([E5, P3, ↑P4])\n     → 512 × 16²",
    fill=C["vlm_soft"], outline=C["vlm"], body_size=23)

box((3490, 300, 4250, 710), "送入 PointACT 的 5 层 feature bank",
    "level 0  E1  [B,V, 64,256,256]  stride 1\nlevel 1  E2  [B,V,128,128,128]  stride 2\nlevel 2  F3  [B,V,256, 64, 64]  stride 4\nlevel 3  F4  [B,V,512, 32, 32]  stride 8\nlevel 4  F5  [B,V,512, 16, 16]  stride 16\n\n同时保留 K、T_camera_from_model、\nview_valid、pixel_valid、pixel_transform",
    fill=C["polar_soft"], outline=C["polar"], body_size=21)

box((2700, 835, 3360, 1160), "法向 decoder｜策略阶段旁路",
    "Up1…Up4 + skip → normal head 64→3\n→ L2-normalized dense normal\n\n只在 standalone normal pretraining 的 forward() 使用；\n策略调用 forward_features()，不经过 decoder。",
    fill=C["loss_soft"], outline=C["loss"], dashed=True, body_size=21)
box((3490, 840, 4250, 1160), "梯度边界",
    "DINOv3：永久冻结\nCGA + projections + fusion：默认可训练\n（cga_freeze=true 时整体 no_grad）\nnormal decoder：策略 loss 不可达\nfeature bank：保留梯度送入 token router",
    fill=C["note_soft"], outline=C["note"], body_size=22)

arrow([(700, 455), (800, 455)], color=C["cga"])
arrow([(700, 820), (750, 820), (750, 530), (800, 530)], color=C["cga"])
arrow([(1430, 455), (1530, 455)], color=C["cga"])
arrow([(700, 1100), (750, 1100), (750, 925), (800, 925)], color=C["dino"], dashed=True)
arrow([(1430, 925), (2630, 925), (2630, 660), (2700, 660)], color=C["dino"], dashed=True)
arrow([(2580, 535), (2700, 535)], color=C["cga"])
arrow([(3360, 535), (3490, 535)], color=C["polar"])
arrow([(3025, 760), (3025, 835)], color=C["loss"], dashed=True, label="仅 normal pretrain", label_at=(3045, 782))


# Section 2: calibrated route and Utonia
section((60, 1300, 4340, 2580), "② 标定路由 + Utonia 五阶段联合注意力", C["polar"])

box((95, 1410, 640, 1710), "3D point stream",
    "point features [xyz + rgb + polar]\ncoord / batch / offset\nvoxel size = 0.01\nEmbedding → stage-0 point token",
    fill=C["point_soft"], outline=C["point"], body_size=22)
box((95, 1760, 640, 2035), "VLM context stream",
    "Qwen2.5-VL context embedding\n→ ctx_proj\n每个 stage 的 CA block：\naction ↔ context，point ↔ context（可选）",
    fill=C["vlm_soft"], outline=C["vlm"], body_size=21)
box((95, 2085, 640, 2390), "action / state stream",
    "learned action-position tokens\n+ CategorySpecificMLP(state) token\n[B, 1+chunk, C]\n跨 pooling 保留并线性投影",
    fill=C["action_soft"], outline=C["action"], body_size=22)

# Stage mapping strip
d.rounded_rectangle((735, 1385, 2570, 1585), radius=22, fill="#F8FAFD", outline="#AAB8C9", width=3)
text(765, 1406, "feature bank ↔ Utonia stage 对齐（每个 stage 开头：Linear(Cpolar→Cstage) + LayerNorm）", 24, C["ink"], True)
stage_cell(765, 1455, 335, 110, "S0 ↔ E1", "64ch @ s1", "64ch", "2 heads")
stage_cell(1115, 1455, 335, 110, "S1 ↔ E2", "128ch @ s2", "128ch", "4 heads")
stage_cell(1465, 1455, 335, 110, "S2 ↔ F3", "256ch @ s4", "256ch", "8 heads")
stage_cell(1815, 1455, 335, 110, "S3 ↔ F4", "512ch @ s8", "512ch", "16 heads")
stage_cell(2165, 1455, 375, 110, "S4 ↔ F5", "512ch @ s16", "768ch", "32 heads")

box((735, 1635, 1410, 2295), "A｜PolarTokenRouter（每个 serialized group）",
    "1. unique(point indices)；检查 group 不跨 batch\n2. x_cam = T_camera_from_model · x_model\n3. uv = π(K · x_cam)，可再乘 pixel_transform\n4. 过滤：finite、z>ε、图像范围、view_valid、pixel_valid\n5. 映射到当前 feature grid：\n   row/col = round((uv − center_offset)/stride)\n   center_offset = (stride−1)/2\n6. 扩展半径 r=1 的 3×3 邻域并去重\n7. 多 view 均衡分配预算；view 内 deterministic\n   2D farthest sampling，最多 32 token/group\n\nall 模式：每个 group 共享该样本全部有效格点；\n不受 32-token cap 限制。",
    fill=C["polar_soft"], outline=C["polar"], body_size=20)

box((1495, 1635, 2060, 2055), "B｜Polar token embedding",
    "route.features\n+ Linear₂→C(normalized x,y)\n+ Embedding(view_id)\n+ learned modality embedding\n→ LayerNorm\n→ 独立 polar_qkv Linear(C→3C)\n\n注：Polar token 不使用 3D RoPE",
    fill=C["polar_soft"], outline=C["polar"], body_size=21)

box((2145, 1635, 2970, 2055), "C｜组内 joint self-attention",
    "sequence = [ action/state | point | polar ]\n\npoint Q,K：应用 3D rotary position encoding\naction QKV：复制到该样本的每个 serialized group\npolar QKV：使用独立投影\n\n→ FlashAttention varlen（无 Flash 时 reference attention）\n→ token 间可双向交互",
    fill="#F2EDFF", outline=C["polar"], body_size=22)

box((3055, 1635, 3635, 2055), "D｜只回写两类输出",
    "point outputs → inverse serialization\n→ shared proj + dropout → point.feat\n\naction outputs → 对同一样本所有 group 求均值\n→ shared proj + dropout → action_feat\n\npolar query outputs：丢弃\npolar_writeback = false",
    fill=C["action_soft"], outline=C["action"], body_size=21)

box((1495, 2115, 3635, 2425), "E｜每个 Utonia encoder stage 的完整 block（默认 depths = 2 / 2 / 2 / 6 / 2）",
    "GridPoolingWithAction（stage>0，stride 2，point/action 通道同时升级；Polar 标定 metadata 原样保留）\n→ PolarStagePreparation（该 stage 的 feature grid 只投影一次）\n→ [ CPE + joint self-attention + residual + point/action MLP ]\n→ CA block：action ↔ VLM context；若 ptv3_apply_point_ca=true，point ↔ VLM context\n→ 下一层。序列化顺序在 z / z-trans / hilbert / hilbert-trans 间轮换，patch size 默认 128。",
    fill="#F7FAFE", outline="#58708F", body_size=21)

box((3730, 1510, 4260, 1875), "Action head",
    "最终 action_out_embeds\n（state token 移除）\n+ final point features\n→ PointWithActionRegressionMLPActionHead\n→ action chunk",
    fill=C["action_soft"], outline=C["action"], body_size=22)
box((3730, 1930, 4260, 2295), "训练信号与可观测性",
    "action regression loss\n沿 joint attention 回传至：\nrouter 后的 Polar adapters / qkv / embeddings、\nUtonia、CGA（若未冻结）\n\nroute stats：valid projections / candidates /\nselected / truncated，供标定排查",
    fill=C["note_soft"], outline=C["note"], body_size=20)

# Main flows in section 2
arrow([(640, 1560), (690, 1560), (690, 1780), (735, 1780)], color=C["point"])
arrow([(640, 1895), (680, 1895), (680, 2475), (1900, 2475), (1900, 2425)], color=C["vlm"],
      label="VLM context", label_at=(1010, 2444))
arrow([(640, 2235), (690, 2235), (690, 1608), (2260, 1608), (2260, 1635)], color=C["action"],
      label="action / state QKV", label_at=(1515, 1585))
arrow([(1410, 1845), (1495, 1845)], color=C["polar"])
arrow([(2060, 1845), (2145, 1845)], color=C["polar"])
arrow([(2970, 1845), (3055, 1845)], color=C["polar"])
arrow([(3635, 1810), (3690, 1810), (3690, 1695), (3730, 1695)], color=C["action"])
arrow([(1100, 1585), (1100, 1635)], color=C["polar"], label="当前 level", label_at=(1125, 1592))
arrow([(3360, 2055), (3360, 2115)], color=C["action"])

# Footer / key take-away
d.rounded_rectangle((60, 2630, 4340, 2785), radius=24, fill="#15263C", outline="#15263C")
text(105, 2660, "关键语义", 29, C["white"], True)
text(300, 2658,
     "Polar 图像不是先压成一个全局向量，而是以“标定 + 局部几何路由”的方式，在 5 个 3D 尺度上作为只读 memory token 参与 action / point 联合注意力。",
     26, C["white"], True)
text(300, 2713,
     "因此动作 loss 可以更新 Polar 特征提取与融合参数，但不会把 attention 输出写回图像网格；冻结边界由 DINO 永久冻结与 cga_freeze 开关共同决定。",
     24, "#C9D7E8")

img.save(OUT, quality=95)
print(OUT)
