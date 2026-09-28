# PPFT 修复 RLBench 不完整点云：测试计划

状态：2026-09-28 完成代码、现有数据与官方权重元数据核查；尚未下载权重、运行 PPFT 或修改推理链路。

## 1. 结论与测试范围

可以测试。PPFT 是深度增强模型，实际接口是 RGB + 受损深度 + 偏振图，输出 dense depth；随后按 RLBench 标定回投为点云。它能测试当前视角可见表面的缺失与错误深度修复，不能据此宣称恢复被遮挡的物体背面或完整 3D 形状。

本实验运行完整 PPFT 深度模型（encoder、decoder、传播模块），与 `plan_ptv3_aligned_polar_ppft.md` 中只提取 encoder 特征的方案不同。

```text
incomplete9 的受损 XYZ + 当前相机标定 → z-buffer → D_in（米，缺失=0）
对应 RGB + dense polarization + 相机视线 → PPFT 输入适配
                                  ↓
                           PPFT → D_pred
                                  ↓
                      回投、裁剪、统一 voxel/采样
                                  ↓
                       修复点云 → 几何评测 / PointACT

原始 coppelia_depth_m + object mask → 仅用于 GT、评测区域与可视化
```

## 2. 权重文件大小

以下为本次从官方下载脚本所指 Google Drive 文件页读取的文件元数据，不是根据参数量估算；没有下载权重正文。

| 文件 | 字节数 | 十进制大小 | 用途 |
|---|---:|---:|---|
| [ppft_fin.pt](https://drive.google.com/file/d/1bhVpk8ZQK_SPqtusKxAamSbgWM1ovPk3/view) | 1,083,919,501 | 1,083.92 MB / 1.084 GB | PPFT 最终 checkpoint，脚本改名为 `ckpts/ppft_final/model.pt` |
| [NYUv2.pt](https://drive.google.com/file/d/1KJUZ4I-v9Nba0DDswHe2-Avq7yll---t/view) | 334,415,907 | 334.42 MB | CompletionFormer foundation |
| [pvt.pth](https://drive.google.com/file/d/1raHsLhsI8LUVLShgVyaRS_zciLpJpo2q/view) | 97,987,902 | 97.99 MB | 原实现初始化依赖 |
| [resnet34.pth](https://drive.google.com/file/d/1I1204ezYAmkmDXcMAQpk4EMsxPV3_tSg/view) | 87,306,240 | 87.31 MB | 原实现初始化依赖 |

PPFT 单文件约 **1.009 GiB**；按原代码准备四个文件共 **1,603,629,550 字节，约 1.604 GB / 1.493 GiB**。本地 `ckpts/ppft_final` 目前只有 `model.txt`。文件大小不等于运行显存，也不能直接换算成有效模型参数量；正式下载后再检查 checkpoint 的 state_dict、优化器状态和共享参数。

若另写完整 state_dict 推理加载器，可能可以省掉被最终权重覆盖的初始化权重；必须先验证键覆盖，不能预先假设只用单文件就能跑原代码。

## 3. 选用现有数据及其边界

使用修正后的 `hybridvla_10tasks_train_keysteps_polar_rlbench9_v2`，目前有 10 任务、1,000 episodes、5,051 keyframes，包含 clean9 / incomplete9 / filled9 与 dense polar。

原始帧位于：

```text
rlbench_custom_render/RLBench/output/ten_tasks_polar_train_20260921/
  <task>/episode_XXXXXX/frames/XXXXXX.npz
  <task>/episode_XXXXXX/snapshots/frames/XXXXXX.json
```

NPZ 含 `rgb`、`coppelia_depth_m`、`coppelia_object_mask`；JSON 含 `cameras.front.intrinsics/to_world`。已抽查一帧为 256×256，深度单位为米。不能把 LMDB 中最多 512 点的 interaction target 当成 dense depth GT。

当前数据先执行 workspace crop 与 1.2 cm voxel，再施加丢点和几何损坏。因此从 incomplete9 z-buffer 得到的零深度同时包含 voxel 稀疏性和真正丢点，不能全部当成 corruption holes。

- 主协议：直接使用现有 incomplete9，回答“PPFT 能否修复这份已生成的数据”。
- 可选独立协议：对 dense depth 施加受控 dropout / depth bias 后再测试，回答传感器深度损坏问题；需另报结果，不能替代主协议。

当前损坏还包括 12–30 mm 视线偏移、支撑物变形和漂浮点等；必须同时测缺失区和保留但错误的观测。

## 4. 输入适配与单帧检查

### 受损深度与相机

在中心化、训练增强之前，将受损 XYZ 变到当前相机坐标并投影，每像素保留最近有效深度。用当前投影位置，不用损坏前 `point_source_pixel_indices` 代替。只读受损坐标，不用 GT 深度把保留点“校正”后再送给 PPFT。

沿用项目的 PyRep 投影与回投逻辑，做 pixel→point→pixel round-trip。实查 K 的 fx/fy 为负值；须把像素、坐标轴、焦距符号一同核验，再转换到 PPFT 的视线约定，不能只把焦距取绝对值。

### RGB 与偏振

官方 `leichenyang-7` 为：

```text
[I_un, DoLP, cos(2 AoLP), sin(2 AoLP), Vx, Vy, Vz]
```

现有 dense sidecar 有 `DoLP/cos2AoLP/sin2AoLP/valid_mask/AoLP_valid_mask`，但没有 I。现有点特征 Nx9 不能直接替代这组 7 通道图像。

- 正式输入：从 v2 相同的修正后渲染器同步导出 intensity/S0，核对官方 `I_un=(I0+I45+I90+I135)/2` 的强度尺度；不要直接混用旧 `frames_spp512` 中的 S0，那一批包含已在 v2 去掉的 workspace helper。
- 快速试跑：允许灰度近似 I，但单独标记为近似输入；不能凭此判断官方 PPFT 的最佳能力。
- 视线 V：按 RLBench 标定构造并转换到 PPFT convention。官方固定 `vd.npy` 属于 HAMMER，不能直接复用；其 XY 符号与常见前向 camera ray 相反。
- 分别处理光学有效性与 AoLP 有效性。低 DoLP 下角度无效不应抹去其他仍有效信息；无效值采用固定、记录在配置中的处理方式，不能借用 GT normals。
- 官方 loader 实际用 `cv2.imread` 的 BGR 顺序后做 ImageNet normalization；checkpoint 复现先遵循实际代码，再将换色序作为独立检查。

### 尺寸与运行依赖

静态检查发现官方末层 prompt 下采样与主分支的尺寸公式不同：256×256 输入预计得到 7×7 与 8×8，不能直接保证可运行。先尝试 pad 到 272×272（四边各 8 像素，depth pad=0，记录 RGB/polar padding 规则、更新 K，输出后 unpad），以真实前向确认各尺度形状；该方案尚未运行验证。

原仓库需要旧版 Torch、DCN 自定义算子等；`test.sh` 还引用当前缺失的 `main_refactored.py`。建议使用独立环境和最小 RLBench 推理入口，不直接套用 HAMMER loader。加载权重必须检查 missing/unexpected keys，单帧记录延迟和峰值显存。

## 5. 分阶段实验

### A. 零样本小规模验证

固定 `phone_on_base`、`stack_wine`、`water_plants` 三个任务，各 10 个 episode、每 episode 3 帧，共约 90 帧。冻结 PPFT，固定 frame manifest 与 corruption seed；调试帧与最终测试帧分开。

先通过：图像/点云叠加对齐、度量单位、投影 round-trip、polar 范围、尺寸与权重覆盖检查。输出同帧的 RGB、DoLP/AoLP、受损深度、预测深度、GT、误差图与同视角点云。

### B. 几何修复评测

| 方案 | 用途 |
|---|---|
| incomplete9 原始输入 | 观测 coverage 与已有点误差参照 |
| 当前 morphology filled9 | 当前实际修复方法 |
| PPFT fill-only | 最终显式 `where(D_in>0,D_in,D_pred)`，只补缺失值 |
| PPFT full enhancement | 允许修正已有错误深度，使用预测深度 |
| CompletionFormer RGB-D | 可选：评估不用偏振的预训练模型 |
| PPFT 偏振置零/跨帧打乱 | 输入依赖诊断；不等同于公平训练出的无偏振模型 |

官方 `preserve_input` 不保证最终预测逐像素等于输入；fill-only 用最终显式合并保证。full enhancement 不用 oracle mask 决定哪些点允许修正。

当前 morphology 的候选洞来自“clean voxel 来源像素 − retained 来源像素”，使用了仿真已知的洞位置。做两套清楚区分的结果：

1. archive-compatible 对照：在同一候选位置比较 morphology 与 PPFT，明确这是已知仿真 hole support 的受控实验。该信息仅用于输出选择/评测，不进入 PPFT。
2. 无 oracle 的输出：PPFT 对全图预测，仅按预测有效性和预先固定的 workspace/camera 规则裁剪。评价 mask 可来自 GT，但输出不能用 GT hole/object/depth 标签筛选。

评分分为全可见有效区域、交互物体区域、实际删除点区域、保留但受损区域以及未受损区域。使用 corruption 的来源记录建立评测 mask，单独统计投影移动和遮挡冲突。

- Depth：MAE/RMSE（mm），误差 >1/2/5 cm 比例。
- Completion：预测有效 coverage 与“误差 <1 cm / <2 cm 的正确恢复率”；不能只看非零像素比例。
- 3D：对可见表面统一 voxel/采样后报告 Chamfer（写明 squared 或 unsquared）及 F-score@1/2 cm。
- 稳定性：原本正确区域是否被破坏、边缘是否过平滑、漂浮点是否减少。
- 资源：batch=1 单帧模型延迟、全前处理/回投延迟、峰值显存。

原始 incomplete 在缺失区没有预测；该组应报 coverage/已有观测误差，不把缺失 0 当作有效深度参与 MAE。所有完成方法在同一 GT 区域评分，同时报告无效输出比例，避免只在自己成功输出的位置算误差。

先看每任务结果，再按 episode 聚合并给出置信区间；不能仅用桌面占多数的全场 RMSE 判断。通过小样本适配检查后扩到 10 任务固定 episode 集。

### C. 必要时微调

官方权重来自真实 HAMMER，RLBench 光学、材质、视角和稀疏度均有域差异，零样本提升尚不能保证。

若输入检查通过但零样本效果不足，按 episode 划分 train/val/test（建议每任务 70/15/15），不能随机切相邻 keyframe；最终加新 seed、新 episode 测试。先只训练偏振分支/融合模块，再按验证集决定是否解冻更多层。实际检查参数的 `requires_grad` 和 optimizer 参数集，不能依赖模块同名属性来冻结。

模型监督只从训练 split 使用 clean depth；测试 GT 只参与评分。报告零样本与微调结果，不将微调后结果称作直接迁移能力。若主张偏振带来收益，补充同预算、同训练数据的 RGB-D baseline。

### D. RLBench 闭环验证

几何结果通过后，在 `run_filled9_rlbench.py` 调用 `build_filled9` 的位置添加 PPFT 前端选项。使用完整修复深度回投，再附 RGB/DoLP/cos/sin 得到 Nx9，统一 workspace crop、voxel、点数与像素索引。

先固定 PointACT checkpoint、相同 reset/corruption seeds，比较 incomplete / morphology / PPFT，10 任务×25 episodes；保存 success rate、每任务结果与失败视频。25 episodes 作为首轮筛选，有趋势后用更多 seeds/episodes 确认。

PointACT 若在 morphology-filled 数据上训练，这一步仅衡量前端替换效果。要判断 PPFT 数据的最佳下游价值，需另做匹配 PPFT 数据、同训练预算的 policy。当前论文 reconstruction decoder 是训练期辅助分支，推理关闭，不能当作现成在线修复器。

## 6. 预期交付与判断依据

后续实施建议新增最小 `ppft_rlbench_adapter.py`、离线 `evaluate_ppft_repair.py`、固定 split/manifest，以及独立结果目录；不覆盖现有 filled9。每个结果保存输入来源、checkpoint 校验值、padding/normalization、种子与评分区域定义。

首轮优先交付 90 帧几何对照。只有交互物体/缺失区误差和正确恢复率改善，且未受损区域没有明显退化，再扩展到闭环。运行时间与显存预算以单帧实测为准。

## 7. 依据

- [论文](https://arxiv.org/abs/2404.04318)、[官方仓库](https://github.com/lastbasket/Polarization-Prompt-Fusion-Tuning)。
- PPFT：`model/ppft/ppft.py:24`、`datasets/hammer.py:113`、`scripts/data_processing/process_hammer.py:65`、`model/ppft/ppft_pvt.py:211`、`scripts/downloads/foundation_ckpt.sh`。
- 数据与标定：`experiments/10_rlbench/repair_10task_polar_rlbench9.py:93`、`:151`、`:254`。
- corruption：`experiments/10_rlbench/create_rlbench_10task_realistic_failure_dataset.py:316`。
- 当前填充：`experiments/10_rlbench/polar_depth_fill.py:17`。
- 可见真值：`experiments/10_rlbench/export_visible_target_gt.py:150`。
- 闭环接入：`experiments/10_rlbench/run_filled9_rlbench.py:167`、`run_filled9_server.py:30`。
- 新 `polar_rotation_aux_filled9` 已包含逐行 `pixel_indices`；旧 PPFT encoder 计划里“filled UV 缺失”的描述已部分过时。见 `experiments/10_rlbench/build_polar_rotation_aux.py:350`。
