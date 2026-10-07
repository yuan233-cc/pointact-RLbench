"""Geometry-only frozen polar teacher + trainable Concerto and metric decoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .action_head_3d.ptv3_backbone import PointTransformerUnetWithAction
from .action_head_3d.polarapp_tasknet_encoder import PolarAppTaskAwareEncoder, load_polarapp_tasknet_checkpoint
from .action_head_3d.cga_dino_normal import CgaDinoNormalNet, load_cga_dino_normal_checkpoint
from .action_head_3d.polar_depth_self_supervision import (
    PolarDepthSelfSupervision, structured_depth_holdout,
    mask_points_at_depth_targets, rasterize_fused_point_features,
)
from pointact.train.ptv3_init import adapt_ptv3_input_stem


@torch.no_grad()
def frozen_cga_features(teacher, observation, prior, rgb, chunk_size=224):
    """Keep native teacher resolution without large convolution index tensors."""
    if teacher.training:
        raise ValueError("Chunked teacher must be in eval mode")
    normals, banks = [], []
    for start in range(0, len(observation), chunk_size):
        section = slice(start, start + chunk_size)
        result = teacher(observation[section], prior[section], rgb[section])
        normals.append(result["normal"].float())
        banks.append(tuple(F.avg_pool2d(f.float(), 4, 4) for f in result["feature_levels"]))
        del result
    return torch.cat(normals), tuple(torch.cat([bank[i] for bank in banks]) for i in range(5))


class WorkspaceGeometryModel(nn.Module):
    """No language/action tokens. Both teachers retain checkpoint preprocessing."""
    def __init__(self, backbone, checkpoint, concerto_checkpoint, dino_weights=None):
        super().__init__()
        self.backbone = backbone
        if backbone == "tasknet":
            self.teacher = PolarAppTaskAwareEncoder(input_mode="native_stokes")
            load_polarapp_tasknet_checkpoint(self.teacher, checkpoint, require_normal_head=True)
            self.mapping = (0, 0, 1, 2, 2)
            channels = tuple(self.teacher.task_feature_channels[i] for i in self.mapping)
        elif backbone == "cga_dinov3":
            config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
            if config["data"]["input_mode"] != "native_cga" or config["data"]["image_size"] != 256:
                raise ValueError("Expected the calibrated native_cga 256 checkpoint")
            self.teacher = CgaDinoNormalNet.from_dinov3(dino_weights, observation_channels=11,
                physical_prior_channels=11, transformer_blocks=config["model"].get("transformer_blocks", 8))
            load_cga_dino_normal_checkpoint(self.teacher, checkpoint, strict=True)
            self.mapping = (0, 1, 2, 3, 4)
            channels = self.teacher.feature_channels
        else:
            raise ValueError(backbone)
        self.teacher.eval().requires_grad_(False)
        self.ptv3_model = PointTransformerUnetWithAction(
            input_size=9, ctx_embed_size=256, enc_channels=(64, 128, 256, 512, 768),
            enc_depths=(3, 3, 3, 12, 3), enc_num_head=(4, 8, 16, 32, 48),
            enc_mode=True, patch_size=256, apply_point_ca=False, ptv3_backend="concerto",
            polar_enabled=True, sfp_feature_channels=channels, polar_fusion_mode="workspace",
            polar_bbox_feature_levels=self.mapping, polar_workspace_attend_action=False,
        )
        module = self.ptv3_model.ptv3_model
        state = torch.load(concerto_checkpoint, map_location="cpu", weights_only=False)
        state = dict(state.get("state_dict", state))
        target = module.state_dict()
        adapt_ptv3_input_stem(state, target, copy_input_channels=6)
        compatible = {k: v for k, v in state.items() if k in target and v.shape == target[k].shape}
        if len(compatible) < 100:
            raise ValueError("Concerto checkpoint has too few compatible encoder tensors")
        module.load_state_dict(compatible, strict=False)
        self.objective = PolarDepthSelfSupervision(feature_channels=channels,
            point_feature_channels=(64, 128, 256, 512, 768),
            normal_weight=1.0, sparse_depth_weight=0.2, anchor_depth_weight=0.05,
            smoothness_weight=0.0, use_polar_features=False, pixel_center_offset=0.0)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, batch):
        images = batch["polar_images"]
        size = images.shape[-1]
        b = len(images)
        with torch.no_grad():
            if self.backbone == "tasknet":
                small = self.teacher.resize_sfp_observation(images[:, 0].float(), (64, 64))
                bank = self.teacher.forward_task_features(small)
                normal = self.teacher.decode_normals(bank, normalize=True, output_frame="canonical")
                normal = F.normalize(F.interpolate(normal, (size, size), mode="bilinear", align_corners=False), dim=1)
                strides = tuple(4.0 * x for x in self.teacher.task_feature_strides)
                offsets = (1.5,) * 3
            else:
                normal, bank = frozen_cga_features(self.teacher, batch["cga_observation"],
                    batch["cga_prior"], batch["rgb"])
                strides = (4., 8., 16., 32., 64.)
                # CGA's MaxPool2d(2) centers are (stride-1)/2.
                offsets = tuple((s - 1) / 2 for s in strides)
        bank = tuple(f[:, None].float() for f in bank)
        references = tuple(bank[i] for i in self.mapping)
        context = {key: batch[key] for key in ("polar_K", "T_camera_from_model", "view_valid",
                    "pixel_valid", "polar_workspace_mask", "point_pixel_indices", "point_pixel_image_hw")}
        context.update(polar_feature_levels=references, polar_bbox_feature_bank=bank,
            polar_bbox_bank_strides=strides, polar_bbox_bank_offsets=offsets,
            polar_feature_strides=tuple(strides[i] for i in self.mapping),
            polar_feature_offsets=tuple(offsets[i] for i in self.mapping),
            polar_image_hw=batch["point_pixel_image_hw"])
        valid = batch["observed_depth_valid"] & batch["polar_workspace_mask"].unsqueeze(2) & batch["pixel_valid"].unsqueeze(2)
        generator = None if self.training else torch.Generator(device=valid.device).manual_seed(173)
        hidden = structured_depth_holdout(valid, 0.7, generator=generator)
        points, counts, keep = mask_points_at_depth_targets(batch["points"], batch["npoints_in_batch"], hidden,
            context["polar_K"], context["T_camera_from_model"], context["view_valid"],
            point_pixel_indices=context["point_pixel_indices"], point_pixel_image_hw=context["point_pixel_image_hw"])
        if (counts < 1).any():
            raise ValueError("Holdout removed all points of a sample; use smaller blocks")
        context["point_pixel_indices"] = context["point_pixel_indices"][keep]
        # Explicit empty sequences: attention contains only point tokens.
        output = self.ptv3_model(points, counts, points.new_empty((b, 0, 256)),
            counts.new_zeros(b), points.new_empty((b, 0, 64)), polar_context=context, return_stage_points=True)
        maps, masks = rasterize_fused_point_features(output[-1], references, context["polar_K"],
            context["T_camera_from_model"], context["polar_image_hw"], context["view_valid"],
            feature_strides=context["polar_feature_strides"], feature_offsets=context["polar_feature_offsets"],
            correspondence_mode="workspace")
        return self.objective(None, maps, masks, images, context["polar_K"],
            observed_depth=batch["observed_depth"], observed_depth_valid=valid,
            pixel_valid=context["pixel_valid"], view_valid=context["view_valid"],
            depth_supervision_mask=hidden, normal_targets=normal[:, None],
            workspace_mask=context["polar_workspace_mask"])
