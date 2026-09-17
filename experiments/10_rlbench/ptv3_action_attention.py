"""Capture direct action-query to point-key attention without editing PTV3."""

from __future__ import annotations

import re

import numpy as np
import torch

from pointact.model.ptv3.concerto.utils import offset2bincount
from ptv3_feature_viz import FeatureCapture


class ActionAttentionCapture(FeatureCapture):
    """FeatureCapture plus exact reconstruction of action-to-point softmax rows.

    PointACT uses FlashAttention, which does not return its attention matrix.  A
    forward-pre-hook applies the module's own qkv projection and reconstructs
    only the action-query rows.  This is diagnostic-only and does not alter the
    values used by the actual forward pass.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._expected_action_outputs: dict[str, torch.Tensor] = {}

    def begin_request(self, batch: dict) -> None:
        super().begin_request(batch)
        self._expected_action_outputs.clear()

    @staticmethod
    def key_prefix(module_name: str) -> str:
        match = re.fullmatch(r"enc\.enc(\d+)\.block(\d+)\.attn", module_name)
        if match is None:
            raise ValueError(f"Unexpected attention module name: {module_name}")
        return f"action_attention_stage{match.group(1)}_block{match.group(2)}"

    @torch.no_grad()
    def make_attention_pre_hook(self, module_name: str, skip_state_token: bool):
        prefix = self.key_prefix(module_name)

        def hook(module, inputs) -> None:
            if not self.enabled_for_request or self.pending_arrays is None:
                return
            point = inputs[0]
            batch_counts = offset2bincount(point.offset)
            if len(batch_counts) != 1:
                raise ValueError("Attention visualization currently requires batch size 1")
            if module.enable_rpe:
                raise ValueError("RPE reconstruction is not implemented")

            pad, _unpad, cu_seqlens = module.get_padding_and_inverse(point)
            patch_lengths = torch.diff(cu_seqlens).tolist()
            order = point.serialized_order[module.order_index][pad]
            point_qkv = module.qkv(point.feat)[order]

            num_tokens = int(point.action_feat.shape[1])
            first_action = 1 if skip_state_token else 0
            action_indices = torch.arange(
                first_action, num_tokens, device=point.feat.device
            )
            action_qkv = module.qkv(point.action_feat)[0]
            num_patches = int(
                torch.div(
                    batch_counts[0] + module.patch_size - 1,
                    module.patch_size,
                    rounding_mode="trunc",
                )
            )

            num_points = len(point.feat)
            num_heads = module.num_heads
            num_actions = len(action_indices)
            per_action = torch.zeros(
                (num_actions, num_points), dtype=torch.float32,
                device=point.feat.device,
            )
            per_head = torch.zeros(
                (num_heads, num_points), dtype=torch.float32,
                device=point.feat.device,
            )
            expected_action = torch.zeros_like(point.action_feat[0], dtype=torch.float32)

            qkv_chunks = torch.split(point_qkv, patch_lengths, dim=0)
            order_chunks = torch.split(order, patch_lengths, dim=0)
            head_dim = module.channels // num_heads
            for point_chunk, index_chunk in zip(qkv_chunks, order_chunks):
                full_qkv = torch.cat([action_qkv, point_chunk], dim=0)
                full_qkv = full_qkv.reshape(
                    -1, 3, num_heads, head_dim
                ).to(torch.float16)
                query, key, value = full_qkv.unbind(dim=1)
                logits = torch.einsum(
                    "qhd,khd->hqk", query, key
                ).float() * module.scale
                probability = torch.softmax(logits, dim=-1)

                # H x selected-action-queries x point-keys.
                point_probability = probability[
                    :, action_indices, num_tokens:
                ]
                action_weight = point_probability.mean(dim=0) / num_patches
                head_weight = point_probability.mean(dim=1) / num_patches
                per_action.index_add_(1, index_chunk, action_weight)
                per_head.index_add_(1, index_chunk, head_weight)

                all_action_probability = probability[:, :num_tokens]
                action_value = torch.einsum(
                    "haL,Lhd->ahd", all_action_probability, value.float()
                ).reshape(num_tokens, module.channels)
                expected_action += action_value.float() / num_patches

            mean_weight = per_action.mean(dim=0)
            self.pending_arrays[f"{prefix}_point_coordinates"] = (
                point.coord.detach().float().cpu().numpy()
            )
            self.pending_arrays[f"{prefix}_point_weights"] = (
                mean_weight.cpu().numpy()
            )
            self.pending_arrays[f"{prefix}_per_action_weights"] = (
                per_action.cpu().numpy()
            )
            self.pending_arrays[f"{prefix}_per_head_weights"] = (
                per_head.cpu().numpy()
            )
            self.pending_arrays[f"{prefix}_point_attention_mass"] = np.asarray(
                float(mean_weight.sum().cpu()), dtype=np.float32
            )
            self.pending_arrays[f"{prefix}_num_action_queries"] = np.asarray(
                num_actions, dtype=np.int64
            )
            self.pending_arrays[f"{prefix}_num_heads"] = np.asarray(
                num_heads, dtype=np.int64
            )
            self.pending_arrays[f"{prefix}_serialized_order"] = (
                order.detach().cpu().numpy().astype(np.int64)
            )

            projection_dtype = module.proj.weight.dtype
            expected_projected = module.proj(
                expected_action.to(projection_dtype)
            ).float()
            self._expected_action_outputs[prefix] = expected_projected.cpu()

        return hook

    def make_attention_post_hook(self, module_name: str):
        prefix = self.key_prefix(module_name)

        @torch.no_grad()
        def hook(_module, _inputs, output) -> None:
            if self.pending_arrays is None or prefix not in self._expected_action_outputs:
                return
            actual = output.action_feat[0].detach().float().cpu()
            expected = self._expected_action_outputs.pop(prefix)
            difference = (actual - expected).abs()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            )
            self.pending_arrays[f"{prefix}_reconstruction_max_abs_error"] = np.asarray(
                float(difference.max()), dtype=np.float32
            )
            self.pending_arrays[f"{prefix}_reconstruction_mean_abs_error"] = np.asarray(
                float(difference.mean()), dtype=np.float32
            )
            self.pending_arrays[f"{prefix}_reconstruction_cosine"] = np.asarray(
                float(cosine), dtype=np.float32
            )

        return hook
