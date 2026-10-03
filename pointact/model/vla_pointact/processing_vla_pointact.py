import os
from typing import Union

import numpy as np
import torch
from easydict import EasyDict
from lerobot.constants import OBS_STATE
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from pointact.constants import DEFAULT_STATE_TOKEN, STATE_END_TOKEN, STATE_START_TOKEN
from pointact.data.polar_material import polar_vlm_image
from pointact.model.backbone.processor_base import RobotPointProcessorBase
from pointact.utils.rotation import convert_rotation
from pointact.utils.torch_utils import pad_vector

RobotInput = Union[np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]]

os.environ["TOKENIZERS_PARALLELISM"] = "0"


class VLAEncDec3DProcessor(RobotPointProcessorBase):
    """Processor for Image, Text, Video, PointCloud and Robotic Action Processing"""

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        states: RobotInput = None,
        actions: RobotInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            return_mm_token_type_ids=False,
            **kwargs,
        )

        text = self._remove_state_tokens(text)
        text_inputs, image_inputs, videos_inputs = self._prepare_image_video_action_inputs(
            images, videos, text, output_kwargs
        )
        robot_inputs = self._prepare_robot_tensor_inputs(states=states, actions=actions)

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **robot_inputs},
        )

    @staticmethod
    def _remove_state_tokens(text):
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        for i in range(len(text)):
            for state_token in [STATE_START_TOKEN, STATE_END_TOKEN, DEFAULT_STATE_TOKEN]:
                text[i] = text[i].replace(state_token, "")
        return text

    @staticmethod
    def _as_batched_tensor(value):
        if value is None:
            return None
        if isinstance(value, list):
            value = torch.stack(value, dim=0)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    def _prepare_robot_tensor_inputs(self, states=None, actions=None):
        robot_inputs = {}
        states = self._as_batched_tensor(states)
        actions = self._as_batched_tensor(actions)
        if states is not None:
            robot_inputs["states"] = states
        if actions is not None:
            robot_inputs["actions"] = actions
        return robot_inputs

    @torch.no_grad
    def _prepare_robot_inputs(self, batch: dict, points_workspace: dict=None, remove_arm: bool=False,
                              point_indices_out: list | None = None):
        """Prepare model inputs from raw robot batch"""
        batch_messages = []
        batch_states = []

        state_keys = [x for x in batch.keys() if x.startswith(OBS_STATE)]
        batch_size = len(batch[state_keys[0]])
        repo_ids = self._resolve_repo_ids(batch, batch_size)

        batch_points, batch_point_centers = [], []
        for i, repo_id in enumerate(repo_ids):
            mini_batch = {k: v[i] for k, v in batch.items()}

            select_video_keys = self.robot_config["select_video_keys_for_vlm"][repo_id]
            select_state_keys = self.robot_config["select_state_keys"][repo_id]

            image_modes = self.robot_config.get("vlm_image_mode", {})
            if image_modes.get(repo_id, "rgb") == "polar":
                if len(select_video_keys) != 1:
                    raise ValueError("Polar VLM image mode requires exactly one selected VLM image key")
                if "polar_dense" not in mini_batch:
                    raise ValueError("Polar VLM image mode requires polar_dense at inference")
                polar = mini_batch["polar_dense"]
                if isinstance(polar, torch.Tensor) and polar.ndim == 3 and polar.shape[-1] == 4:
                    polar = polar.permute(2, 0, 1)
                elif isinstance(polar, np.ndarray) and polar.ndim == 3 and polar.shape[-1] == 4:
                    polar = np.moveaxis(polar, -1, 0)
                image_values = {select_video_keys[0]: polar_vlm_image(polar)}
            else:
                image_values = {key: mini_batch[key] for key in select_video_keys}

            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image_values[k]} for k in select_video_keys),
                    ],
                }
            ]
            messages[0]["content"].append(
                {"type": "text", "text": f"{mini_batch['task']}"},
            )

            state = None
            if len(select_state_keys) > 0:
                state_parts = []
                for key in select_state_keys:
                    value = mini_batch[key]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    state_parts.append(np.asarray(value))
                state = torch.as_tensor(np.concatenate(state_parts, axis=-1), dtype=torch.float32)

            workspace = self._resolve_points_workspace(repo_id, points_workspace)
            if point_indices_out is None:
                point_cloud = self._prepare_point_cloud_for_sample(
                    mini_batch, repo_id, workspace, remove_arm=remove_arm,
                )
            else:
                if "observation.points" not in mini_batch or "point_pixel_indices" not in mini_batch:
                    raise ValueError("Material conditioning needs precomputed observation.points and point_pixel_indices")
                point_pixels = np.asarray(mini_batch["point_pixel_indices"], dtype=np.int32).reshape(-1)
                cloud = self._as_numpy_point_cloud(mini_batch["observation.points"])
                if len(cloud) != len(point_pixels):
                    raise ValueError("Point cloud and point pixel indices have different lengths")
                mini_batch = dict(mini_batch)
                mini_batch["observation.points"] = np.column_stack((cloud, point_pixels))
                conditioned_cloud = self._build_existing_point_cloud(mini_batch, workspace)
                if remove_arm and "observation.robot_joints_bbox" in mini_batch:
                    conditioned_cloud = self._remove_robot_arm_points(conditioned_cloud, mini_batch)
                conditioned_cloud = self._subsample_point_cloud(
                    conditioned_cloud, self.robot_config["max_npoints"][repo_id])
                point_indices_out.append(torch.as_tensor(conditioned_cloud[:, -1].copy(), dtype=torch.long))
                point_cloud = np.ascontiguousarray(conditioned_cloud[:, :-1])
            point_cloud = torch.from_numpy(point_cloud).float()
            point_cloud, state, point_center = self._center_point_cloud_and_state(
                point_cloud,
                state,
                center_state=self._repo_config_flag("is_action_eef", repo_id, default=True),
            )
            if state is not None:
                state = self._normalize_robot_state(state.numpy(), repo_id)
                state = torch.as_tensor(state, dtype=torch.float32)
                batch_states.append(pad_vector(state, self.robot_config["max_state_dim"]))
            batch_messages.append(messages)
            batch_point_centers.append(point_center.numpy())
            batch_points.append(point_cloud)
            
        return batch_messages, batch_states or None, batch_points, batch_point_centers, repo_ids

    def _action_dim(self, repo_id: str) -> int:
        select_action_keys = self.robot_config["select_action_keys"][repo_id]
        return sum(self.robot_config["features"][repo_id][key]["shape"][0] for key in select_action_keys)

    def _process_robot_outputs(self, repo_ids: list[str], actions: torch.Tensor):
        """Slice padded model actions back to each robot's configured action dimension."""
        output_actions = []
        for i, repo_id in enumerate(repo_ids):
            output_actions.append(actions[i].detach().cpu().float()[..., : self._action_dim(repo_id)])
        return torch.stack(output_actions, dim=0)

    def _build_action_output(self, repo_ids: list[str], actions: torch.Tensor, pred_rot_type: str):
        output_actions = self._process_robot_outputs(repo_ids, actions).numpy()
        for i, repo_id in enumerate(repo_ids):
            output_actions[i] = self._unnormalize_robot_action(output_actions[i], repo_id)

        if pred_rot_type == "euler":
            quat = convert_rotation(
                output_actions[..., 3:6], "euler", "quat", euler_order_src="xyz", quat_order_dst="xyzw"
            )
            output_actions = np.concatenate([output_actions[..., :3], quat, output_actions[..., 6:]], -1)
        elif pred_rot_type == "rot6d":
            quat = convert_rotation(
                output_actions[..., 3:9], "rot6d", "quat", quat_order_dst="xyzw"
            )
            output_actions = np.concatenate([output_actions[..., :3], quat, output_actions[..., 9:]], -1)

        return EasyDict({"action": output_actions})

    @torch.no_grad
    def select_action(
        self, model, batch: dict, pred_rot_type: str, use_cot=False, 
        points_workspace: dict=None, remove_arm: bool=False, **kwargs
    ):
        conditioned = bool(model.config.use_polar_material_conditioning)
        point_indices = [] if conditioned else None
        batch_messages, batch_states, batch_points, batch_point_centers, repo_ids = self._prepare_robot_inputs(
            batch, points_workspace=points_workspace, remove_arm=remove_arm,
            point_indices_out=point_indices,
        )
        device = model.device

        inputs = self.apply_chat_template(
            batch_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"states": batch_states},
        ).to(device)
        # print(inputs['input_ids'])

        inputs["input_id_lens"] = inputs["attention_mask"].sum(dim=1).long().to(device)
        inputs["points"] = torch.cat(batch_points, 0).to(device)
        inputs["npoints_in_batch"] = torch.LongTensor([len(x) for x in batch_points]).to(device)
        inputs["attention_mask"] = inputs["attention_mask"].bool().to(device)
        if getattr(model.config, "polar_enabled", False):
            required = ("polar_images", "polar_K", "view_valid")
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"Polar inference inputs are missing {missing}")
            for key in required:
                inputs[key] = torch.as_tensor(batch[key]).to(device)
            for key in ("polar_rgb", "polar_physical_prior"):
                if key in batch:
                    inputs[key] = torch.as_tensor(batch[key]).to(device)
            if "T_camera_from_world" in batch:
                camera_from_world = torch.as_tensor(batch["T_camera_from_world"], dtype=torch.float32)
                centers = torch.as_tensor(np.stack(batch_point_centers), dtype=torch.float32)
                model_from_world = torch.eye(4, dtype=torch.float32).expand(len(centers), 4, 4).clone()
                model_from_world[:, :3, 3] = -centers
                inputs["T_camera_from_model"] = (
                    camera_from_world @ torch.linalg.inv(model_from_world).unsqueeze(1)
                ).to(device)
            elif "T_camera_from_model" in batch:
                inputs["T_camera_from_model"] = torch.as_tensor(
                    batch["T_camera_from_model"]
                ).to(device)
            else:
                raise ValueError(
                    "Polar inference requires T_camera_from_world or an already centered "
                    "T_camera_from_model; calibration is never replaced with identity"
                )
            for key in ("pixel_valid", "polar_pixel_transform"):
                if key in batch:
                    inputs[key] = torch.as_tensor(batch[key]).to(device)
        if conditioned:
            rgb_images, dense_polar, candidates = [], [], []
            for index, repo_id in enumerate(repo_ids):
                key = self.robot_config["select_video_keys"][repo_id][0]
                rgb = torch.as_tensor(batch[key][index]).float()
                if rgb.ndim == 3 and rgb.shape[-1] == 3:
                    rgb = rgb.permute(2, 0, 1)
                if rgb.max() > 1:
                    rgb = rgb / 255.0
                rgb_images.append(rgb)
                polar = torch.as_tensor(batch["polar_dense"][index]).float()
                if polar.ndim == 3 and polar.shape[-1] == 4:
                    polar = polar.permute(2, 0, 1)
                dense_polar.append(polar)
                supplied = batch.get("material_candidates")
                candidate = (supplied[index] if supplied is not None else
                             self.robot_config["material_candidates"][repo_id])
                candidates.append(torch.as_tensor(candidate, dtype=torch.float32))
            inputs["material_rgb"] = torch.stack(rgb_images).to(device)
            inputs["polar_dense"] = torch.stack(dense_polar).to(device)
            inputs["point_pixel_indices"] = torch.cat(point_indices).to(device)
            max_count = max(len(candidate) for candidate in candidates)
            material_values = torch.zeros(len(candidates), max_count, candidates[0].shape[-1])
            material_mask = torch.zeros(len(candidates), max_count, dtype=torch.bool)
            for index, candidate in enumerate(candidates):
                material_values[index, :len(candidate)] = candidate
                material_mask[index, :len(candidate)] = True
            inputs["material_candidates"] = material_values.to(device)
            inputs["material_candidate_mask"] = material_mask.to(device)

        actions, _ = model.sample_actions(
            **inputs, 
        )
        outs = self._build_action_output(repo_ids, actions.cpu(), pred_rot_type)
        for i in range(len(outs.action)):
            repo_id = repo_ids[i]
            use_delta_action = self._repo_config_flag("is_delta_action", repo_id, default=False)
            if not use_delta_action:
                outs.action[i, :, :3] += batch_point_centers[i][None, :]
        return outs



# VLAEncDec3DProcessor.register_for_auto_class()
