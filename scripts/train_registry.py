from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pointact.train.pipeline_config import TrainPipelineConfig


@dataclass(frozen=True)
class TrainRecipe:
    model_class: str
    config_class: str
    processor_class: str
    data_module_fn: str
    trainer_class: str
    config_kwargs_fn: Callable[[TrainPipelineConfig], dict[str, Any]]


def _eo1_config_kwargs(training_args: TrainPipelineConfig) -> dict[str, Any]:
    return {
        "action_act": training_args.action_act,
        "max_action_dim": training_args.max_action_dim,
        "max_state_dim": training_args.max_state_dim,
        "action_chunk_size": training_args.chunk_size,
        "use_robot_state": training_args.use_robot_state,
    }


def _vla_dual_config_kwargs(training_args: TrainPipelineConfig) -> dict[str, Any]:
    return {
        "max_action_dim": training_args.max_action_dim,
        "max_state_dim": training_args.max_state_dim,
        "action_chunk_size": training_args.chunk_size,
        "use_robot_state": training_args.use_robot_state,
    }


def _point_config_kwargs(training_args: TrainPipelineConfig) -> dict[str, Any]:
    return {
        "max_action_dim": training_args.max_action_dim,
        "max_state_dim": training_args.max_state_dim,
        "action_chunk_size": training_args.chunk_size,
        "flow_matching_target": training_args.flow_matching_target,
        "flow_matching_loss": training_args.flow_matching_loss,
        "ctx_embed_size": training_args.ctx_embed_size,
        "time_embed_size": training_args.time_embed_size,
        "use_robot_state": training_args.use_robot_state,
        "ptv3_patch_size": training_args.ptv3_patch_size,
        "ptv3_enc_mode": training_args.ptv3_enc_mode,
        "ptv3_enc_channels": training_args.ptv3_enc_channels,
        "ptv3_enc_depths": training_args.ptv3_enc_depths,
        "ptv3_enc_num_head": training_args.ptv3_enc_num_head,
        "ptv3_dec_channels": training_args.ptv3_dec_channels,
        "ptv3_dec_depths": training_args.ptv3_dec_depths,
        "ptv3_dec_num_head": training_args.ptv3_dec_num_head,
        "ptv3_clf_head_pos_bins": training_args.ptv3_clf_head_pos_bins,
        "ptv3_apply_point_ca": training_args.ptv3_apply_point_ca,
        "ptv3_input_channels": training_args.ptv3_input_channels,
        "ptv3_backend": training_args.ptv3_backend,
        "polar_enabled": training_args.polar_enabled,
        "sfp_checkpoint": training_args.sfp_checkpoint,
        "sfp_freeze": training_args.sfp_freeze,
        "sfp_allow_random_init": training_args.sfp_allow_random_init,
        "sfp_feature_levels": training_args.sfp_feature_levels,
        "polar_neighbor_radius": training_args.polar_neighbor_radius,
        "polar_max_tokens_per_group": training_args.polar_max_tokens_per_group,
        "polar_max_views": training_args.polar_max_views,
        "polar_writeback": training_args.polar_writeback,
        "use_polar_material_conditioning": training_args.use_polar_material_conditioning,
        "use_target_reconstruction": training_args.use_target_reconstruction,
        "target_reconstruction_weight": training_args.target_reconstruction_weight,
        "target_mask_loss_weight": training_args.target_mask_loss_weight,
        "target_reconstruction_max_points": training_args.target_reconstruction_max_points,
        "action_regression_loss": training_args.action_regression_loss,
        "action_head_pos_center": training_args.action_head_pos_center,
        "regression_head_heatmap_temp": training_args.regression_head_heatmap_temp,
    }


EO1_RECIPE = TrainRecipe(
    model_class="pointact.model.eo1.modeling_eo1.EO1VisionFlowMatchingModel",
    config_class="pointact.model.eo1.configuration_eo1.EO1VisionFlowMatchingConfig",
    processor_class="pointact.model.eo1.processing_eo1.EO1VisionProcessor",
    data_module_fn="pointact.data.dataset.create_monolithic_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_eo1_config_kwargs,
)

VLA_DUAL_RECIPE = TrainRecipe(
    model_class="pointact.model.vla_dual.modeling_vla_dual.VLADualFlowMatchingModel",
    config_class="pointact.model.vla_dual.configuration_vla_dual.VLADualFlowMatchingConfig",
    processor_class="pointact.model.vla_dual.processing_vla_dual.VLADualProcessor",
    data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_vla_dual_config_kwargs,
)

EO1_POINT_RECIPE = TrainRecipe(
    model_class="pointact.model.eo1.modeling_eo1_point.EO1VisionPointFlowMatchingModel",
    config_class="pointact.model.eo1.configuration_eo1.EO1VisionPointFlowMatchingConfig",
    processor_class="pointact.model.eo1.processing_eo1.EO1VisionPointProcessor",
    data_module_fn="pointact.data.dataset.create_monolithic_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_point_config_kwargs,
)

VLA_DUAL_POINT_RECIPE = TrainRecipe(
    model_class="pointact.model.vla_dual.modeling_vla_dual_point.VLADualPointFlowMatchingModel",
    config_class="pointact.model.vla_dual.configuration_vla_dual.VLADualPointFlowMatchingConfig",
    processor_class="pointact.model.vla_pointact.processing_vla_pointact.VLAEncDec3DProcessor",
    data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_point_config_kwargs,
)

VLA3D_CLASSIFICATION_RECIPE = TrainRecipe(
    model_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DClassificationModel",
    config_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DModelConfig",
    processor_class="pointact.model.vla_pointact.processing_vla_pointact.VLAEncDec3DProcessor",
    data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_point_config_kwargs,
)

VLA3D_ACTION_CLASSIFICATION_RECIPE = TrainRecipe(
    model_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DWithActionClassificationModel",
    config_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DModelConfig",
    processor_class="pointact.model.vla_pointact.processing_vla_pointact.VLAEncDec3DProcessor",
    data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_point_config_kwargs,
)

VLA3D_REGRESSION_RECIPE = TrainRecipe(
    model_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DRegressionModel",
    config_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DModelConfig",
    processor_class="pointact.model.vla_pointact.processing_vla_pointact.VLAEncDec3DProcessor",
    data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_point_config_kwargs,
)

VLA3D_ACTION_REGRESSION_RECIPE = TrainRecipe(
    model_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DWithActionRegressionModel",
    config_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DModelConfig",
    processor_class="pointact.model.vla_pointact.processing_vla_pointact.VLAEncDec3DProcessor",
    data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
    trainer_class="pointact.train.trainer.VLATrainer",
    config_kwargs_fn=_point_config_kwargs,
)

TRAIN_REGISTRY = {
    "EO1VisionFlowMatchingModel": EO1_RECIPE,
    "EO1VisionPointFlowMatchingModel": EO1_POINT_RECIPE,
    "VLADualFlowMatchingModel": VLA_DUAL_RECIPE,
    "VLADualPointFlowMatchingModel": VLA_DUAL_POINT_RECIPE,
    "VLAEncDec3DClassificationModel": VLA3D_CLASSIFICATION_RECIPE,
    "VLAEncDec3DWithActionClassificationModel": VLA3D_ACTION_CLASSIFICATION_RECIPE,
    "VLAEncDec3DRegressionModel": VLA3D_REGRESSION_RECIPE,
    "VLAEncDec3DWithActionRegressionModel": VLA3D_ACTION_REGRESSION_RECIPE,
}


def _vla3d_recipe_for(model_class_name: str) -> TrainRecipe:
    return TrainRecipe(
        model_class=f"pointact.model.vla_pointact.modeling_vla_pointact.{model_class_name}",
        config_class="pointact.model.vla_pointact.modeling_vla_pointact.VLAEncDec3DModelConfig",
        processor_class="pointact.model.vla_pointact.processing_vla_pointact.VLAEncDec3DProcessor",
        data_module_fn="pointact.data.dataset.create_dual_prompt_data_module",
        trainer_class="pointact.train.trainer.VLATrainer",
        config_kwargs_fn=_point_config_kwargs,
    )


def resolve_recipe(model_class_name: str) -> TrainRecipe:
    if model_class_name not in TRAIN_REGISTRY:
        available = ", ".join(sorted(TRAIN_REGISTRY))
        raise ValueError(f"Unknown training recipe/model_class {model_class_name!r}. Available: {available}")
    return TRAIN_REGISTRY[model_class_name]
