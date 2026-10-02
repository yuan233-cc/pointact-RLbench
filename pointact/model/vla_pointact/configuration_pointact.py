from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

class VLAEncDec3DModelConfig(PretrainedConfig):
    model_type = "VLAEncDec3DModel"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        action_chunk_size=50,
        max_action_dim=32,
        max_state_dim=64,
        max_num_embodiments=1,
        use_robot_state=True,
        ctx_embed_size=512,
        num_denoise_steps=10,
        num_timestep_buckets=1000,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        flow_matching_target="x-pred",  # v-pred, x-pred
        flow_matching_loss="x-loss",    # v-loss, x-loss 
        time_embed_size=512,
        ptv3_patch_size=1024,
        ptv3_enc_mode=True,
        ptv3_enc_channels=(32, 64, 128, 256, 512),
        ptv3_enc_depths=(2, 2, 2, 6, 2),
        ptv3_enc_num_head=(2, 4, 8, 16, 32),
        ptv3_dec_channels = (64, 64, 128, 256),
        ptv3_dec_depths=(2, 2, 2, 2),
        ptv3_dec_num_head=(4, 4, 8, 16),
        ptv3_clf_head_pos_bins=100,
        ptv3_apply_point_ca=False,
        ptv3_input_channels=6,
        ptv3_backend="concerto",
        polar_enabled=False,
        polar_backbone="sfp_wild",
        sfp_checkpoint=None,
        sfp_freeze=True,
        sfp_allow_random_init=False,
        sfp_feature_levels=("x1", "x2", "x3", "x4", "x5"),
        cga_checkpoint=None,
        cga_freeze=False,
        cga_allow_random_init=False,
        cga_residual_blocks=16,
        polar_neighbor_radius=1,
        polar_max_tokens_per_group=32,
        polar_max_views=8,
        polar_token_mode="local",
        polar_writeback=False,
        use_polar_depth_self_supervision=False,
        polar_depth_loss_weight=0.1,
        polar_consistency_weight=1.0,
        sparse_depth_consistency_weight=1.0,
        depth_smoothness_weight=0.01,
        polar_depth_keep_probability=0.7,
        polar_depth_min=0.05,
        polar_depth_max=4.5,
        use_polar_material_conditioning=False,
        use_target_reconstruction=False,
        target_reconstruction_weight=0.1,
        target_mask_loss_weight=0.1,
        target_reconstruction_max_points=256,
        action_regression_loss="l2",
        action_head_pos_center="moe", 
        regression_head_heatmap_temp=0.1,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"](
                hidden_size=1280,
                out_hidden_size=2048,
                tokens_per_second=2,
            )

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.action_chunk_size = action_chunk_size
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.use_robot_state = use_robot_state

        self.ctx_embed_size = ctx_embed_size
        self.ptv3_patch_size = ptv3_patch_size
        self.ptv3_enc_mode = ptv3_enc_mode
        self.ptv3_enc_channels = ptv3_enc_channels
        self.ptv3_enc_depths = ptv3_enc_depths
        self.ptv3_enc_num_head = ptv3_enc_num_head
        self.ptv3_dec_channels = ptv3_dec_channels
        self.ptv3_dec_depths = ptv3_dec_depths
        self.ptv3_dec_num_head = ptv3_dec_num_head
        self.ptv3_input_channels = ptv3_input_channels
        self.ptv3_apply_point_ca = ptv3_apply_point_ca
        self.ptv3_backend = ptv3_backend
        self.polar_enabled = polar_enabled
        self.polar_backbone = polar_backbone
        self.sfp_checkpoint = sfp_checkpoint
        self.sfp_freeze = sfp_freeze
        self.sfp_allow_random_init = sfp_allow_random_init
        self.sfp_feature_levels = list(sfp_feature_levels)
        self.cga_checkpoint = cga_checkpoint
        self.cga_freeze = cga_freeze
        self.cga_allow_random_init = cga_allow_random_init
        self.cga_residual_blocks = cga_residual_blocks
        self.polar_neighbor_radius = polar_neighbor_radius
        self.polar_max_tokens_per_group = polar_max_tokens_per_group
        self.polar_max_views = polar_max_views
        self.polar_token_mode = polar_token_mode
        self.polar_writeback = polar_writeback
        self.use_polar_depth_self_supervision = use_polar_depth_self_supervision
        self.polar_depth_loss_weight = polar_depth_loss_weight
        self.polar_consistency_weight = polar_consistency_weight
        self.sparse_depth_consistency_weight = sparse_depth_consistency_weight
        self.depth_smoothness_weight = depth_smoothness_weight
        self.polar_depth_keep_probability = polar_depth_keep_probability
        self.polar_depth_min = polar_depth_min
        self.polar_depth_max = polar_depth_max
        if polar_enabled:
            if ptv3_backend not in ("concerto", "utonia") or not ptv3_enc_mode:
                raise ValueError(
                    "Polar joint attention supports concerto or utonia encoder-only PointACT"
                )
            if polar_backbone not in ("sfp_wild", "cga_transformer"):
                raise ValueError(
                    "polar_backbone must be 'sfp_wild' or 'cga_transformer'"
                )
            if list(sfp_feature_levels) != ["x1", "x2", "x3", "x4", "x5"]:
                raise ValueError("The supported stage mapping is exactly x1..x5")
            if polar_writeback:
                raise ValueError("polar_writeback is not implemented; Polar query outputs are discarded")
            if polar_token_mode not in ("local", "all"):
                raise ValueError("polar_token_mode must be 'local' or 'all'")
        if cga_residual_blocks < 0:
            raise ValueError("cga_residual_blocks must be non-negative")
        if use_polar_depth_self_supervision and not polar_enabled:
            raise ValueError("Polar/depth self-supervision requires polar_enabled=True")
        if use_polar_depth_self_supervision and polar_backbone != "sfp_wild":
            raise ValueError("Normal/depth self-supervision requires polar_backbone='sfp_wild'")
        if not 0 < polar_depth_min < polar_depth_max:
            raise ValueError("Expected 0 < polar_depth_min < polar_depth_max")
        if not 0 < polar_depth_keep_probability < 1:
            raise ValueError("polar_depth_keep_probability must be in (0,1)")
        self.use_polar_material_conditioning = use_polar_material_conditioning
        self.use_target_reconstruction = use_target_reconstruction
        self.target_reconstruction_weight = target_reconstruction_weight
        self.target_mask_loss_weight = target_mask_loss_weight
        self.target_reconstruction_max_points = target_reconstruction_max_points

        # classification head
        self.ptv3_clf_head_pos_bins = ptv3_clf_head_pos_bins
        # choices: moe (per point prediction), zero (directly predict action from the global feature)
        self.action_head_pos_center = action_head_pos_center 

        # regression head
        self.regression_head_heatmap_temp = regression_head_heatmap_temp

        self.action_regression_loss = action_regression_loss

        # diffusion
        self.num_denoise_steps = num_denoise_steps
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s
        self.flow_matching_target = flow_matching_target
        self.flow_matching_loss = flow_matching_loss
        self.time_embed_size = time_embed_size

        super().__init__(**kwargs)

# VLAEncDec3DModelConfig.register_for_auto_class()
