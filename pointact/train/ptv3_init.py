from __future__ import annotations

import torch


PTV3_INPUT_STEM_KEY = "embedding.stem.linear.weight"


def adapt_ptv3_input_stem(
    state_dict: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    copy_input_channels: int | None = None,
) -> tuple[int, int] | None:
    """Adapt a pretrained PTv3 input stem to the requested input features.

    ``copy_input_channels`` is an explicit semantic boundary. Only that prefix
    is copied from the pretrained stem and every remaining target column is
    initialized to zero. This is needed when source and target tensors have the
    same shape but their trailing channels have different meanings, such as
    Concerto normals versus PointACT polarization features.

    Returns ``(copied_channels, zero_initialized_channels)`` when the stem was
    adapted, otherwise ``None``.
    """
    key = PTV3_INPUT_STEM_KEY
    if key not in state_dict or key not in target_state:
        return None

    source = state_dict[key]
    target = target_state[key]

    if copy_input_channels is not None:
        if copy_input_channels <= 0:
            raise ValueError("ptv3_init_copy_input_channels must be positive")
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(
                "cannot copy a PTv3 input-channel prefix from incompatible stems: "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}"
            )
        if copy_input_channels > min(source.shape[1], target.shape[1]):
            raise ValueError(
                "ptv3_init_copy_input_channels exceeds the available input channels: "
                f"requested={copy_input_channels}, source={source.shape[1]}, target={target.shape[1]}"
            )

        adapted = target.new_zeros(target.shape)
        adapted[:, :copy_input_channels] = source[:, :copy_input_channels].to(
            device=adapted.device,
            dtype=adapted.dtype,
        )
        state_dict[key] = adapted
        return copy_input_channels, target.shape[1] - copy_input_channels

    if source.shape == target.shape:
        return None
    if source.ndim == target.ndim == 2 and source.shape[0] == target.shape[0]:
        copied_channels = min(source.shape[1], target.shape[1])
        adapted = target.new_zeros(target.shape)
        adapted[:, :copied_channels] = source[:, :copied_channels].to(
            device=adapted.device,
            dtype=adapted.dtype,
        )
        state_dict[key] = adapted
        return copied_channels, target.shape[1] - copied_channels
    return None
