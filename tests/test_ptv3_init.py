import torch

from pointact.train.ptv3_init import PTV3_INPUT_STEM_KEY, adapt_ptv3_input_stem


def test_explicit_prefix_copy_zeros_polar_columns_when_stem_shapes_match():
    source = torch.arange(36, dtype=torch.float32).reshape(4, 9)
    target = torch.full((4, 9), 123.0)
    state_dict = {PTV3_INPUT_STEM_KEY: source.clone()}

    result = adapt_ptv3_input_stem(
        state_dict,
        {PTV3_INPUT_STEM_KEY: target},
        copy_input_channels=6,
    )

    assert result == (6, 3)
    torch.testing.assert_close(state_dict[PTV3_INPUT_STEM_KEY][:, :6], source[:, :6])
    torch.testing.assert_close(state_dict[PTV3_INPUT_STEM_KEY][:, 6:], torch.zeros(4, 3))


def test_default_matching_stem_keeps_all_pretrained_channels():
    source = torch.arange(36, dtype=torch.float32).reshape(4, 9)
    state_dict = {PTV3_INPUT_STEM_KEY: source.clone()}

    result = adapt_ptv3_input_stem(
        state_dict,
        {PTV3_INPUT_STEM_KEY: torch.empty(4, 9)},
    )

    assert result is None
    torch.testing.assert_close(state_dict[PTV3_INPUT_STEM_KEY], source)


def test_xyzpolar_stem_reuses_concerto_xyzrgb_weights():
    source = torch.arange(36, dtype=torch.float32).reshape(4, 9)
    state_dict = {PTV3_INPUT_STEM_KEY: source.clone()}

    result = adapt_ptv3_input_stem(
        state_dict,
        {PTV3_INPUT_STEM_KEY: torch.empty(4, 6)},
        copy_input_channels=6,
    )

    assert result == (6, 0)
    torch.testing.assert_close(state_dict[PTV3_INPUT_STEM_KEY], source[:, :6])
