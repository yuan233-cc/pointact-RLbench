import copy
import unittest

import torch

from pointact.model.vla_pointact.action_head_3d.polar_router import (
    PolarTokenRouter,
    camera_from_model,
    sfp_feature_geometry,
)
from pointact.model.vla_pointact.action_head_3d.sfp_wild_encoder import (
    SfpWildFeatureEncoder,
    assemble_onlyiun_pol_vd,
    load_sfp_wild_checkpoint,
)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def make_route_point(features, coords, batch, view_valid=None, pixel_transform=None):
    batch_size, views = features.shape[:2]
    K = torch.tensor([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])
    point = AttrDict(
        polar_features=features,
        polar_K=K.expand(batch_size, views, 3, 3).clone(),
        T_camera_from_model=torch.eye(4).expand(batch_size, views, 4, 4).clone(),
        view_valid=(
            torch.ones(batch_size, views, dtype=torch.bool)
            if view_valid is None else view_valid
        ),
        polar_image_hw=torch.tensor([8, 8]).expand(batch_size, views, 2).clone(),
        coord=coords,
        batch=batch,
    )
    if pixel_transform is not None:
        point.polar_pixel_transform = pixel_transform
    return point


class SfpEncoderTest(unittest.TestCase):
    def test_official_channels_and_odd_shapes(self):
        K = torch.tensor([[10.0, 0.0, 2.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
        K = K.reshape(1, 1, 3, 3)
        value = torch.ones(1, 1, 1, 33, 35)
        image = assemble_onlyiun_pol_vd(value, value * 0.2, value * (torch.pi / 4), K)
        self.assertEqual(image.shape, (1, 1, 7, 33, 35))
        self.assertTrue(torch.allclose(image[:, :, 2], torch.zeros_like(value[:, :, 0]), atol=1e-6))
        self.assertTrue(torch.allclose(image[:, :, 3], torch.ones_like(value[:, :, 0]), atol=1e-6))
        encoder = SfpWildFeatureEncoder(residual_num=1).eval()
        with torch.no_grad():
            outputs = encoder.forward_features(image.flatten(0, 1))
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 64, 33, 35), (1, 128, 16, 17), (1, 256, 8, 8),
             (1, 512, 4, 4), (1, 512, 2, 2)],
        )

    def test_official_decoder_handles_odd_shapes_and_normalizes(self):
        encoder = SfpWildFeatureEncoder(residual_num=1).eval()
        image = torch.randn(2, 7, 33, 35)
        with torch.no_grad():
            levels, raw_normals = encoder.forward_with_normals(image)
            normalized = encoder.decode_normals(levels, normalize=True)
        self.assertEqual(tuple(raw_normals.shape), (2, 3, 33, 35))
        self.assertTrue(torch.allclose(
            normalized.norm(dim=1), torch.ones(2, 33, 35), atol=1e-5, rtol=1e-5
        ))

    def test_full_and_encoder_only_checkpoint_loading(self):
        import tempfile

        source = SfpWildFeatureEncoder(residual_num=1)
        full_state = source.state_dict()
        encoder_state = {
            key: value
            for key, value in full_state.items()
            if not key.startswith(("up1.", "up2.", "up3.", "up4.", "outc."))
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(full_state, checkpoint.name)
            report = load_sfp_wild_checkpoint(
                SfpWildFeatureEncoder(residual_num=1), checkpoint.name,
                require_decoder=True,
            )
            self.assertTrue(report["decoder_loaded"])
            self.assertEqual(report["missing_keys"], [])
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(encoder_state, checkpoint.name)
            report = load_sfp_wild_checkpoint(
                SfpWildFeatureEncoder(residual_num=1), checkpoint.name
            )
            self.assertFalse(report["decoder_loaded"])
            self.assertTrue(all(key.startswith(
                ("up1.", "up2.", "up3.", "up4.", "outc.")
            ) for key in report["missing_keys"]))


class PolarRouterTest(unittest.TestCase):
    def test_exact_feature_centers_and_filters(self):
        self.assertEqual([sfp_feature_geometry(i) for i in range(5)],
                         [(1, 0.0), (2, 0.5), (4, 1.5), (8, 3.5), (16, 7.5)])
        features = torch.arange(2 * 1 * 8 * 8 * 2, dtype=torch.float32).reshape(2, 1, 8, 8, 2)
        coords = torch.tensor([
            [0.0, 0.0, 1.0],       # image center, sample 0
            [100.0, 0.0, 1.0],     # outside
            [0.0, 0.0, -1.0],      # behind camera
            [0.0, 0.0, 1.0],       # image center, sample 1
        ])
        point = make_route_point(features, coords, torch.tensor([0, 0, 0, 1]))
        routes = PolarTokenRouter(0, neighbor_radius=0, max_tokens=32)(
            point, torch.tensor([0, 1, 2, 2, 3]), torch.tensor([4, 1])
        )
        self.assertEqual([len(route.features) for route in routes], [1, 1])
        self.assertTrue(torch.equal(routes[0].features, features[0, 0, 4, 4].reshape(1, 2)))
        self.assertTrue(torch.equal(routes[1].features, features[1, 0, 4, 4].reshape(1, 2)))

    def test_crop_resize_affine_and_missing_view(self):
        features = torch.randn(2, 1, 4, 4, 3)
        coords = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        affine = torch.eye(3).expand(2, 1, 3, 3).clone()
        affine[:, :, 0, 0] = 0.5
        affine[:, :, 1, 1] = 0.5
        point = make_route_point(
            features, coords, torch.tensor([0, 1]),
            view_valid=torch.tensor([[True], [False]]), pixel_transform=affine,
        )
        point.polar_image_hw[:] = torch.tensor([4, 4])
        routes = PolarTokenRouter(0, neighbor_radius=0, max_tokens=1)(
            point, torch.tensor([0, 1]), torch.tensor([1, 1])
        )
        self.assertEqual(len(routes[0].features), 1)
        self.assertEqual(len(routes[1].features), 0)

    def test_center_rotation_transform_composition(self):
        camera_world = torch.eye(4).reshape(1, 1, 4, 4)
        model_world = torch.eye(4).reshape(1, 4, 4)
        model_world[0, :3, 3] = torch.tensor([-1.0, -2.0, -3.0])
        camera_model = camera_from_model(camera_world, model_world)
        model_point = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(camera_model[0, 0] @ model_point,
                                       torch.tensor([1.0, 2.0, 3.0, 1.0])))


class JointAttentionTest(unittest.TestCase):
    @staticmethod
    def _point(features, polar_features, view_valid):
        from pointact.model.ptv3.concerto.structure import Point

        return Point({
            "feat": features,
            "action_feat": torch.randn(2, 3, features.shape[-1]),
            "coord": torch.tensor([
                [0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0],
                [0.0, 0.0, 1.0], [0.1, 0.0, 1.0],
            ]),
            "batch": torch.tensor([0, 0, 0, 1, 1]),
            "offset": torch.tensor([3, 5]),
            "serialized_order": torch.tensor([[0, 1, 2, 3, 4]]),
            "serialized_inverse": torch.tensor([[0, 1, 2, 3, 4]]),
            "polar_features": polar_features,
            "polar_K": torch.tensor(
                [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]
            ).expand(2, 1, 3, 3).clone(),
            "T_camera_from_model": torch.eye(4).expand(2, 1, 4, 4).clone(),
            "view_valid": view_valid,
            "polar_image_hw": torch.tensor([8, 8]).expand(2, 1, 2).clone(),
        })

    def test_empty_routes_are_numerically_baseline(self):
        from pointact.model.ptv3.concerto.model_ca_action import SerializedAttentionWithAction

        torch.manual_seed(4)
        baseline = SerializedAttentionWithAction(
            8, 2, 2, attn_drop=0.0, proj_drop=0.0, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        ).eval()
        polar = copy.deepcopy(baseline)
        polar.configure_polar(0, neighbor_radius=1, max_tokens=8)
        polar.copy_polar_qkv_()
        features = torch.randn(5, 8)
        bank = torch.randn(2, 1, 8, 8, 8)
        view_valid = torch.zeros(2, 1, dtype=torch.bool)
        point_a = self._point(features.clone(), bank.clone(), view_valid)
        point_b = self._point(features.clone(), bank.clone(), view_valid)
        point_b.action_feat.copy_(point_a.action_feat)
        out_a, out_b = baseline(point_a), polar(point_b)
        self.assertTrue(torch.equal(out_a.feat, out_b.feat))
        self.assertTrue(torch.equal(out_a.action_feat, out_b.action_feat))

    def test_polar_kv_path_has_finite_gradients(self):
        from pointact.model.ptv3.concerto.model_ca_action import SerializedAttentionWithAction

        attention = SerializedAttentionWithAction(
            8, 2, 2, attn_drop=0.0, proj_drop=0.0, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        )
        attention.configure_polar(0, neighbor_radius=0, max_tokens=8)
        attention.copy_polar_qkv_()
        features = torch.randn(5, 8, requires_grad=True)
        bank = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        point = self._point(features, bank, torch.ones(2, 1, dtype=torch.bool))
        output = attention(point)
        (output.feat.square().mean() + output.action_feat.square().mean()).backward()
        for gradient in (features.grad, bank.grad, attention.polar_qkv.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_utonia_polar_path_preserves_rope_and_has_finite_gradients(self):
        from pointact.model.ptv3.utonia.model_ca_action import (
            SerializedAttentionWithAction,
        )

        torch.manual_seed(9)
        attention = SerializedAttentionWithAction(
            12, 2, 2, attn_drop=0.0, proj_drop=0.0, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        )
        attention.configure_polar(0, neighbor_radius=0, max_tokens=8)
        attention.copy_polar_qkv_()
        features = torch.randn(5, 12, requires_grad=True)
        bank = torch.randn(2, 1, 8, 8, 12, requires_grad=True)
        point = self._point(features, bank, torch.ones(2, 1, dtype=torch.bool))
        output = attention(point)
        (output.feat.square().mean() + output.action_feat.square().mean()).backward()
        for gradient in (features.grad, bank.grad, attention.polar_qkv.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_unfrozen_sfp_adapter_and_polar_qkv_receive_action_gradient(self):
        from pointact.model.ptv3.concerto.model_ca_action import (
            PolarStagePreparation,
            SerializedAttentionWithAction,
        )

        encoder = SfpWildFeatureEncoder(residual_num=1).train()
        adapter = PolarStagePreparation(64, 8, 0)
        attention = SerializedAttentionWithAction(
            8, 2, 2, attn_drop=0.0, proj_drop=0.0, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        )
        attention.configure_polar(0, neighbor_radius=0, max_tokens=8)
        polar_images = torch.randn(2, 7, 16, 16, requires_grad=True)
        x1 = encoder.inc(polar_images).reshape(2, 1, 64, 16, 16)
        levels = (x1, None, None, None, None)
        point = self._point(
            torch.randn(5, 8), torch.empty(2, 1, 16, 16, 8),
            torch.ones(2, 1, dtype=torch.bool),
        )
        point.polar_feature_levels = levels
        point = adapter(point)
        output = attention(point)
        output.action_feat.square().mean().backward()
        gradients = (
            encoder.inc.double_conv[0].weight.grad,
            adapter.adapter.weight.grad,
            attention.polar_qkv.weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        "Flash/reference parity requires an Ampere-or-newer CUDA GPU",
    )
    def test_flash_and_reference_attention_output_and_gradient(self):
        try:
            import flash_attn
        except ImportError:
            self.skipTest("flash_attn is unavailable")
        from pointact.model.ptv3.concerto.model_ca_action import SerializedAttentionWithAction

        torch.manual_seed(8)
        module = SerializedAttentionWithAction(
            8, 2, 4, attn_drop=0.0, proj_drop=0.0, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        ).cuda().eval()
        reference_qkv = torch.randn(7, 24, device="cuda", dtype=torch.float16, requires_grad=True)
        flash_qkv = reference_qkv.detach().clone().requires_grad_(True)
        reference = module._reference_joint_attention(reference_qkv)
        flash = flash_attn.flash_attn_qkvpacked_func(
            flash_qkv.reshape(1, 7, 3, 2, 4), dropout_p=0.0, softmax_scale=module.scale
        ).reshape(7, 8)
        self.assertTrue(torch.allclose(reference, flash, atol=2e-3, rtol=2e-3))
        reference.float().square().mean().backward()
        flash.float().square().mean().backward()
        self.assertTrue(torch.allclose(reference_qkv.grad, flash_qkv.grad, atol=3e-3, rtol=3e-3))


if __name__ == "__main__":
    unittest.main()
