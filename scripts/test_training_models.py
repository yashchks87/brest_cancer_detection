import importlib.util
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ('torch', 'torchvision')):
    raise unittest.SkipTest('Model tests require PyTorch and torchvision.')

import torch
from torch import nn
from torchvision import models

from scripts.training_models import build_model


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8, 1000)
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(images.shape[0])
        return self.fc(self.features(images).flatten(1))


class TinyConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.LayerNorm((8, 1, 1)), nn.Flatten(1), nn.Linear(8, 1000),
        )
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(images.shape[0])
        return self.classifier(self.features(images))


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 8
        self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.heads = nn.Linear(8, 1000)
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(images.shape[0])
        return self.heads(self.features(images))


class TrainingModelTests(unittest.TestCase):
    VIT_PRETRAINED_SIZE = 224

    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def tiny_model(self, approach='advanced', **kwargs):
        name, backbone = ('resnet18', TinyResNet) if approach == 'simple' else (
            'convnext_tiny', TinyConvNeXt,
        )
        with patch(f'scripts.training_models.models.{name}', return_value=backbone()):
            return build_model(approach, pretrained=False, **kwargs)

    def test_real_vit_shape_and_backprop_without_download(self):
        with patch('torchvision.models._api.load_state_dict_from_url',
                   side_effect=AssertionError('Unexpected pretrained download')):
            model = build_model('vit', pretrained=False, image_size=64, view_chunk_size=2)
        images = torch.randn(2, 3, 3, 64, 64, requires_grad=True)
        mask = torch.tensor([[True, False, False], [True, True, False]])
        logits = model(images, mask)
        self.assertEqual(logits.shape, (2,))
        nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
        self.assertGreater(model.encoder.conv_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.attention_score.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.classifier[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(images.grad[mask].abs().sum().item(), 0)
        self.assertEqual(images.grad[~mask].abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            reference = model(images.detach(), mask)
            model.view_chunk_size = 100
            torch.testing.assert_close(model(images.detach(), mask), reference)
            order = torch.tensor([2, 0, 1])
            torch.testing.assert_close(model(images.detach()[:, order], mask[:, order]), reference)
            changed = images.detach().clone()
            changed[~mask] = float('nan')
            torch.testing.assert_close(model(changed, mask), reference)

    def test_vit_pretrained_interpolates_position_embeddings(self):
        state = models.vit_b_16(weights=None, image_size=self.VIT_PRETRAINED_SIZE).state_dict()
        pretrained_tokens = state['encoder.pos_embedding'].shape[1]
        with patch('scripts.training_models._vit_pretrained_state', return_value=state) as source, patch(
            'torchvision.models._api.load_state_dict_from_url',
            side_effect=AssertionError('Unexpected pretrained download'),
        ):
            model = build_model('vit', pretrained=True, image_size=64)
        source.assert_called_once()
        tokens = (64 // 16) ** 2 + 1
        self.assertEqual(model.encoder.encoder.pos_embedding.shape, (1, tokens, 768))
        self.assertNotEqual(pretrained_tokens, tokens)
        self.assertEqual(state['encoder.pos_embedding'].shape[1], pretrained_tokens)
        self.assertIsInstance(model.encoder.heads, nn.Identity)
        torch.testing.assert_close(model.encoder.conv_proj.weight, state['conv_proj.weight'])

    def test_vit_pretrained_uses_standard_weights_at_224(self):
        with patch('scripts.training_models.models.vit_b_16', return_value=TinyViT()) as constructor, patch(
            'scripts.training_models._vit_pretrained_state',
            side_effect=AssertionError('No interpolation expected'),
        ):
            build_model('vit', pretrained=True, image_size=self.VIT_PRETRAINED_SIZE)
        constructor.assert_called_once_with(weights=models.ViT_B_16_Weights.DEFAULT,
                                           image_size=self.VIT_PRETRAINED_SIZE)

    def test_vit_rejects_incompatible_image_sizes(self):
        for image_size in (None, 100, 0, -16, 32, 64.0, True):
            with self.subTest(image_size=image_size), self.assertRaises(ValueError):
                build_model('vit', pretrained=False, image_size=image_size)
        for approach in ('simple', 'advanced'):
            for image_size in (63, 512.0, False):
                with self.subTest(approach=approach, image_size=image_size):
                    with self.assertRaises(ValueError):
                        build_model(approach, pretrained=False, image_size=image_size)

    def assert_real_backprop(self, approach, images, mask):
        with patch('torchvision.models._api.load_state_dict_from_url',
                   side_effect=AssertionError('Unexpected pretrained download')) as download:
            model = build_model(approach, pretrained=False)
        download.assert_not_called()
        images.requires_grad_()
        logits = model(images, mask)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())
        nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
        encoder_gradient = next(model.encoder.parameters()).grad
        self.assertIsNotNone(encoder_gradient)
        self.assertTrue(torch.isfinite(encoder_gradient).all())
        self.assertGreater(encoder_gradient.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(images.grad[mask].abs().sum().item(), 0)
        self.assertEqual(images.grad[~mask].abs().sum().item(), 0)
        classifier = model.encoder.fc if approach == 'simple' else model.classifier[-1]
        self.assertGreater(classifier.weight.grad.abs().sum().item(), 0)
        if approach == 'advanced':
            self.assertGreater(model.attention_score.weight.grad.abs().sum().item(), 0)

    def test_real_resnet_shape_and_backprop_without_download(self):
        self.assert_real_backprop(
            'simple', torch.randn(2, 1, 3, 64, 64), torch.ones(2, 1, dtype=torch.bool),
        )

    def test_real_convnext_shape_and_backprop_without_download(self):
        self.assert_real_backprop(
            'advanced', torch.randn(2, 3, 3, 64, 64),
            torch.tensor([[True, False, False], [True, False, True]]),
        )

    def test_simple_uses_sole_valid_view_at_padded_position(self):
        model = self.tiny_model('simple', view_chunk_size=1).eval()
        images = torch.randn(2, 3, 3, 64, 64)
        mask = torch.tensor([[False, True, False], [False, False, True]])
        with torch.no_grad():
            expected = model.encoder(torch.stack([images[0, 1], images[1, 2]])).squeeze(-1)
            model.encoder.batch_sizes.clear()
            actual = model(images, mask)
            self.assertEqual(model.encoder.batch_sizes, [1, 1])
            images[~mask] = float('nan')
            torch.testing.assert_close(model(images, mask), expected)
        torch.testing.assert_close(actual, expected)

    def test_simple_rejects_multiple_valid_views(self):
        images = torch.randn(2, 3, 3, 64, 64)
        masks = [
            torch.tensor([[False, True, False], [True, False, True]]),
            torch.ones(2, 3, dtype=torch.bool),
        ]
        for training in (True, False):
            model = self.tiny_model('simple').train(training)
            for index, mask in enumerate(masks):
                with self.subTest(training=training, case=index), self.assertRaisesRegex(
                    ValueError, 'exactly one valid view',
                ):
                    model(images, mask)
            self.assertEqual(model.encoder.batch_sizes, [])

    def test_masked_images_have_no_effect(self):
        images = torch.randn(2, 4, 3, 64, 64)
        for approach in ('simple', 'advanced'):
            mask = torch.tensor([[True, False, False, False], [False, True, False, False]])
            if approach == 'advanced':
                mask[1, 3] = True
            changed = images.clone()
            changed[~mask] = float('nan')
            model = self.tiny_model(approach).eval()
            with self.subTest(approach=approach), torch.no_grad():
                torch.testing.assert_close(model(images, mask), model(changed, mask))

    def test_advanced_bag_order_invariant_in_eval(self):
        model = self.tiny_model(view_chunk_size=2).eval()
        images = torch.randn(2, 4, 3, 64, 64)
        mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
        order = torch.tensor([3, 0, 2, 1])
        with torch.no_grad():
            torch.testing.assert_close(model(images, mask), model(images[:, order], mask[:, order]))

    def test_variable_view_counts_match_individual_bags(self):
        model = self.tiny_model(view_chunk_size=2).eval()
        images = torch.randn(3, 5, 3, 64, 64)
        mask = torch.tensor([
            [False, False, True, False, False],
            [True, False, True, False, True],
            [True, True, True, True, True],
        ])
        with torch.no_grad():
            batched = model(images, mask)
            self.assertEqual(model.encoder.batch_sizes, [2, 2, 2, 2, 1])
            for index in range(3):
                bag = images[index, mask[index]].unsqueeze(0)
                actual = model(bag, torch.ones(bag.shape[:2], dtype=torch.bool))
                torch.testing.assert_close(actual, batched[index:index + 1])
            model.view_chunk_size = 100
            torch.testing.assert_close(model(images, mask), batched)

    def test_training_encodes_all_valid_views_together_with_gradients(self):
        model = self.tiny_model(dropout=0, view_chunk_size=1).train()
        images = torch.randn(2, 4, 3, 64, 64, requires_grad=True)
        mask = torch.tensor([[True, False, True, False], [True, True, False, True]])
        model(images, mask).sum().backward()
        self.assertEqual(model.encoder.batch_sizes, [5])
        gradients = images.grad.flatten(2).abs().sum(dim=2)
        self.assertTrue((gradients[mask] > 0).all())
        self.assertTrue((gradients[~mask] == 0).all())

    def test_invalid_masks_rejected(self):
        images = torch.randn(2, 2, 3, 64, 64)
        invalid_masks = [
            None, [[True, True], [True, True]], torch.ones(2, 2),
            torch.ones(2, 2, dtype=torch.int64), torch.ones(2, dtype=torch.bool),
            torch.ones(2, 2, 1, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool), torch.zeros(2, 2, dtype=torch.bool),
            torch.tensor([[True, False], [False, False]]),
        ]
        for approach in ('simple', 'advanced'):
            model = self.tiny_model(approach)
            for index, mask in enumerate(invalid_masks):
                with self.subTest(approach=approach, case=index), self.assertRaises(ValueError):
                    model(images, mask)
            self.assertEqual(model.encoder.batch_sizes, [])

    def test_invalid_images_rejected(self):
        invalid_images = [
            None, torch.zeros(2, 3, 64, 64), torch.zeros(2, 2, 1, 64, 64),
            torch.zeros(2, 2, 4, 64, 64), torch.zeros(2, 2, 3, 64, 64, dtype=torch.uint8),
            torch.zeros(0, 2, 3, 64, 64), torch.zeros(2, 0, 3, 64, 64),
            torch.zeros(2, 2, 3, 0, 64),
        ]
        for approach in ('simple', 'advanced'):
            model = self.tiny_model(approach)
            for index, images in enumerate(invalid_images):
                with self.subTest(approach=approach, case=index), self.assertRaises(ValueError):
                    model(images, torch.ones(2, 2, dtype=torch.bool))

    def test_unknown_approach_rejected(self):
        for approach in ('unknown', '', 'Simple', None):
            with self.subTest(approach=approach), self.assertRaises(ValueError):
                build_model(approach, pretrained=False)

    def test_invalid_factory_arguments_rejected(self):
        invalid_arguments = [
            {'pretrained': 'false'}, {'pretrained': 1}, {'dropout': -0.1},
            {'dropout': 1}, {'dropout': float('nan')}, {'dropout': float('inf')},
            {'dropout': True}, {'dropout': '0.2'}, {'view_chunk_size': 0},
            {'view_chunk_size': -1}, {'view_chunk_size': 1.5}, {'view_chunk_size': True},
        ]
        for approach in ('simple', 'advanced'):
            for kwargs in invalid_arguments:
                with self.subTest(approach=approach, kwargs=kwargs), self.assertRaises(ValueError):
                    build_model(approach, **kwargs)

    def test_pretrained_weight_selection_without_downloading(self):
        for approach, name, backbone, weights in (
            ('simple', 'resnet18', TinyResNet, models.ResNet18_Weights.DEFAULT),
            ('advanced', 'convnext_tiny', TinyConvNeXt, models.ConvNeXt_Tiny_Weights.DEFAULT),
        ):
            for pretrained in (None, False, True):
                kwargs = {} if pretrained is None else {'pretrained': pretrained}
                with self.subTest(approach=approach, pretrained=pretrained), patch(
                    f'scripts.training_models.models.{name}', return_value=backbone(),
                ) as constructor:
                    build_model(approach, **kwargs)
                    constructor.assert_called_once_with(weights=None if pretrained is False else weights)


if __name__ == '__main__':
    unittest.main()
