import torch
import torch.nn as nn
from torchvision import models


def make_regression_head(n_inputs, hidden_layers):
    layers = []
    in_features = n_inputs
    for out_features in hidden_layers[:-1]:
        layers.extend(
            [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
            ]
        )
        in_features = out_features
    layers.append(nn.Linear(in_features, hidden_layers[-1]))
    return nn.Sequential(*layers)


class MobileNetV2Regressor(nn.Module):
    def __init__(self, bbox_input=True, pretrained=True, freeze_encoder=False, head=None):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.mobilenet_v2(weights=weights)
        if freeze_encoder:
            for param in self.features.parameters():
                param.requires_grad = False

        n_inputs = self.features.classifier[-1].in_features
        self.features.classifier = self.features.classifier[:-1]
        if bbox_input:
            n_inputs += 4
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [1000, 500, 1])

    def forward(self, batch):
        image, bbox = batch
        features = self.features(image)
        if self.bbox_input:
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)


class ConvNeXtRegressor(nn.Module):
    def __init__(self, variant="tiny", bbox_input=True, pretrained=True, freeze_encoder=False, head=None):
        super().__init__()
        if variant != "tiny":
            raise ValueError("Only convnext_tiny is currently supported")
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.convnext_tiny(weights=weights)
        if freeze_encoder:
            for param in self.features.parameters():
                param.requires_grad = False

        n_inputs = self.features.classifier[-1].in_features
        self.features.classifier[-1] = nn.Identity()
        if bbox_input:
            n_inputs += 4
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [512, 128, 1])

    def forward(self, batch):
        image, bbox = batch
        features = self.features(image)
        if self.bbox_input:
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)


class DINOv2Regressor(nn.Module):
    def __init__(
        self,
        dinov2_model="dinov2_vits14",
        feature_dim=384,
        bbox_input=True,
        freeze_encoder=True,
        trainable_blocks=0,
        head=None,
    ):
        super().__init__()
        self.encoder = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.freeze_encoder = freeze_encoder
        self.trainable_blocks = trainable_blocks
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif trainable_blocks:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if not hasattr(self.encoder, "blocks"):
                raise ValueError("DINOv2 model does not expose encoder blocks for partial fine-tuning")
            for block in self.encoder.blocks[-trainable_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, "norm"):
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True

        n_inputs = feature_dim + (4 if bbox_input else 0)
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [512, 128, 1])

    def _features(self, image):
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(image)
            if isinstance(features, dict):
                if "x_norm_clstoken" in features:
                    return features["x_norm_clstoken"]
                if "x_norm_patchtokens" in features:
                    return features["x_norm_patchtokens"].mean(dim=1)
        features = self.encoder(image)
        if features.ndim == 3:
            return features[:, 0]
        return features

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, batch):
        image, bbox = batch
        if self.freeze_encoder:
            with torch.no_grad():
                features = self._features(image)
        else:
            features = self._features(image)
        if self.bbox_input:
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)


class CLIPRegressor(nn.Module):
    def __init__(
        self,
        clip_model="ViT-B-32",
        pretrained="openai",
        feature_dim=512,
        bbox_input=True,
        freeze_encoder=True,
        trainable_blocks=0,
        head=None,
    ):
        super().__init__()
        import open_clip

        self.encoder = open_clip.create_model(clip_model, pretrained=pretrained)
        self.freeze_encoder = freeze_encoder
        self.trainable_blocks = trainable_blocks
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif trainable_blocks:
            for param in self.encoder.parameters():
                param.requires_grad = False
            visual = getattr(self.encoder, "visual", None)
            transformer = getattr(visual, "transformer", None)
            resblocks = getattr(transformer, "resblocks", None)
            if resblocks is None:
                raise ValueError("CLIP model does not expose visual transformer blocks for partial fine-tuning")
            for block in resblocks[-trainable_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(visual, "ln_post"):
                for param in visual.ln_post.parameters():
                    param.requires_grad = True
            if hasattr(visual, "proj") and visual.proj is not None:
                visual.proj.requires_grad = True

        n_inputs = feature_dim + (4 if bbox_input else 0)
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [512, 128, 1])

    def _features(self, image):
        features = self.encoder.encode_image(image)
        return features.float()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, batch):
        image, bbox = batch
        if self.freeze_encoder:
            with torch.no_grad():
                features = self._features(image)
        else:
            features = self._features(image)
        if self.bbox_input:
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)


def build_model(config):
    model_name = config["model"]
    if model_name == "mobilenet_v2":
        return MobileNetV2Regressor(
            bbox_input=config.get("bbox_input", True),
            pretrained=config.get("pretrained", True),
            freeze_encoder=config.get("freeze_encoder", False),
            head=config.get("head"),
        )
    if model_name == "dinov2":
        return DINOv2Regressor(
            dinov2_model=config.get("dinov2_model", "dinov2_vits14"),
            feature_dim=config.get("feature_dim", 384),
            bbox_input=config.get("bbox_input", True),
            freeze_encoder=config.get("freeze_encoder", True),
            trainable_blocks=config.get("trainable_blocks", 0),
            head=config.get("head"),
        )
    if model_name == "convnext":
        return ConvNeXtRegressor(
            variant=config.get("convnext_variant", "tiny"),
            bbox_input=config.get("bbox_input", True),
            pretrained=config.get("pretrained", True),
            freeze_encoder=config.get("freeze_encoder", False),
            head=config.get("head"),
        )
    if model_name == "clip":
        return CLIPRegressor(
            clip_model=config.get("clip_model", "ViT-B-32"),
            pretrained=config.get("pretrained", "openai"),
            feature_dim=config.get("feature_dim", 512),
            bbox_input=config.get("bbox_input", True),
            freeze_encoder=config.get("freeze_encoder", True),
            trainable_blocks=config.get("trainable_blocks", 0),
            head=config.get("head"),
        )
    raise ValueError(f"Unsupported model: {model_name}")
