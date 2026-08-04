# ============================================================================
# models.py — defines every "brain" (encoder) we tested, all sharing one shape.
#
# WHERE THIS FITS IN THE PROJECT:
#   data.py hands out (image_crop, bbox) pairs
#                    |
#                    v
#   THIS FILE turns (image_crop, bbox) -> ONE NUMBER (length in cm) or
#                                          ONE CLASS INDEX (species)
#                    |
#                    v
#   train_baseline.py / train_classifier.py call build_model(config) to build
#   whichever network the experiment's config.json asks for, then train it.
#
# THE BIG IDEA THIS FILE IS BUILT AROUND:
#   Every model in this project is the SAME three-part machine:
#     1. ENCODER ("the eye")   — turns a photo into a list of numbers (features)
#     2. + BBOX ("the hint")   — 4 extra numbers giving the fish's real scale
#     3. HEAD ("the decision") — a small network turning features+hint into
#                                 the final answer
#   Only part 1 (the encoder) ever changes between experiments — MobileNetV2,
#   EfficientNet-B0, ConvNeXt-Tiny, CLIP, and DINOv2 are five different
#   "eyes" bolted onto the exact same "hint + decision" machinery below.
#   That is what makes the encoder comparison in this project fair: whichever
#   encoder wins, it's not because it also got a better head or extra inputs.
# ============================================================================

import torch
import torch.nn as nn
from torchvision import models


def make_regression_head(n_inputs, hidden_layers):
    """Builds the small "decision maker" network shared by every model below.

    Example: make_regression_head(1284, [1000, 500, 1]) builds:
        Linear(1284 -> 1000) -> BatchNorm -> ReLU
     -> Linear(1000 -> 500)  -> BatchNorm -> ReLU
     -> Linear(500 -> 1)                              <- final answer, no activation

    Despite the name "regression_head", this is also reused for CLASSIFICATION
    (see train_classifier.py) simply by changing the LAST number in the list —
    e.g. [512, 128, 7] outputs 7 numbers (one score per fish species) instead
    of 1 number (length in cm). The architecture code doesn't need to know or
    care which task it's being used for; only the final width differs.

    n_inputs      : how many numbers come IN (the encoder's feature vector,
                    optionally + 4 bbox numbers)
    hidden_layers : e.g. [1000, 500, 1] -> two hidden layers of size 1000 and
                    500, then a final output layer of size 1 (or 7 for 7-way
                    species classification, etc.)
    """
    layers = []
    in_features = n_inputs
    # Build every layer EXCEPT the last one with BatchNorm + ReLU after it.
    # BatchNorm keeps the numbers flowing through the network in a stable
    # range (helps training converge); ReLU is the non-linearity that lets a
    # stack of Linear layers do more than just one big linear transform.
    for out_features in hidden_layers[:-1]:
        layers.extend(
            [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
            ]
        )
        in_features = out_features
    # The FINAL layer has no BatchNorm/ReLU after it — for regression we want
    # a plain, unrestricted number (length can be any positive value); for
    # classification these are "logits" that get turned into probabilities
    # later by the loss function (CrossEntropyLoss does that internally).
    layers.append(nn.Linear(in_features, hidden_layers[-1]))
    return nn.Sequential(*layers)


class MobileNetV2Regressor(nn.Module):
    """Encoder #1: MobileNetV2 — the AutoFish paper's original baseline "eye".

    Layman: a compact, efficient CNN originally designed to run on phones.
    It's the "apprentice" we train specifically for this one job (full
    fine-tuning is the default — see freeze_encoder below).

    Technical: built from inverted-residual blocks (expand -> depthwise conv
    -> project), ~3.4M parameters, ImageNet-pretrained.
    """

    def __init__(self, bbox_input=True, pretrained=True, freeze_encoder=False, head=None):
        super().__init__()
        # Load MobileNetV2 with ImageNet weights (unless pretrained=False, in
        # which case it starts from random weights — not used in this project,
        # but supported for completeness/ablations).
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.mobilenet_v2(weights=weights)

        if freeze_encoder:
            # "Freezing" = locking these weights so they never change during
            # training; only the small head (self.classifier, built below)
            # learns. Used for foundation-model "frozen" experiments; the
            # baseline itself always trains the whole encoder (freeze_encoder
            # defaults to False for MobileNetV2).
            for param in self.features.parameters():
                param.requires_grad = False

        # torchvision's MobileNetV2 ships with its OWN 1000-way ImageNet
        # classifier head. We don't want that — we want OUR OWN small head
        # that outputs a length (or species) — so we:
        #   1. read how many features the encoder produces just before its
        #      original head (n_inputs, e.g. 1280 for MobileNetV2)
        n_inputs = self.features.classifier[-1].in_features
        #   2. chop the original 1000-class head OFF, keeping only the part
        #      of MobileNetV2 that produces the 1280-length feature vector
        self.features.classifier = self.features.classifier[:-1]

        if bbox_input:
            # Reserve 4 extra input slots for the bounding-box "scale hint"
            # (see data.py for exactly how those 4 numbers are computed).
            n_inputs += 4
        self.bbox_input = bbox_input

        # Build our own small head. Default shape [1000, 500, 1] matches the
        # AutoFish paper's original baseline recipe (see configs/baseline_official.json).
        self.classifier = make_regression_head(n_inputs, head or [1000, 500, 1])

    def forward(self, batch):
        # `forward` is what actually runs when you call `model((image, bbox))`.
        # This is the exact sequence of operations for one batch of fish:
        image, bbox = batch
        features = self.features(image)          # photo -> 1280 numbers (the "eye" looking)
        if self.bbox_input:
            # torch.cat glues the bbox's 4 numbers onto the end of the 1280
            # feature numbers -> one combined vector of length 1284.
            # Example: features=[0.12, -0.4, ..., 0.05] (1280 values) and
            # bbox=[0.10, 0.22, 0.30, 0.18] (4 values) become one 1284-long vector.
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)          # 1284 numbers -> 1 final number (length in cm)


class ConvNeXtRegressor(nn.Module):
    """Encoder #2: ConvNeXt-Tiny — a modern (2022) CNN.

    Layman: borrows good ideas from Transformers (bigger filters, different
    normalisation) while staying a "classic" convolutional design underneath.

    Technical: ~28M parameters, large-kernel depthwise convolutions,
    LayerNorm instead of BatchNorm internally, ImageNet-pretrained.
    Same wiring pattern as MobileNetV2Regressor above — only the backbone differs.
    """

    def __init__(self, variant="tiny", bbox_input=True, pretrained=True, freeze_encoder=False, head=None):
        super().__init__()
        if variant != "tiny":
            # Only the "tiny" size of ConvNeXt is wired up in this project;
            # larger variants (small/base/large) would need their own weight
            # enum here and were left as future work.
            raise ValueError("Only convnext_tiny is currently supported")
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.convnext_tiny(weights=weights)
        if freeze_encoder:
            for param in self.features.parameters():
                param.requires_grad = False

        # Same "remove the built-in ImageNet head" trick as MobileNetV2 above,
        # but ConvNeXt's classifier is structured differently internally, so
        # instead of slicing off the last layer we replace it with
        # nn.Identity() — a "do nothing" layer that just passes its input
        # through unchanged, effectively deleting that layer's effect.
        n_inputs = self.features.classifier[-1].in_features
        self.features.classifier[-1] = nn.Identity()
        if bbox_input:
            n_inputs += 4
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [512, 128, 1])

    def forward(self, batch):
        # Identical pattern to MobileNetV2Regressor.forward — this is exactly
        # why the comparison between encoders is fair: every model funnels
        # through this same "features -> concat bbox -> small head" recipe.
        image, bbox = batch
        features = self.features(image)
        if self.bbox_input:
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)


class EfficientNetRegressor(nn.Module):
    """Encoder #3: EfficientNet-B0 — a CNN scaled "in balance".

    Layman: instead of just making a network deeper OR wider OR fed bigger
    images (as older designs did somewhat arbitrarily), EfficientNet scales
    all three together by a single "compound" formula, aiming for a sweet
    spot of accuracy per parameter.

    Technical: ~5.3M parameters, MBConv blocks (mobile inverted bottleneck,
    similar family to MobileNetV2's blocks but with squeeze-and-excitation
    attention added), ImageNet-pretrained. Added to this project because it
    came out closest to the baseline (see docs/ODE_REPORT.md).
    """

    def __init__(self, variant="b0", bbox_input=True, pretrained=True, freeze_encoder=False, head=None):
        super().__init__()
        if variant != "b0":
            raise ValueError("Only efficientnet_b0 is currently supported")
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.efficientnet_b0(weights=weights)
        if freeze_encoder:
            for param in self.features.parameters():
                param.requires_grad = False

        n_inputs = self.features.classifier[-1].in_features
        self.features.classifier = nn.Identity()  # delete EfficientNet's built-in ImageNet head
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
    """Encoder #4: DINOv2 ViT-S/14 — a self-supervised Vision Transformer.

    Layman: unlike the three CNNs above, DINOv2 never saw a single human
    label during its pretraining — it taught itself to understand images
    purely by comparing different "views" of the same picture. It also works
    completely differently internally: instead of scanning with small sliding
    filters (like a CNN), it chops the image into a grid of patches and lets
    every patch "look at" every other patch (self-attention).

    Technical: ~21M-parameter Vision Transformer (ViT-S/14, patch size 14),
    DINO/iBOT self-supervised objective, no ImageNet labels used.

    Because this is a Transformer, not a CNN, its `_features()` method below
    looks quite different from the CNN encoders above — a ViT naturally
    produces MANY feature vectors (one per patch, plus one special [CLS]
    "summary" vector), and we have to explicitly choose which of those we
    hand to the regression head. That choice (CLS token vs. patch tokens) is
    one of the most important experiments in this whole project (see
    `use_patch_tokens` below and docs/DEFENSE_A_TO_Z_LAYMAN_TECHNICAL_GUIDE.md §2.4).
    """

    def __init__(
        self,
        dinov2_model="dinov2_vits14",  # which DINOv2 size to download from Meta's model hub
        feature_dim=384,               # length of the feature vector this model produces
        bbox_input=True,
        freeze_encoder=True,           # DINOv2 defaults to FROZEN (unlike the CNNs above,
                                        # which default to fully trainable) — foundation models
                                        # are usually evaluated frozen first to test whether their
                                        # pretrained features are already useful "as is".
        trainable_blocks=0,            # >0 -> unfreeze only the LAST N transformer blocks
                                        # ("partial fine-tuning" / "last-block fine-tuning")
        head=None,
        use_patch_tokens=False,        # False (default) = use the [CLS] summary token.
                                        # True = average all the patch tokens instead — this is
                                        # the change that produced our best DINOv2 result.
    ):
        super().__init__()
        # Downloads (and caches) the pretrained DINOv2 weights straight from
        # Meta's research GitHub repo via PyTorch Hub — no local weight file needed.
        self.encoder = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.freeze_encoder = freeze_encoder
        self.trainable_blocks = trainable_blocks
        self.use_patch_tokens = use_patch_tokens

        if freeze_encoder:
            # Fully frozen: lock every weight, and also put the encoder into
            # eval() mode (see the custom train() override below for why that
            # matters — BatchNorm/Dropout-style layers behave differently in
            # train vs eval mode, and a frozen encoder should always act as if
            # it's in "evaluation" mode).
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif trainable_blocks:
            # PARTIAL fine-tuning ("last-block fine-tuning"): start by freezing
            # everything...
            for param in self.encoder.parameters():
                param.requires_grad = False
            if not hasattr(self.encoder, "blocks"):
                raise ValueError("DINOv2 model does not expose encoder blocks for partial fine-tuning")
            # ...then selectively UNfreeze just the last `trainable_blocks`
            # transformer blocks. Example: trainable_blocks=1 unfreezes only
            # the very last block, leaving all earlier blocks locked. This is
            # a middle ground between "fully frozen" and "fully fine-tuned":
            # it lets the model adapt a little to fish without destroying all
            # of its pretrained general-purpose knowledge.
            for block in self.encoder.blocks[-trainable_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Also unfreeze the final normalisation layer, since it directly
            # feeds into whichever token(s) we read out below.
            if hasattr(self.encoder, "norm"):
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True
        # (else: freeze_encoder=False and trainable_blocks=0 means FULL
        # fine-tuning — every weight in the encoder stays trainable.)

        n_inputs = feature_dim + (4 if bbox_input else 0)
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [512, 128, 1])

    def _features(self, image):
        """Turns a batch of images into a batch of feature vectors — this is
        the ViT-specific logic that has no equivalent in the CNN classes above.

        A DINOv2 forward pass produces a dictionary of different token types:
          - "x_norm_clstoken"    : ONE vector per image (the [CLS] summary token)
          - "x_norm_patchtokens" : MANY vectors per image (one per 14x14 patch,
                                    e.g. a 224x224 image -> a 16x16 grid = 256 patches)
        We must collapse this down to exactly ONE vector per image before it
        can be concatenated with the 4 bbox numbers and fed to the head.
        """
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(image)
            if isinstance(features, dict):
                # use_patch_tokens: mean-pool the spatial patch tokens (keeps local
                # geometry, better suited to size/length than the global CLS token).
                # Example: 256 patch vectors of length 384 each -> average them
                # element-wise -> ONE vector of length 384, but built from
                # information that still "knows" roughly where in the image
                # each contribution came from (unlike the CLS token, which is
                # a single learned summary optimised for "what is this?").
                if self.use_patch_tokens and "x_norm_patchtokens" in features:
                    return features["x_norm_patchtokens"].mean(dim=1)
                if "x_norm_clstoken" in features:
                    # Default path: just take the ready-made [CLS] summary vector.
                    return features["x_norm_clstoken"]
                if "x_norm_patchtokens" in features:
                    # Fallback if a future DINOv2 variant has no CLS token at all.
                    return features["x_norm_patchtokens"].mean(dim=1)
        # Fallback path for encoders that don't expose forward_features() at
        # all (kept generic in case a different ViT implementation is swapped in).
        features = self.encoder(image)
        if features.ndim == 3:
            return features[:, 0]  # assume token 0 is the CLS-like summary token
        return features

    def train(self, mode=True):
        # PyTorch calls model.train() before every training epoch and
        # model.eval() before validation/testing. We override this so that,
        # even when the OUTER model is asked to go into "train" mode, a
        # FROZEN encoder is forced back into "eval" mode underneath — this
        # matters because some layers (e.g. certain normalisation layers)
        # behave differently in train vs eval mode, and a frozen encoder
        # should never flip that behaviour just because the head is training.
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, batch):
        image, bbox = batch
        if self.freeze_encoder:
            # torch.no_grad() tells PyTorch "don't bother tracking how to
            # compute gradients through this part" — since the encoder's
            # weights never change (frozen), this saves memory and computation.
            with torch.no_grad():
                features = self._features(image)
        else:
            features = self._features(image)
        if self.bbox_input:
            features = torch.cat([features, bbox], dim=1)
        return self.classifier(features)


class CLIPRegressor(nn.Module):
    """Encoder #5: CLIP ViT-B/32 — a vision-LANGUAGE Transformer.

    Layman: CLIP learned by looking at ~400 million pictures paired with their
    internet captions, and training itself to match the right picture to the
    right caption. Its "understanding" of images is shaped by language, which
    turns out to transfer to our fish-length task much better than DINOv2's
    purely self-supervised (no language) features do.

    Technical: OpenCLIP implementation, ViT-B/32 image encoder (~88M
    parameters for the image tower), contrastive image-text pretraining. We
    only use its IMAGE encoder (`encode_image`) — the text side of CLIP is
    irrelevant to this regression task and is simply never called.
    """

    def __init__(
        self,
        clip_model="ViT-B-32",     # which CLIP architecture/size
        pretrained="openai",       # which pretrained weight set to load
        feature_dim=512,
        bbox_input=True,
        freeze_encoder=True,       # like DINOv2, CLIP defaults to frozen
        trainable_blocks=0,        # partial fine-tuning, same idea as DINOv2 above
        head=None,
    ):
        super().__init__()
        import open_clip  # imported here (not at the top of the file) so that

        # projects/environments that never use CLIP don't need the
        # `open_clip` package installed just to import this models.py file.

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
            # CLIP's internal naming is different from DINOv2's: the image
            # side lives under `.visual`, and its Transformer blocks are
            # called `.transformer.resblocks` (residual blocks) rather than
            # just `.blocks`. We have to walk down that specific path to find
            # the last few blocks to unfreeze.
            visual = getattr(self.encoder, "visual", None)
            transformer = getattr(visual, "transformer", None)
            resblocks = getattr(transformer, "resblocks", None)
            if resblocks is None:
                raise ValueError("CLIP model does not expose visual transformer blocks for partial fine-tuning")
            for block in resblocks[-trainable_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Also unfreeze the final layer-norm and the output projection
            # (CLIP-specific final steps that shape the 512-d feature vector
            # we actually read out in _features() below).
            if hasattr(visual, "ln_post"):
                for param in visual.ln_post.parameters():
                    param.requires_grad = True
            if hasattr(visual, "proj") and visual.proj is not None:
                visual.proj.requires_grad = True

        n_inputs = feature_dim + (4 if bbox_input else 0)
        self.bbox_input = bbox_input
        self.classifier = make_regression_head(n_inputs, head or [512, 128, 1])

    def _features(self, image):
        # CLIP conveniently exposes a single ready-made method for "turn an
        # image into its embedding vector" — much simpler than DINOv2's
        # dictionary-of-token-types above, because CLIP was designed from the
        # start to produce one summary vector per image (to compare against
        # one summary vector per caption).
        features = self.encoder.encode_image(image)
        # OpenCLIP can internally use half-precision (float16) for speed;
        # .float() makes sure we always get back float32, matching the rest
        # of our pipeline (bbox tensor, loss function, etc.).
        return features.float()

    def train(self, mode=True):
        # Same reasoning as DINOv2Regressor.train() above: keep a frozen
        # encoder pinned in eval() mode even while the outer model trains.
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
    """The single entry point every training/evaluation script calls.

    WHERE THIS IS CALLED FROM:
        train_baseline.py, evaluate.py, train_classifier.py, evaluate_classifier.py
        all do:  model = build_model(config)
        where `config` is the dict loaded from one configs/<experiment>.json file.

    In other words: this function is the "switchboard" that reads the
    `"model"` field of a config file (e.g. "mobilenet_v2", "dinov2", ...) and
    constructs the matching class above with all the right settings pulled
    out of that same config file. Adding a brand-new encoder to this project
    means: (1) write a new `<Something>Regressor` class above with the same
    forward(batch) contract, (2) add one more `if model_name == "...":` branch
    here.

    config.get("key", default) reads a setting from the JSON file if present,
    otherwise falls back to a sensible default — so a config file only needs
    to specify the settings it wants to override.
    """
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
            use_patch_tokens=config.get("use_patch_tokens", False),
        )
    if model_name == "convnext":
        return ConvNeXtRegressor(
            variant=config.get("convnext_variant", "tiny"),
            bbox_input=config.get("bbox_input", True),
            pretrained=config.get("pretrained", True),
            freeze_encoder=config.get("freeze_encoder", False),
            head=config.get("head"),
        )
    if model_name == "efficientnet":
        return EfficientNetRegressor(
            variant=config.get("efficientnet_variant", "b0"),
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
    # Safety net: if a config.json has a typo in its "model" field or asks
    # for an encoder that was never implemented, fail loudly and immediately
    # rather than silently doing the wrong thing.
    raise ValueError(f"Unsupported model: {model_name}")
