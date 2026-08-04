# ============================================================================
# data.py — turns the preprocessed fish crops into training examples.
#
# WHERE THIS FITS IN THE PROJECT:
#   scripts/build_autofish_index.py  -->  data/processed/index.csv   (one row per fish)
#   scripts/make_crops.py            -->  data/processed/crops/*.png (one masked photo per fish)
#                                              |
#                                              v
#                                THIS FILE (data.py) reads both of those
#                                              |
#                                              v
#                          train_baseline.py / train_classifier.py
#                          (they ask this file for one training example at a time)
#
# In plain words: by the time code runs in this file, all the hard image work
# (masking, cropping, resizing to squares) has ALREADY been done and saved to
# disk as PNG files. This file's only job is to load one of those PNGs, turn it
# into a PyTorch tensor (a big grid of numbers a neural network can read), and
# hand it back together with the "answer key" (the true length, or the true
# species) for that fish.
# ============================================================================

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CropDataset(Dataset):
    """A PyTorch Dataset = "a thing you can ask for item number i, and it hands
    you back one training example." PyTorch's DataLoader (used in the training
    scripts) wraps this class and automatically requests items in batches
    (e.g. 32 fish at a time), shuffles them, and runs multiple workers in
    parallel to keep the GPU fed with data.

    Every "item" this class produces is ONE fish, seen in ONE photo:
        input  = (masked square crop image, 4 bounding-box numbers)
        output = either the fish's true length in cm (regression)
                 or its species index (classification) — see `label_map` below.
    """

    def __init__(
        self,
        index_csv,       # path to data/processed/index.csv — the master spreadsheet of all fish
        crops_dir,       # path to data/processed/crops/ — the folder of pre-made PNG crops
        split,           # "train", "val", "test", or "all" — which rows of the spreadsheet to keep
        groups=None,     # optional extra filter: only keep these specific fish-group numbers
        augment=False,   # True only for training: randomly jitter colors so the model doesn't
                         # memorise exact lighting/colour and generalises better
        image_size=None, # e.g. 224 -> resize every crop to 224x224 pixels before feeding the network
        normalize_mean=None,  # per-channel (R,G,B) mean used to normalise pixel values
        normalize_std=None,   # per-channel (R,G,B) std-dev used to normalise pixel values
        label_map=None,  # optional {"cod": 0, "haddock": 1, ...}. When given, this Dataset becomes
                         # a CLASSIFICATION dataset (predict species) instead of a REGRESSION
                         # dataset (predict length). See __getitem__ below for exactly where
                         # this branches.
    ):
        # Remember whether we are in "predict species" mode or "predict length" mode.
        # Example: label_map={"cod":0,"haddock":1,...} means "classify species".
        # label_map=None (the default) means "regress length in cm".
        self.label_map = label_map

        # ---- Load the master spreadsheet (index.csv) into memory as a table ----
        # Every row = one fish seen in one photo. Columns include: annotation_id,
        # species, length_cm, bbox_x/y/w/h, width, height, split, group, set_name...
        self.df = pd.read_csv(index_csv)

        # ---- Keep only the rows belonging to this split ----
        # "split" is precomputed by build_autofish_index.py from the OFFICIAL group
        # list (train groups vs. val groups vs. test groups) — see that script for
        # exactly how a fish group gets assigned "train"/"val"/"test". Passing
        # split="all" skips this filter (used e.g. for building the classifier's
        # label map from the full dataset).
        if split != "all":
            self.df = self.df[self.df["split"] == split]

        # ---- Optional extra filter by explicit group numbers ----
        # Not normally needed (the split column already encodes the official
        # groups), but useful for one-off experiments on a subset of groups.
        if groups:
            self.df = self.df[self.df["group"].isin(groups)]

        # After filtering, row positions have gaps (e.g. 0, 5, 9, ...). Reset them
        # to be a clean 0, 1, 2, ... sequence so `self.df.iloc[idx]` below works.
        self.df = self.df.reset_index(drop=True)

        self.crops_dir = Path(crops_dir)

        # ---- Build the image preprocessing pipeline ----
        # `transforms.Compose` chains several steps together; each crop image
        # passes through them in order, left to right below.
        t = []
        if image_size:
            # Example: image_size=224 -> every crop becomes exactly 224x224 pixels,
            # because every network in this project (MobileNetV2, EfficientNet,
            # ConvNeXt, CLIP, DINOv2) expects a fixed input size.
            t.append(transforms.Resize((image_size, image_size)))

        # Convert the image from a Pillow image (pixels 0-255, shape H x W x 3)
        # into a PyTorch tensor (pixels 0.0-1.0, shape 3 x H x W). Neural network
        # layers in PyTorch expect "channels first" tensors, hence the reshuffle.
        t.append(transforms.ToTensor())

        if augment:
            # ONLY used for the training split (see train_baseline.py, which
            # passes augment=True for train_ds and augment=False for val_ds).
            # Randomly nudges brightness/contrast/saturation/hue on every epoch,
            # so the model sees slightly different lighting each time and can't
            # just memorise exact pixel values — this fights overfitting.
            # Example: the SAME fish crop might look a little darker in epoch 1
            # and a little more saturated in epoch 2, but it is still the same fish
            # with the same true length.
            t.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.5,
                    saturation=0.4,
                    hue=0.3,
                )
            )

        if normalize_mean is not None and normalize_std is not None:
            # Rescale each colour channel to roughly zero-mean, unit-variance,
            # using the same statistics the pretrained encoder (e.g. ImageNet's
            # MobileNetV2) was originally trained with. This is required so the
            # pretrained weights "see" images in the numeric range they expect —
            # skipping this step would badly hurt transfer-learning performance.
            t.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

        self.transform = transforms.Compose(t)

    def __len__(self):
        # PyTorch calls this to know how many training examples exist in total
        # (e.g. len(train_ds) == 10759 for the training split). Used internally
        # by DataLoader to know when one epoch is finished.
        return len(self.df)

    def __getitem__(self, idx):
        # PyTorch's DataLoader calls this once per index it wants, e.g.
        # dataset[0], dataset[1], ... (in a random order when shuffle=True).
        # `idx` is just a row number into our filtered table `self.df`.
        row = self.df.iloc[idx]

        # ---- Load the image: the crop file already produced by make_crops.py ----
        # Filenames are the zero-padded annotation_id, e.g. annotation_id=6721
        # -> "data/processed/crops/006721.png". This PNG is ALREADY masked
        # (background blacked out) and ALREADY a square crop of just this one
        # fish — see scripts/make_crops.py for that upstream processing step.
        image = Image.open(self.crops_dir / f"{int(row.annotation_id):06d}.png").convert("RGB")
        image = self.transform(image)  # resize -> tensor -> [jitter] -> normalize (see __init__)

        # ---- Build the 4-number "scale hint" (bounding box) ----
        # bbox_x/y/w/h in the CSV are pixel coordinates in the ORIGINAL, un-cropped
        # photo (not the crop!). Dividing by the original photo's width/height
        # squashes them into the 0..1 range, which is a scale the network can
        # learn from regardless of how big the original photos were.
        # WHY THIS MATTERS: every crop gets resized to the same 224x224 square,
        # so a tiny sardine and a huge cod would look the SAME SIZE in the crop
        # alone. These 4 numbers are how the model finds out the fish's real
        # size in the photo before cropping/resizing destroyed that information.
        # Example: bbox_x=100, width=1000 -> 100/1000 = 0.1 (the box starts 10%
        # of the way across the original photo).
        bbox = torch.tensor(
            [
                row.bbox_x / row.width,
                row.bbox_y / row.height,
                row.bbox_w / row.width,
                row.bbox_h / row.height,
            ],
            dtype=torch.float32,
        )

        # ---- Build the target ("answer key") for this fish ----
        # This is the ONE line where the dataset decides "are we doing
        # classification or regression?" — controlled entirely by whether
        # label_map was passed in (see train_classifier.py vs train_baseline.py).
        if self.label_map is not None:
            # CLASSIFICATION mode (used by train_classifier.py): look up this
            # fish's species name (e.g. "cod") in the map to get its class
            # index (e.g. 0). dtype=torch.long is required by PyTorch's
            # cross-entropy loss function.
            target = torch.tensor(self.label_map[row.species], dtype=torch.long)
        else:
            # REGRESSION mode (used by train_baseline.py): the target is simply
            # the fish's true length in centimetres, as a single-element tensor
            # (shape [1]) so it lines up with the model's single-number output.
            target = torch.tensor([row.length_cm], dtype=torch.float32)

        # ---- Extra bookkeeping info, NOT fed to the model ----
        # This dictionary rides along with every batch purely so that
        # evaluate.py / evaluate_classifier.py can write rich per-fish
        # prediction CSVs afterwards (which fish, which species, which
        # occlusion set, etc.) without needing to re-look-up the original CSV.
        # The model itself only ever sees `image` and `bbox` above.
        meta = {
            "annotation_id": int(row.annotation_id),
            "fish_id": int(row.fish_id),
            "group": int(row.group),
            "set_name": row.set_name,           # "Set1"/"Set2" (non-occluded) or "All" (occluded)
            "species": row.species,
            "length_cm": float(row.length_cm),
        }

        # Final shape handed to the training loop:
        #   (image, bbox)  -> goes INTO the model (see models.py forward() methods)
        #   target         -> what the model's output gets compared against (the loss)
        #   meta           -> bookkeeping only, used after training for analysis
        return (image, bbox), target, meta
