"""
Train a SmallUNet to separate pets from the background (binary segmentation).

Dataset: Oxford-IIIT Pet (about 7,400 photos of cats and dogs).
Every photo comes with a "trimap" mask that labels each pixel:
    1 = pet,  2 = background,  3 = the border around the pet
We turn this into a simple yes/no mask: 1 = pet (including its border), 0 = background.

Run with:   python train.py
"""

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet

from model import SmallUNet

# ---------------------------------------------------------------------------
# Settings: change these to experiment
# ---------------------------------------------------------------------------
DATA_DIR = "./data"  # where the dataset is downloaded to
IMAGE_SIZE = (
    128  # images are resized to IMAGE_SIZE x IMAGE_SIZE (must be divisible by 8)
)
BATCH_SIZE = 32  # how many images the model sees at once
EPOCHS = 20  # how many full passes over the training data
LEARNING_RATE = 1e-3  # how big each update step is
VAL_FRACTION = 0.15  # fraction of the training images held back for validation
SEED = 0  # fixes the random train/validation split so it's the same every run
SAVE_PATH = "unet_pet.pt"  # where the best model weights are saved

# Use Apple's GPU (MPS) if available, then an NVIDIA GPU (CUDA), otherwise the CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class PetSegmentation(Dataset):
    """
    Wraps torchvision's OxfordIIITPet dataset and returns, for each sample:
        image: float tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE)
        mask:  float tensor of shape (1, IMAGE_SIZE, IMAGE_SIZE), 1 = pet, 0 = background
    """

    def __init__(self, split, train):
        # split is "trainval" (for training) or "test" (for checking performance)
        self.data = OxfordIIITPet(
            DATA_DIR, split=split, target_types="segmentation", download=True
        )
        self.train = train  # only use random augmentation during training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, trimap = self.data[index]  # both are PIL images

        # Resize. The mask uses NEAREST so pixel labels stay as whole numbers (1, 2 or 3).
        image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        trimap = TF.resize(
            trimap, [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.NEAREST
        )

        # Data augmentation: randomly mirror half the training images.
        # The image and its mask must get the SAME flip so they still line up.
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)

        # Convert the image to a tensor with values in [0, 1], then normalise
        # with the standard ImageNet mean and std
        image = TF.to_tensor(image)
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Turn the trimap into a binary mask: anything that isn't background (2) is pet
        trimap = TF.pil_to_tensor(trimap)
        mask = (trimap != 2).float()

        return image, mask


def split_train_val(num_images):
    """
    Shuffle the image indices (with a fixed seed) and cut off the first VAL_FRACTION for validation.
    Because the seed is fixed, every script that calls this gets the SAME split.
    Returns (train_indices, val_indices).
    """
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(num_images, generator=generator).tolist()
    num_val = int(num_images * VAL_FRACTION)
    return indices[num_val:], indices[:num_val]


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------
def dice_loss(logits, masks):
    """
    Dice loss measures how much the predicted pet region overlaps the true one.
    0 = perfect overlap, 1 = no overlap. It complements BCE by focusing on the
    pet shape as a whole rather than on each pixel separately.
    """
    probs = torch.sigmoid(logits)  # turn scores into probabilities between 0 and 1
    overlap = (probs * masks).sum()
    total = probs.sum() + masks.sum()
    return 1 - (2 * overlap + 1) / (total + 1)  # the +1 avoids dividing by zero


def iou_score(logits, masks):
    """
    Intersection over Union (IoU): the area where prediction AND truth are pet,
    divided by the area where prediction OR truth is pet. 1.0 is perfect.
    """
    preds = (logits > 0.5).float()  # a positive score means the model predicts "pet"
    intersection = (preds * masks).sum()
    union = ((preds + masks) > 0.5).float().sum()
    return (intersection / union).item()


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()  # turn on training behaviour (e.g. BatchNorm updates its statistics)
    total_loss = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # 1. Forward pass: get the model's predictions
        logits = model(images)

        # 2. Measure how wrong they are. BCE scores each pixel; Dice scores the overall shape.
        loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(
            logits, masks
        )

        # 3. Backward pass: work out how to change each weight to reduce the loss
        optimizer.zero_grad()
        loss.backward()

        # 4. Update the weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()  # no gradients needed when we're only evaluating, which saves memory
def evaluate(model, loader):
    model.eval()  # turn off training behaviour
    total_iou = 0
    total_accuracy = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        preds = (logits > 0.5).float()

        total_iou += iou_score(logits, masks)
        total_accuracy += (preds == masks).float().mean().item()

    return total_iou / len(loader), total_accuracy / len(loader)


def main():
    print(f"Using device: {DEVICE}")

    # Load the data. We split the "trainval" images into three roles:
    #   train      - the model learns from these
    #   validation - checked after every epoch to choose the best weights
    #   test       - used only ONCE at the very end, for an honest final score
    # If we chose the best epoch using the test set, the test score would be
    # slightly too optimistic, because we'd have picked the epoch that suited it best.
    trainval_augmented = PetSegmentation(
        split="trainval", train=True
    )  # with random flips
    trainval_plain = PetSegmentation(
        split="trainval", train=False
    )  # no flips, for validation
    test_data = PetSegmentation(split="test", train=False)

    train_indices, val_indices = split_train_val(len(trainval_plain))
    train_data = Subset(trainval_augmented, train_indices)
    val_data = Subset(trainval_plain, val_indices)
    print(
        f"Training images: {len(train_data)}, validation images: {len(val_data)}, test images: {len(test_data)}"
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Create the model and optimizer
    model = SmallUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # The training loop
    best_iou = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_iou, val_accuracy = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val IoU: {val_iou:.4f} | "
            f"val pixel accuracy: {val_accuracy:.4f}"
        )

        # Keep the weights from the best epoch so far (judged on the validation set)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Best val IoU: {best_iou:.4f}, weights saved to {SAVE_PATH}")

    # Final check: load the best weights and score them on the test set, once
    model.load_state_dict(torch.load(SAVE_PATH))
    test_iou, test_accuracy = evaluate(model, test_loader)
    print(f"Test IoU: {test_iou:.4f} | test pixel accuracy: {test_accuracy:.4f}")


# This makes sure main() only runs when you start this file directly,
# which DataLoader workers need on macOS
if __name__ == "__main__":
    main()
