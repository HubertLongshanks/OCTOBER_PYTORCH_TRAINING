"""
Load the trained SmallUNet and see how it does on a few validation images.

For each image this saves a row of four pictures:
    the photo  |  the true mask  |  the model's mask  |  the model's mask drawn over the photo
and prints the IoU score for that image.

Run with:   python test.py      (after training with: python train.py)
"""

import os

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window needed
import matplotlib.pyplot as plt
import torch

from model import SmallUNet
from train import DEVICE, SAVE_PATH, PetSegmentation, iou_score, split_train_val

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NUM_IMAGES = 6  # how many validation images to look at
OUTPUT_DIR = "predictions"  # where the pictures are saved
OUTPUT_FILE = "val_predictions.png"

# The ImageNet mean and std used in train.py. We need them to undo the normalisation
# so the photos look normal again when we display them.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_displayable(image):
    """Turn a normalised (3, H, W) tensor back into an (H, W, 3) picture with values in [0, 1]."""
    image = image * STD + MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()  # matplotlib wants channels last


def main():
    print(f"Using device: {DEVICE}")

    # Rebuild the exact same validation set that train.py used (same seed, same split)
    trainval_plain = PetSegmentation(split="trainval", train=False)
    _, val_indices = split_train_val(len(trainval_plain))

    # Load the trained weights into a fresh model
    model = SmallUNet().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()  # evaluation mode: BatchNorm uses its saved statistics

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(NUM_IMAGES, 4, figsize=(12, 3 * NUM_IMAGES))
    titles = ["Photo", "True mask", "Predicted mask", "Prediction overlay"]

    for row in range(NUM_IMAGES):
        image, mask = trainval_plain[val_indices[row]]

        # The model expects a batch, so add a batch dimension: (3, H, W) -> (1, 3, H, W)
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(DEVICE)).cpu()

        iou = iou_score(logits, mask.unsqueeze(0))
        prediction = (logits[0, 0] > 0.5).float()  # positive score = pet
        print(f"Image {row + 1}: IoU = {iou:.3f}")

        photo = to_displayable(image)
        pictures = [photo, mask[0], prediction, photo]
        for col in range(4):
            ax = axes[row, col]
            if col in (1, 2):
                ax.imshow(pictures[col], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(pictures[col])
            if col == 3:
                # Draw the predicted pet area in semi-transparent red on top of the photo
                red = torch.zeros(*prediction.shape, 4)
                red[..., 0] = 1.0
                red[..., 3] = prediction * 0.4
                ax.imshow(red.numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col])
        axes[row, 2].set_title(
            f"IoU {iou:.2f}" if row else f"{titles[2]}\nIoU {iou:.2f}"
        )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=100)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
