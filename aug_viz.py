import torch
import cv2
from matplotlib import pyplot as plt
from modules.dataset.augmentation import AugmentationPipe

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the image
image_path = "xfeat_tests/u000100_jpg.rf.fda1dd1f216fa1a09424164894ba0d8e.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to match expected input resolution
warp_resolution = (1200, 900)
img = cv2.resize(img, warp_resolution)

# Convert image to torch tensor
img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

# Initialize augmentation pipeline
augmentation_pipe = AugmentationPipe(device=device).to(device)

# Forward pass through the augmentation pipeline
img_tensor = img_tensor.to(device)
original, geom_transformed, photometric_transformed, details = augmentation_pipe(img_tensor, TPS=True)

# Convert tensors back to images for visualization
def tensor_to_image(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

original_img = tensor_to_image(original)
geom_trans_img = tensor_to_image(geom_transformed)
photometric_trans_img = tensor_to_image(photometric_transformed)

# Visualize the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original_img / 255.0)

plt.subplot(1, 3, 2)
plt.title("Geometrically Transformed")
plt.imshow(geom_trans_img)

plt.subplot(1, 3, 3)
plt.title("Photometrically Transformed")
plt.imshow(photometric_trans_img)

plt.tight_layout()
plt.show()
