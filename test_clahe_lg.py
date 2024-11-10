import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from modules.xfeat import XFeat
from modules.dataset.augmentation import generateRandomHomography
from modules.training.losses import dual_softmax_loss, alike_distill_loss

#xfeat = XFeat()

# Load the checkpoint
checkpoint_path = 'ckpts/xfeat_synthetic_29500.pth'  # Replace with your checkpoint file path
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Initialize XFeat with the loaded weights
xfeat = XFeat(weights=checkpoint['model_state_dict'])


# Normalize each channel separately to avoid color imbalance
def normalize_image(image):
    # Split the image into its R, G, B channels
    channels_bgr = cv2.split(image)
    channels_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(channels_lab)

    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))



    # Normalize each channel to the range [0, 1]
    normalized_channels_lab = clahe.apply(l)
    normalized_image_lab = cv2.merge([normalized_channels_lab, a, b])

    normalized_image = cv2.cvtColor(normalized_image_lab, cv2.COLOR_LAB2BGR)

    return normalized_image


# Convert the normalized image back to 8-bit format
def convert_to_uint8(img):
    img_uint8 = (img * 255).astype(np.uint8)  # Rescale to [0, 255] and convert to uint8
    return img_uint8

# Example usage
im1 = cv2.imread('ImgPairs/u000101_jpg.rf.edb8b83097560dbf47c5116b0bd27e7f.jpg')
im2 = cv2.imread('ImgPairs/u000103_jpg.rf.fa94df945189d816ac49bb818475029a.jpg')

# Normalize images
im1 = normalize_image(im1)
im2 = normalize_image(im2)

# Convert back to uint8 before using OpenCV functions that require it
im1 = convert_to_uint8(im1)
im2 = convert_to_uint8(im2)

# Extract features using XFeat
d0 = xfeat.detectAndCompute(im1, top_k=4096)[0]
d1 = xfeat.detectAndCompute(im2, top_k=4096)[0]

# Update with image resolution (required)
d0.update({'image_size': (im1.shape[1], im1.shape[0])})
d1.update({'image_size': (im2.shape[1], im2.shape[0])})

print(f'd0: {d0}')
print(f'd1: {d1}')

# Match features using LightGlue
mkpts_0, mkpts_1 = xfeat.match_lighterglue(d0, d1)

# Debugging: Print the matched keypoints
print(f'mkpts_0: {mkpts_0}')
print(f'mkpts_1: {mkpts_1}')

# Calculate number of features detected and matched
num_features_detected = len(mkpts_0)
num_features_matched = len(mkpts_1)

# Print the metrics
print(f'Number of features detected: {num_features_detected}')
print(f'Number of features matched: {num_features_matched}')

# Ensure there are matched features before calculating loss
if num_features_detected > 0 and num_features_matched > 0:
    # Calculate losses and accuracy
    features1 = torch.tensor(mkpts_0, dtype=torch.float32)
    features2 = torch.tensor(mkpts_1, dtype=torch.float32)
    loss, conf = dual_softmax_loss(features1, features2)
    print(f'Loss: {loss.item()}')

    # Warp corners and draw matches
    def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
        H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        mask = mask.flatten()
        h, w = img1.shape[:2]
        corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_img1, H)
        img2_with_corners = img2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i-1][0].astype(int))
            end_point = tuple(warped_corners[i][0].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]
        img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None, matchColor=(0, 255, 0), flags=2)
        return img_matches

    canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1])
    plt.show()
else:
    print("No features detected or matched.")