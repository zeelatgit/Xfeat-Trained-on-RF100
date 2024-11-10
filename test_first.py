''' This script demonstrates how to use the XFeat class to extract features from two images
and match them using the MNN matching algorithm. It also shows how to calculate the number of
features detected and matched, the loss, and the accuracy. Finally, it visualizes the matches
and the warped corners of the first image in the second image space. '''

import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from modules.xfeat import XFeat
from modules.dataset.augmentation import generateRandomHomography
from modules.training.losses import dual_softmax_loss, alike_distill_loss

xfeat = XFeat()

# Load the checkpoint
#checkpoint_path = 'ckpts/xfeat_synthetic_29500.pth'  # Replace with your checkpoint file path
#checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Initialize XFeat with the loaded weights
#xfeat = XFeat(weights=checkpoint['model_state_dict'])

def generateControlledRandomHomography(shape, max_displacement=30):
    h, w = shape
    src_points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    dst_points = src_points + np.random.uniform(-max_displacement, max_displacement, src_points.shape).astype(np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    return H

# Example usage
im1 = cv2.imread('ImgPairs/u000101_jpg.rf.edb8b83097560dbf47c5116b0bd27e7f.jpg')
im2 = cv2.imread('ImgPairs/u000103_jpg.rf.fa94df945189d816ac49bb818475029a.jpg')
#H1 = generateControlledRandomHomography(im1.shape[:2], max_displacement=2000)
#im2 = cv2.warpPerspective(im1, H1, (im1.shape[1], im1.shape[0]))

# Continue with feature matching and visualization as before



def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

#Use out-of-the-box function for extraction + MNN matching
mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2, top_k = 4096)

# Calculate number of features detected and matched
num_features_detected = len(mkpts_0)
num_features_matched = len(mkpts_1)

# Print the metrics
print(f'Number of features detected: {num_features_detected}')
print(f'Number of features matched: {num_features_matched}')

# Calculate losses and accuracy
features1 = torch.tensor(mkpts_0, dtype=torch.float32)
features2 = torch.tensor(mkpts_1, dtype=torch.float32)

loss, conf = dual_softmax_loss(features1, features2)
print(f'Loss: {loss.item()}')

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
plt.figure(figsize=(12,12))
plt.imshow(canvas[..., ::-1]), plt.show()