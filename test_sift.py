import numpy as np
import cv2
import matplotlib.pyplot as plt

# Normalize each channel separately to avoid color imbalance
def normalize_image(image):
    channels_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(channels_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_channels_lab = clahe.apply(l)
    normalized_image = cv2.merge([normalized_channels_lab, a, b])
    return normalized_image

# Convert the normalized image back to 8-bit format
def convert_to_uint8(img):
    img_uint8 = (img * 255).astype(np.uint8)  # Rescale to [0, 255] and convert to uint8
    return img_uint8

# Load the images
im1 = cv2.imread('ImgPairs/u000101_jpg.rf.edb8b83097560dbf47c5116b0bd27e7f.jpg')
im2 = cv2.imread('ImgPairs/u000103_jpg.rf.fa94df945189d816ac49bb818475029a.jpg')

# Normalize images
im1 = normalize_image(im1)
im2 = normalize_image(im2)

# Convert back to uint8 before using OpenCV functions that require it
im1 = convert_to_uint8(im1)
im2 = convert_to_uint8(im2)

# Convert images to grayscale
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Initialize BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches
ref_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Check if there are enough matches to compute homography
if len(ref_points) >= 4 and len(dst_points) >= 4:
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.RANSAC, 5.0)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = im1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = im2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(im1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    # Visualize the matches
    plt.figure(figsize=(12, 12))
    plt.imshow(img_matches)
    plt.title('SIFT Feature Matches with Warped Corners')
    plt.show()
else:
    print("Not enough matches found to compute homography.")