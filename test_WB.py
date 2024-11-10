import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps

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
def gray_world(image):
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Form a grayscale image
    imagegray=image.convert('L')

    # Convert to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    imageGray=np.array(imagegray, np.float64)

    x,y = image.size

    # Get mean value of pixels
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)
    meanGray=np.mean(imageGray)

    # Gray World Algorithm
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=int(imageR[i][j]*meanGray/meanR)
            imageG[i][j]=int(imageG[i][j]*meanGray/meanG)
            imageB[i][j]=int(imageB[i][j]*meanGray/meanB)

    # Create the white balanced image
    whitebalancedIm = np.zeros((y, x, 3), dtype = "uint8")
    whitebalancedIm[:, :, 0]= imageR;
    whitebalancedIm[:, :, 1]= imageG;
    whitebalancedIm[:, :, 2]= imageB;

    # Plotting the compensated image
    #plt.figure(figsize = (20, 20))
    #plt.subplot(1, 2, 1)
    #plt.title("Compensated Image")
    #plt.imshow(image)
    #plt.subplot(1, 2, 2)
    #plt.title("White Balanced Image")
    #plt.imshow(whitebalancedIm)
    #plt.show()

    return Image.fromarray(whitebalancedIm)


 # flag = 0 for Red, Blue Compensation via green channel
# flag = 1 for Red Compensation via green channel
def compensate_RB(image, flag):
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Get maximum and minimum pixel value
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()

    # Convert to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)

    x,y = image.size

    # Normalizing the pixel value to range (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=(imageR[i][j]-minR)/(maxR-minR)
            imageG[i][j]=(imageG[i][j]-minG)/(maxG-minG)
            imageB[i][j]=(imageB[i][j]-minB)/(maxB-minB)

    # Getting the mean of each channel
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)


    # Compensate Red and Blue channel
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j]=int((imageR[i][j]+(meanG-meanR)*(1-imageR[i][j])*imageG[i][j])*maxR)
                imageB[i][j]=int((imageB[i][j]+(meanG-meanB)*(1-imageB[i][j])*imageG[i][j])*maxB)

        # Scaling the pixel values back to the original range
        for i in range(0, y):
            for j in range(0, x):
                imageG[i][j]=int(imageG[i][j]*maxG)

    # Compensate Red channel
    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j]=int((imageR[i][j]+(meanG-meanR)*(1-imageR[i][j])*imageG[i][j])*maxR)

        # Scaling the pixel values back to the original range
        for i in range(0, y):
            for j in range(0, x):
                imageB[i][j]=int(imageB[i][j]*maxB)
                imageG[i][j]=int(imageG[i][j]*maxG)

    # Create the compensated image
    compensateIm = np.zeros((y, x, 3), dtype = "uint8")
    compensateIm[:, :, 0]= imageR;
    compensateIm[:, :, 1]= imageG;
    compensateIm[:, :, 2]= imageB;

    # Plotting the compensated image
    #plt.figure(figsize = (20, 20))
    #plt.subplot(1, 2, 1)
    #plt.title("Original Image")
    #plt.imshow(image)
    #plt.subplot(1, 2, 2)
    #plt.title("RB Compensated Image")
    #plt.imshow(compensateIm)
    #plt.show()

    compensateIm=Image.fromarray(compensateIm)

    return compensateIm


# Convert the normalized image back to 8-bit format
def convert_to_uint8(img):
    # Convert PIL Image to numpy array
    img_array = np.array(img, dtype=np.float32)
    img_uint8 = img_array.astype(np.uint8)  # Rescale to [0, 255] and convert to uint8
    return img_uint8

# Example usage
im1 = cv2.imread('ImgPairs/u000101_jpg.rf.edb8b83097560dbf47c5116b0bd27e7f.jpg')
im2 = cv2.imread('ImgPairs/u000103_jpg.rf.fa94df945189d816ac49bb818475029a.jpg')

# Convert numpy array to PIL Image
im1 = Image.fromarray(im1)
im2 = Image.fromarray(im2)

# Compensate RB
im1 = compensate_RB(im1, 1)
im2 = compensate_RB(im2, 1)

# Plot the compensated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Compensated Image 1")
plt.imshow(im1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Compensated Image 2")
plt.imshow(im2)
plt.axis('off')

plt.show()

# Whie Balance Images
im1 = gray_world(im1)
im2 = gray_world(im2)

# Plot the compensated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Balanced Image 1")
plt.imshow(im1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Balanced Image 2")
plt.imshow(im2)
plt.axis('off')

plt.show()

# Convert back to uint8 before using OpenCV functions that require it
im1 = convert_to_uint8(im1)
im2 = convert_to_uint8(im2)




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