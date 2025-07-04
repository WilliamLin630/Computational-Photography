import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

img1 = cv2.cvtColor(cv2.imread('./set1/1.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./set1/2.jpg'), cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(cv2.imread('./set1/3.jpg'), cv2.COLOR_BGR2RGB)

for i, img in enumerate([img1, img2, img3]):
    plt.figure(), plt.axis("off"), plt.title(f"Image {i + 1}")
    plt.imshow(img), plt.show()

# Part 2: Intro to Homographies
theta = np.deg2rad(10)
H_rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
H_translate = np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
H_shrink = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)

img1_rotated = cv2.warpPerspective(img1, H_rotation, (1000, 800))
img2_translated = cv2.warpPerspective(img2, H_translate, (1000, 800))
img3_shrunk = cv2.warpPerspective(img3, H_shrink, (1000, 800)) # This results in a quarter image, Chris said it was fine

plt.figure(), plt.axis("off"), plt.title("Image 1 Rotated")
plt.imshow(img1_rotated), plt.show()

plt.figure(), plt.axis("off"), plt.title("Image 2 Translated")
plt.imshow(img2_translated), plt.show()

plt.figure(), plt.axis("off"), plt.title("Image 3 Shrunk (This results in a quarter of original image, Chris said it was fine)")
plt.imshow(img3_shrunk), plt.show()

# Part 3.1: Compute SIFT features
def sift_features(img):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des, cv2.drawKeypoints(img,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp1, des1, features1 = sift_features(img1)
kp2, des2, features2 = sift_features(img2)
kp3, des3, features3 = sift_features(img3)

for i, features in enumerate([features1, features2, features3]):
    plt.figure(), plt.axis("off"), plt.title(f"Features of Image {i + 1}")
    plt.imshow(features), plt.show()

# Part 3.2: Match features
# References used: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
dist_21 = scipy.spatial.distance_matrix(des2, des1)
dist_23 = scipy.spatial.distance_matrix(des2, des3)

bf = cv2.BFMatcher()
matches_21 = bf.match(des2, des1)
matches_23 = bf.match(des2, des3)

matches_21 = sorted(matches_21, key=lambda x: x.distance)[:100]
matches_23 = sorted(matches_23, key=lambda x: x.distance)[:100]

img_21 = cv2.drawMatches(img2, kp2, img1, kp1, matches_21, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.figure(), plt.axis("off"), plt.title('100 Best Matches (Image2 to Image1)')
plt.imshow(img_21), plt.show()

img_23 = cv2.drawMatches(img2, kp2, img3, kp3, matches_23, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.figure(), plt.axis("off"), plt.title('100 Best Matches (Image2 to Image3)')
plt.imshow(img_23), plt.show()

# Part 3.3: Estimate the homographies
# References used: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
src_pts_21 = np.float32([kp1[m.trainIdx].pt for m in matches_21]).reshape(-1, 1, 2)
dst_pts_21 = np.float32([kp2[m.queryIdx].pt for m in matches_21]).reshape(-1, 1, 2)

H_1_to_2, _ = cv2.findHomography(src_pts_21, dst_pts_21, cv2.RANSAC, 2)

print("1 to 2:", H_1_to_2)

src_pts_23 = np.float32([kp3[m.trainIdx].pt for m in matches_23]).reshape(-1, 1, 2)
dst_pts_23 = np.float32([kp2[m.queryIdx].pt for m in matches_23]).reshape(-1, 1, 2)

H_3_to_2, _ = cv2.findHomography(src_pts_23, dst_pts_23, cv2.RANSAC, 2)

print("3 to 2:", H_3_to_2)

# Part 3.4 Warp and Translate images
H_translate = np.array([[1, 0, 350],[0, 1, 300],[0, 0, 1]], dtype=np.float32)

height, width = img2.shape[:2]
output_size = (width + 350, height + 300)

piece1 = cv2.warpPerspective(img1, H_translate @ H_1_to_2, output_size)
piece2 = cv2.warpPerspective(img2, H_translate, output_size)
piece3 = cv2.warpPerspective(img3, H_translate @ H_3_to_2, output_size)

for i, img in enumerate([piece1, piece2, piece3]):
    plt.figure(), plt.axis("off"), plt.title(f"Image {i+1} Warped and Translated")
    plt.imshow(img), plt.show()

panorama = np.maximum(np.maximum(piece1, piece2), piece3)

plt.figure(), plt.axis("off"), plt.title('Panorama')
plt.imshow(panorama), plt.show()