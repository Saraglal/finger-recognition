import cv2
import os

source_image = cv2.imread("test.BMP")   # The finger print that we'll be comparing to the dataset
accuracy = 0   # To calculate the matching rate
kp1 = None  # For keypoints found on the first image
kp2 = None  # For keypoints found on the second image
mp = None   # For match points between the two images

bestFile = ""   # To store the name of the file that best matches the source image to show it in the output
bestImage = None    # To store the image that best matches the source image

sift = cv2.SIFT.create()    # Using SIFT Algorithm to perform the comparison
kp1, des1 = sift.detectAndCompute(source_image, None)   # To get keypoints and descriptors of source image

# Iterate over the images in the database to perform the comparison to find the best match
for file in [file for file in os.listdir("database")][:]:
    target_image = cv2.imread("./database/" + file)

    kp2, des2 = sift.detectAndCompute(target_image, None)   # To get keypoints and descriptors for target img

    # Get all matching points of the two images using nearest neighbor
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),dict()).knnMatch(des1, des2, k=2)
    mp = []
    thresholdRatio = 0.1
    for p, q in matches:
        # Only pick the matching points that are less than the threshold ratio
        if p.distance < thresholdRatio * q.distance:
            mp.append(p)

    # Store the minimum number of keypoints of the two images
    keypoints = min(len(kp1), len(kp2)) 

    # If the image matches the target image  
    if len(mp) / keypoints * 100 > accuracy:
        accuracy = round(len(mp) / keypoints * 100, 2)
        # Storing the image info that best fit
        bestFile = file
        bestImage = target_image

# Showing the result
print('The best match :'+ bestFile)
print('Accuracy :' + str(accuracy) + "%")
result = cv2.drawMatches(source_image, kp1, bestImage, kp2, mp, None)
result = cv2.resize(result, None, fx=2.5, fy=2.5)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
