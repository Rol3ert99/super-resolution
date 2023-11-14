import cv2
import numpy as np



def interpolation(image, scale):
    pass

original_image = cv2.imread("image_test.jpg")
scaled_image = interpolation(original_image, scale=4)

cv2.imshow("Original Image", original_image)
cv2.imshow("Scaled Image", scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()