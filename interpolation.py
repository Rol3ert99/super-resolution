import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def interpolation(original_img, scale):
    old_h, old_w, c = original_img.shape
    new_h, new_w = original_img.shape[0] * scale, original_img.shape[1] * scale

    resized_image = np.zeros((new_h, new_w, c))

    scale_factor = 1 / scale
    for i in range(new_h):
        for j in range(new_w):
            x = i * scale_factor
            y = j * scale_factor
            
            x_floor = math.floor(x)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))

            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized_image[i,j,:] = q
    return resized_image.astype(np.uint8)

if __name__ == '__main__':
    original_image = cv2.imread("test_image.jpg")
    scaled_image = interpolation(original_image, scale = 2)

    # cv2.imshow("Original Image", original_image)
    # cv2.imshow("Scaled Image", scaled_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
    plt.show()