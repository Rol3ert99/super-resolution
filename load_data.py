import cv2
import os
from sklearn.model_selection import train_test_split

input_folder = "./images"

image_files = [f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")]

images_Y = [cv2.imread(os.path.join(input_folder, img)) for img in image_files]

images_X = [cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA) for img in images_Y]

X_train, X_test, Y_train, Y_test = train_test_split(images_X, images_Y, test_size=0.2, random_state=42)

print(len(X_train))
print(X_train[0].shape)
print(len(Y_train))
print(Y_train[0].shape)
