import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")
    
def preprocess_image(image_path):
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

def plot_image(image, title=""):
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
MAX_FILE_SIZE = 5 * 1024 * 1024

st.set_page_config(layout="wide", page_title="AI powered superresolution")

st.write("## Upload file")

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        my_upload = Image.open(my_upload)
        my_upload_np = np.array(my_upload)
        if my_upload_np.shape[-1] == 4:
          my_upload_np = my_upload_np[...,:-1]

        hr_size = (tf.convert_to_tensor(my_upload_np.shape[:-1]) // 4) * 4
        my_upload_np = tf.image.crop_to_bounding_box(my_upload_np, 0, 0, hr_size[0], hr_size[1])
        my_upload_np = tf.cast(my_upload_np, tf.float32)
        my_upload_np = tf.expand_dims(my_upload_np, 0)

        model = hub.load(SAVED_MODEL_PATH)
        fake_image = model(my_upload_np)

        st.image(fake_image)

