import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def preprocess_image(image_path):
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

st.set_page_config(layout="wide", page_title="AI powered superresolution")
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
MAX_FILE_SIZE = 5 * 1024 * 1024
col1, col2 = st.columns(2)


st.write("## Upload file")
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        my_upload = Image.open(my_upload)


        my_upload.save('./image.png')
        SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        IMAGE_PATH = './image.png'

        hr_image = preprocess_image(IMAGE_PATH)

        model = hub.load(SAVED_MODEL_PATH)
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)

        fake_image = tf.squeeze(fake_image)
        fake_image = np.asarray(fake_image)
        fake_image = tf.clip_by_value(fake_image, 0, 255)
        fake_image = Image.fromarray(tf.cast(fake_image, tf.uint8).numpy())

        with st.container():
          col1, col2 = st.columns(2)
          with col1:
            st.header("Before")
            col1.image(my_upload)
          
          with col2:
            st.header("After")
            st.image(fake_image)