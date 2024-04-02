import streamlit as st
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import cv2
import io

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

model = hub.load(SAVED_MODEL_PATH)

st.set_page_config(page_title="Super-Resolution App", layout='wide')

st.title('Welcome to our Super-Resolution App!')

st.markdown("""Our project is a web application that allows for enhancing the 
            resolution of images online. Utilizing advanced artificial 
            intelligence technologies, our tool can significantly increase 
            the quality of images, restoring their detail and sharpness.  
            Discover the potential of artificial intelligence and make your 
            images look better than ever before!  
            [Github](https://github.com/Rol3ert99/super-resolution)  
            * **Authors:** Robert Walery, Konrad Maciejczyk  
            * **Python libraries:** tensorflow, opencv, streamlit, numpy
            """)

st.write("## Upload file")
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)


def read_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


if my_upload is not None:
    col1.header("Before")
    col2.header("After")
    col1.image(my_upload, use_column_width=True)
    image = read_image_bytes(my_upload)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image_resolution = image.shape[:2]
    image = np.expand_dims(image, axis=0)
    
    hr_image = model(image)
    hr_image = np.array(hr_image)
    hr_image = hr_image[0,:,:,:]
    hr_image = np.clip(hr_image, 0, 255)
    hr_image = hr_image.astype(np.uint8)
    hr_image_resolution = hr_image.shape[:2]

    col2.image(hr_image, use_column_width=True)

    col1.write("Original resolution: " + str(image_resolution[0]) + 'x' + str(image_resolution[1]))

    col2.write("Received resolution: " + str(hr_image_resolution[0]) + 'x' + str(hr_image_resolution[1]))

    _, buffer = cv2.imencode('.png', cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR))
    img_io = io.BytesIO(buffer)
    col2.download_button(label='Click here to download',
                         data=img_io,
                         file_name='enhanced_image.png',
                         mime='image/png')

    
