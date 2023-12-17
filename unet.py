import tensorflow as tf

def create_unet_model():
    # encoder
    input = tf.keras.Input(shape=(48, 48, 3))
    conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(input)
    conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(conv1_1)
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1_2)
    conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(maxpool1)
    conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(conv2_1)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2_2)
    conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(maxpool2)
    conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv3_1)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv3_2)
    conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(maxpool3)
    conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv4_1)

    #decoder
    up_conv1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2), padding='same')(conv4_2)
    up_conv1 = tf.keras.layers.Concatenate(axis=3)([up_conv1, conv3_2]) 
    conv5_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(up_conv1)
    conv5_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv5_1)
    up_conv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2,2), strides=(2,2), padding='same')(conv5_2)
    up_conv2 = tf.keras.layers.Concatenate(axis=3)([up_conv2, conv2_2]) 
    conv6_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(up_conv2)
    conv6_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(conv6_1)
    up_conv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2,2), strides=(2,2), padding='same')(conv6_1)
    up_conv3 = tf.keras.layers.Concatenate(axis=3)([up_conv3, conv1_2])
    conv7_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(up_conv3)
    conv7_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(conv7_1)
    up_conv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2,2), strides=(2,2), padding='same')(conv7_2)
    conv8_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(up_conv4)
    conv8_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(conv8_1)
    output = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='same')(conv8_2)

    model_unet = tf.keras.Model(inputs=input, outputs=output)
    return model_unet

