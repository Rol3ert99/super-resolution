from tensorflow import image

def psnr(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)