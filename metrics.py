from tensorflow import image

def psnr(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)