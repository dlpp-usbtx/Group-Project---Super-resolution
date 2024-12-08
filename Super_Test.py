import PIL
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from cv2 import dnn_superres
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Calculations
def calculate_psnr(original_img, other_img):
    other_img_resized = other_img.resize(original_img.size, resample=PIL.Image.BICUBIC)

    original_array = np.array(original_img)
    other_array = np.array(other_img_resized)

    if original_array.shape[-1] == 4:
        original_array = original_array[:, :, :3]
    if other_array.shape[-1] == 4:
        other_array = other_array[:, :, :3]

    psnr = tf.image.psnr(original_array, other_array, max_val=255.0)

    return psnr.numpy()



# Model Imports
## CNN Model
### Upscales upto x3. Small input with size 256x256 recommended to reduce comutational workload.
modl = tf.keras.models.load_model('./models/image_upscale_model')

## EDSRx4
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel("./models/pre-trained/EDSR_x4.pb")
sr.setModel("edsr", 4)


# Upscaling Methods
def upscale_interpolation(img_path, upscale_factor, max_input_size=(256, 256)):
    img = load_img(img_path)
    if img.size[0] > max_input_size[0] or img.size[1] > max_input_size[1]:
        img.thumbnail(max_input_size)
    w = int(img.size[0] * upscale_factor)
    h = int(img.size[1] * upscale_factor)
    interpolated_img = img.resize((w, h), resample=PIL.Image.BICUBIC)
    # interpolated_img = interpolated_img.resize(img.size, resample=PIL.Image.BICUBIC)

    return interpolated_img


def edsr_upscaling(superres_model, img_path, upscale_factor, max_input_size=(256, 256)):
    img = load_img(img_path)

    if img.size[0] > max_input_size[0] or img.size[1] > max_input_size[1]:
        img.thumbnail(max_input_size)

    lowres_input = img_to_array(img)
    lowres_input = lowres_input.astype("uint8")

    hr_image = superres_model.upsample(lowres_input)

    w = img.size[0] * upscale_factor
    h = img.size[1] * upscale_factor
    hr_image = cv2.resize(hr_image, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
    hr_image_pil = Image.fromarray(hr_image)

    return hr_image_pil


def model_upscaling(superres_model, img_path, max_input_size=(256, 256)):
    img = load_img(img_path)
    if img.size[0] > max_input_size[0] or img.size[1] > max_input_size[1]:
        img.thumbnail(max_input_size)

    # Convert the image to grayscale
    img = img.convert('YCbCr')
    y, cb, cr = img.split()
    y = img_to_array(y)

    y = y.astype("float32") / 255.0  # Convert to float32
    input = tf.expand_dims(y, axis=0)

    out = superres_model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )

    return out_img


if __name__ == "__main__":
    single_image_path = "./imgs/img1.jpg"
    
    # Upscale the single image
    edsr_upscaled = edsr_upscaling(sr, single_image_path, upscale_factor=3)
    cnn_upscaled = model_upscaling(modl, single_image_path)
    interpolated_image = upscale_interpolation(single_image_path, upscale_factor=3)

    # Load the original image and resize it for display
    original_image = Image.open(single_image_path)      

    # Display The Images
    original_image.show(title="Original Image")
    edsr_upscaled.show(title="EDSR Upscaled Image")
    cnn_upscaled.show(title="CNN Upscaled Image")
    interpolated_image.show(title="Interpolated Image")

    # Calculate 
    psnr_interpolated = calculate_psnr(original_image, interpolated_image)
    print(f"PSNR between original and interpolated: {psnr_interpolated}")
    
    psnr_upscaled = calculate_psnr(original_image, edsr_upscaled)
    print(f"PSNR between original and edsr_upscaled: {psnr_upscaled}")

    psnr_upscaled = calculate_psnr(original_image, cnn_upscaled)
    print(f"PSNR between original and cnn_upscaled: {psnr_upscaled}")
