import io
import PIL
import cv2
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from cv2 import dnn_superres
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for, flash
from custom_layers import DepthToSpaceLayer

app = Flask(__name__)
app.secret_key = '24TNT1_nhom1'  # Replace with a secure secret key


def model_upscaling(model, imgs, max_input_size=(1200, 1200)):
    img = imgs
    if img.size[0] > max_input_size[0] or img.size[1] > max_input_size[1]:
        img.thumbnail(max_input_size)

    # Convert the image to grayscale
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )

def upscale_multiple_times(model, img, num_iterations, max_input_size=(1200, 1200)):
    """
    Upscale an image multiple times using a model.

    Args:
        model: The trained model used for upscaling.
        img: The initial PIL Image to upscale.
        num_iterations: Number of times to upscale.
        max_input_size: Maximum allowed input size for each upscale step.

    Returns:
        A PIL Image that has been upscaled multiple times.
    """
    current_image = img

    for i in range(num_iterations):
        print(f"Upscaling iteration {i + 1}/{num_iterations}")
        # Perform upscaling
        current_image = model_upscaling(model, current_image, max_input_size)

    return current_image


# Specify the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    durl = '#'  # Replace with the desired URL or leave it as '#' if not needed
    return render_template('index.html', durl=durl)

@app.route('/process_upload', methods=['GET', 'POST'])

def process_upload():
    if request.method == 'POST':
      if 'file' not in request.files:
          flash('No file part', 'danger')
          return redirect(request.url)

      file = request.files['file']

      if file.filename == '':
          flash('No selected file', 'danger')
          return redirect(request.url)

      if file and allowed_file(file.filename):
        try:
          # Read the uploaded file
          # image_stream = io.BytesIO(file.read())
          # original_image = Image.open(image_stream)

          upscale_factor = 3

          image_stream = io.BytesIO(file.read())
          img = Image.open(image_stream)
          # img = load_img(file.filename)

          lowres_input = get_lowres_image(img, upscale_factor)
          w = lowres_input.size[0] * upscale_factor
          h = lowres_input.size[1] * upscale_factor
          highres_img = img.resize((w, h))

          modl = tf.keras.models.load_model('Models/image_upscale_model/my_model.keras', custom_objects={"DepthToSpaceLayer": DepthToSpaceLayer})
          # processed_image = model_upscaling(modl, img)

          num_iterations = 5
          processed_image = upscale_multiple_times(modl, img, num_iterations)

          # Convert images to base64 for display
          original_image_stream = io.BytesIO()
          processed_image_stream = io.BytesIO()

          original_image_rgb = img.convert('RGB')
          processed_image_rgb = processed_image.convert('RGB')

          original_image_rgb.save(original_image_stream, format='JPEG')
          processed_image_rgb.save(processed_image_stream, format='JPEG')

          original_image_url = 'data:image/jpeg;base64,' + base64.b64encode(original_image_stream.getvalue()).decode()
          processed_image_url = 'data:image/jpeg;base64,' + base64.b64encode(processed_image_stream.getvalue()).decode()

          return render_template(
              'index.html',
              durl='#',
              original_image_url=original_image_url,
              processed_image_url=processed_image_url,
              original_image=img,  # Pass the actual images
              processed_image=processed_image
          )
        except Exception as e:
          flash(f'Error processing the image: {str(e)}', 'danger')
          return redirect(request.url)
      else:
        flash('Invalid file format', 'danger')
        return redirect(request.url)

    # Handle GET request if needed
    return render_template('index.html', durl='#')


if __name__ == '__main__':
    app.run(debug=True)
