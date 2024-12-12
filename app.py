import io
import os
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


def upscale_image(model, img):
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

def upscale_multiple_times(model, img, num_iterations, upscale_factor=3):
    """
    Upscale an image multiple times using a model.
    
    Args:
        model: The trained model used for upscaling.
        img: The initial PIL Image to upscale.
        num_iterations: Number of times to upscale.
        upscale_factor: The factor by which to upscale (default is 3).
    
    Returns:
        A PIL Image that has been upscaled multiple times.
    """
    current_image = img

    for i in range(num_iterations):
        print(f"Upscaling iteration {i + 1}/{num_iterations}")
        # Upscale the image
        lowres_input = get_lowres_image(current_image, upscale_factor)
        current_image = upscale_image(model, lowres_input)

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
          # Tạo thư mục lưu ảnh nếu chưa tồn tại
          UPLOAD_FOLDER = 'static/uploads'
          if not os.path.exists(UPLOAD_FOLDER):
              os.makedirs(UPLOAD_FOLDER)

          upscale_factor = 3

          img = Image.open(io.BytesIO(file.read()))
          # lowres_input = get_lowres_image(img, upscale_factor)

          # Load model
          modl = tf.keras.models.load_model(
              '../Models/image_upscale_model/my_model.keras',
              custom_objects={"DepthToSpaceLayer": DepthToSpaceLayer}
          )

          # Upscale ảnh
          num_iterations = 1
          prediction = upscale_multiple_times(modl, img, num_iterations, upscale_factor)

          # Lưu ảnh vào thư mục
          original_output_path = os.path.join(UPLOAD_FOLDER, f"original_{file.filename}")
          prediction_output_path = os.path.join(UPLOAD_FOLDER, f"prediction_{file.filename}")

          img.save(original_output_path)
          prediction.save(prediction_output_path)

          # Truyền URL ảnh vào giao diện
          original_image_url = url_for('static', filename=f"uploads/original_{file.filename}")
          prediction_image_url = url_for('static', filename=f"uploads/prediction_{file.filename}")

          return render_template(
              'index.html',
              original_image_url=original_image_url,
              prediction_image_url=prediction_image_url,
              original_image=img,  # Để hiển thị độ phân giải gốc
              processed_image=prediction  # Để hiển thị độ phân giải upscale
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
