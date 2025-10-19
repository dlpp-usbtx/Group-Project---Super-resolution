# Image Upscaling Using CNN

This project implements a simple web application for upscaling images using a Convolutional Neural Network (CNN). The application is built using Flask, a micro web framework for Python. The image upscaling is performed using a pre-trained CNN model.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python (>= 3.9)
- Flask
- TensorFlow
- OpenCV (cv2)
- Pillow (PIL)
- Werkzeug

Install the required packages using the following command:

```bash
pip install requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/image-upscaling-cnn.git
cd image-upscaling-cnn
```

2. Run the Flask application:

```bash
python app.py
```

3. Open your web browser and go to [http://localhost:5000/](http://localhost:5000/) to access the application.

## How It Works

1. **Model Loading:**
   - The Flask application loads a pre-trained image upscaling model (`image_upscale_model`) using TensorFlow.

2. **Image Processing:**
   - When an image is uploaded through the web interface, the application processes the image using the loaded model.

3. **Image Upscaling:**
   - The model performs upscaling on the input image, converting it to a higher resolution.

4. **Display Results:**
   - The original and processed images are displayed on the web interface.

## File Structure

- **app.py:** The main Flask application script.
- **models/:** Directory containing the pre-trained image upscaling model.
- **static/:** Directory for static files (CSS, JavaScript, etc.).
- **templates/:** HTML templates for rendering web pages.

## Configuration

- **ALLOWED_EXTENSIONS:** Set of allowed file extensions for image uploads (`{'png', 'jpg', 'jpeg'}`).
- **app.secret_key:** Flask secret key for session management. Replace with a secure secret key.

## Report  
- [Group-Project---Super-resolution](Group-Project---Super-resolution) 

---
