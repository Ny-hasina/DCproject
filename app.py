from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/images/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('image_tool.html')

# Image upload and manipulation route
@app.route('/upload', methods=['POST'])
def upload_image():
    print("Upload route called")  # Debugging
    if 'file' not in request.files:
        print("No file part in request")  # Debugging
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        print("No file selected")  # Debugging
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        print("File is valid")  # Debugging
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")  # Debugging
        file.save(filepath)

        # Load the image
        image = cv2.imread(filepath)
        if image is None:
            print("Error: Could not read image file")  # Debugging
            return redirect(request.url)

        # Apply the selected filter
        filter_type = request.form.get('filter')
        print(f"Applying filter: {filter_type}")  # Debugging
        if filter_type == 'pixelate':
            processed_image = pixelate_filter(image)
        elif filter_type == 'rainbow':
            processed_image = rainbow_filter(image)
        elif filter_type == 'mirror':
            processed_image = mirror_filter(image)
        elif filter_type == 'cartoonify':
            processed_image = cartoonify_filter(image)
        else:
            processed_image = image  # Default: no filter

        # Save the processed image
        cv2.imwrite(filepath, processed_image)
        print(f"Processed image saved to: {filepath}")  # Debugging

        return render_template('image_tool.html', filename=filename)
    
    print("File upload failed")  # Debugging
    return redirect(request.url)

# Pixelate Filter
def pixelate_filter(image):
    height, width = image.shape[:2]
    # Resize to a smaller version
    small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    # Resize back to the original size
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated

# Rainbow Filter
def rainbow_filter(image):
    # Create a rainbow gradient
    rainbow = np.zeros_like(image)
    height, width = image.shape[:2]
    for i in range(height):
        rainbow[i, :] = [int(255 * i / height), int(255 * (1 - i / height)), 255]
    # Blend the rainbow with the image
    blended = cv2.addWeighted(image, 0.7, rainbow, 0.3, 0)
    return blended

# Mirror Effect
def mirror_filter(image):
    height, width = image.shape[:2]
    # Flip the left half of the image
    image[:, :width // 2] = cv2.flip(image[:, width // 2:], 1)
    return image

# Cartoonify Filter
def cartoonify_filter(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply median blur to reduce noise
    gray = cv2.medianBlur(gray, 5)
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Apply bilateral filter for cartoon effect
    color = cv2.bilateralFilter(image, 9, 300, 300)
    # Combine edges with the color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

#plot part
@app.route('/visualization')
def visualization():
    data = load_data()
    create_visualization(data)
    return render_template('visualization.html')
# Load the dataset
def load_data():
    data = pd.read_csv('data/temperatures.csv')
    return data

# Create a visualization
def create_visualization(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['temperature'], marker='o', color='b', linestyle='-')
    plt.title('Daily Temperatures')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.savefig('static/images/temperature_plot.png')  # Save the plot
    plt.close()

  
if __name__ == '__main__':
    app.run(debug=True, port=5001)