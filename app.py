from flask import Flask, render_template, request, redirect, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from PIL import Image
import random
import turtle
import pyaudio

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

# Base Shape Class
class Shape:
    def __init__(self, color, x, y, size):
        self.color = color
        self.x = x
        self.y = y
        self.size = size

    def draw(self, artist):
        artist.penup()
        artist.goto(self.x, self.y)
        artist.pendown()
        artist.color(self.color)

    def resize(self, new_size):
        self.size = new_size

# Circle subclass
class Circle(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        artist.circle(self.size)
        artist.end_fill()

# Square subclass
class Square(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(4):
            artist.forward(self.size)
            artist.left(90)
        artist.end_fill()

# Rectangle subclass
class Rectangle(Shape):
    def __init__(self, color, x, y, width, height):
        super().__init__(color, x, y, width)
        self.height = height

    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(2):
            artist.forward(self.size)  # width
            artist.left(90)
            artist.forward(self.height)  # height
            artist.left(90)
        artist.end_fill()

# Triangle subclass
class Triangle(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(3):
            artist.forward(self.size)
            artist.left(120)
        artist.end_fill()

# Star subclass
class Star(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(5):
            artist.forward(self.size)
            artist.right(144)
        artist.end_fill()

# Function to generate and save the shape
# Function to generate and save the shape
def generate_art(shape_type, color, size):
    # Set up the Turtle graphics window
    screen = turtle.Screen()
    screen.bgcolor("white")
    artist = turtle.Turtle()
    artist.speed(0)

    # Create shape objects based on the form data
    if shape_type == 'circle':
        shape = Circle(color, random.randint(-200, 200), random.randint(-200, 200), size)
    elif shape_type == 'square':
        shape = Square(color, random.randint(-200, 200), random.randint(-200, 200), size)
    elif shape_type == 'rectangle':
        shape = Rectangle(color, random.randint(-200, 200), random.randint(-200, 200), size, random.randint(30, 100))
    elif shape_type == 'triangle':
        shape = Triangle(color, random.randint(-200, 200), random.randint(-200, 200), size)
    elif shape_type == 'star':
        shape = Star(color, random.randint(-200, 200), random.randint(-200, 200), size)

    # Draw the shape
    shape.draw(artist)

    # Save the image with Pillow
    canvas = screen.getcanvas()
    canvas.postscript(file="temp_image.ps", colormode='color')  # Save as PostScript
    image = Image.open("temp_image.ps")
    image.save(f"{UPLOAD_FOLDER}/{shape_type}_{color}_{size}.png")

    turtle.bye()  # Close the Turtle window

    return f"{shape_type}_{color}_{size}.png"  # Return filename for use in template



# Home route
@app.route('/')
def home():
    return render_template('image_tool.html')


# Image upload and manipulation route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the image
        image = cv2.imread(filepath)
        if image is None:
            return redirect(request.url)

        # Apply the selected filter
        filter_type = request.form.get('filter')
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
        return render_template('image_tool.html', filename=filename)
    
    return redirect(request.url)

# Filters

def pixelate_filter(image):
    height, width = image.shape[:2]
    small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated

def rainbow_filter(image):
    rainbow = np.zeros_like(image)
    height, width = image.shape[:2]
    for i in range(height):
        rainbow[i, :] = [int(255 * i / height), int(255 * (1 - i / height)), 255]
    blended = cv2.addWeighted(image, 0.7, rainbow, 0.3, 0)
    return blended

def mirror_filter(image):
    height, width = image.shape[:2]
    image[:, :width // 2] = cv2.flip(image[:, width // 2:], 1)
    return image

def cartoonify_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


# Shape generation route
@app.route('/generate_shape', methods=['POST'])
def generate_shape_route():
    shape_type = request.form.get('shape_type', 'circle')  # Default to circle
    color = request.form.get('color', 'red')  # Default to red
    size = int(request.form.get('size', 100))  # Default size to 100

    filename = generate_art(shape_type, color, size)

    return jsonify({
        "shape_type": shape_type,
        "color": color,
        "size": size,
        "file_path": f"images/uploads/{filename}"
    })


# Plot part
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

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import sounddevice as sd
import wave

# Configuration pour les fichiers audio
UPLOAD_FOLDER = 'static/audio/uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fonction pour vérifier les extensions des fichiers audio
def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route pour la manipulation audio
@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Vérifier si un fichier audio est envoyé
    if 'audio_file' not in request.files:
        return redirect(request.url)

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return redirect(request.url)

    if audio_file and allowed_audio_file(audio_file.filename):
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # Obtenir l'effet sélectionné
        effect = request.form.get('effect')
        
        # Processus basé sur l'effet sélectionné
        if effect == 'speed_up':
            processed_audio = speed_up_audio(audio_path, factor=1.5)
        elif effect == 'reverb':
            processed_audio = add_reverb(audio_path, factor=1.2)
        elif effect == 'overlay':
            # Si un autre fichier audio pour overlay est sélectionné
            overlay_file = request.files['overlay_file']
            overlay_filename = secure_filename(overlay_file.filename)
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            overlay_file.save(overlay_path)
            processed_audio = overlay_sounds(audio_path, overlay_path)
        
        # Sauvegarder l'audio traité
        output_filename = f"processed_{effect}_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        save_audio(processed_audio, output_path)

        return render_template('audio_tool.html', audio_path=output_path)

    return redirect(request.url)

# Fonction pour accélérer l'audio
def speed_up_audio(audio_file, factor=1.5):
    data, fs = read_audio(audio_file)
    new_data = data[::int(factor)]  # Accélérer en réduisant la taille des échantillons
    return new_data, fs

# Fonction pour ajouter de la réverbération (simple effet avec ajout de retard)
def add_reverb(audio_file, factor=1.2):
    data, fs = read_audio(audio_file)
    reverb = np.concatenate([data, np.zeros(int(fs))])  # Ajouter un délai pour la réverbération
    reverb[:len(data)] = reverb[:len(data)] * factor  # Appliquer l'effet de volume
    return reverb, fs

# Fonction pour superposer deux pistes audio
def overlay_sounds(audio_file_1, audio_file_2):
    data_1, fs_1 = read_audio(audio_file_1)
    data_2, fs_2 = read_audio(audio_file_2)
    
    if fs_1 != fs_2:
        raise ValueError("Les fichiers audio doivent avoir le même taux d'échantillonnage.")
    
    # Superposer les deux pistes
    min_len = min(len(data_1), len(data_2))
    combined = data_1[:min_len] + data_2[:min_len]  # Additionner les échantillons
    return combined, fs_1

# Fonction pour lire un fichier audio
def read_audio(file_path):
    with wave.open(file_path, 'rb') as wf:
        fs = wf.getframerate()  # Fréquence d'échantillonnage
        n_samples = wf.getnframes()
        data = np.frombuffer(wf.readframes(n_samples), dtype=np.int16)
    return data, fs

# Fonction pour sauvegarder un fichier audio
def save_audio(data, output_path):
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 octets par échantillon (16 bits)
        wf.setframerate(44100)  # Fréquence d'échantillonnage de 44.1 kHz
        wf.writeframes(data.tobytes())

# Fonction pour lire un fichier audio avec sounddevice (facultatif, juste pour tester)
def play_audio(file_path):
    data, fs = read_audio(file_path)
    sd.play(data, fs)
    sd.wait()





if __name__ == '__main__':
    app.run(debug=True, port=5001)
