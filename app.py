from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
import pyaudio
from pydub import AudioSegment, effects
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename
import cv2
import os
import io
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
import random
import subprocess
import turtle
import sys
import pygame
import base64
import math
import re

# Création de l'application Flask
app = Flask(__name__)

# Dossiers pour sauvegarder les images
UPLOAD_FOLDER = 'static/uploads_shape'
OUTPUT_FOLDER = 'static/output_shape'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Classe de base Forme
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

# Classes pour les formes (Circle, Square, Rectangle, Triangle, Star)
class Circle(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        artist.circle(self.size)
        artist.end_fill()


class Square(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(4):
            artist.forward(self.size)
            artist.left(90)
        artist.end_fill()


class Rectangle(Shape):
    def __init__(self, color, x, y, width, height):
        super().__init__(color, x, y, width)
        self.height = height

    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(2):
            artist.forward(self.size)  # largeur
            artist.left(90)
            artist.forward(self.height)  # hauteur
            artist.left(90)
        artist.end_fill()


class Triangle(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(3):
            artist.forward(self.size)
            artist.left(120)  # angle pour un triangle équilatéral
        artist.end_fill()


class Star(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(5):
            artist.forward(self.size)
            artist.right(144)  # angle pour dessiner une étoile à 5 branches
        artist.end_fill()

# Route pour dessiner une forme et afficher l'image générée
@app.route('/generate_static_shapes')
def generate_static_shapes():
    return render_template('shape_generator.html')
@app.route('/draw_shape', methods=['POST'])
def draw_shape():
    shape_type = request.form['shape_type']
    color = request.form['color']
    x = int(request.form['x'])
    y = int(request.form['y'])
    size = int(request.form['size'])

    # Créer un artiste avec turtle
    screen = turtle.Screen()
    screen.setup(width=600, height=600)  # Taille de la fenêtre de dessin
    artist = turtle.Turtle()
    artist.speed(5)

    # Créer la forme basée sur l'entrée
    if shape_type == 'circle':
        shape = Circle(color, x, y, size)
    elif shape_type == 'square':
        shape = Square(color, x, y, size)
    elif shape_type == 'rectangle':
        shape = Rectangle(color, x, y, size, size)  # Utilisation de size pour largeur et hauteur
    elif shape_type == 'triangle':
        shape = Triangle(color, x, y, size)
    elif shape_type == 'star':
        shape = Star(color, x, y, size)
    else:
        return "Invalid shape type", 400

    # Dessiner la forme
    shape.draw(artist)

    # Sauvegarder le dessin en tant qu'image PNG
    filename = 'drawn_shape.png'
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    # Convertir le dessin Turtle en image avec Pillow (PIL)
    canvas = screen.getcanvas()
    canvas.postscript(file=filepath.replace('.png', '.eps'))  # Sauvegarder en format EPS
    img = Image.open(filepath.replace('.png', '.eps'))
    img.save(filepath, 'PNG')

    # Fermer la fenêtre Turtle après le dessin
    screen.bye()

    return render_template('shape_generator.html', filename=filename)

# Route pour afficher l'image générée
@app.route('/output_shape/<filename>')
def send_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('indx.html')


# Configuration pour les fichiers image
UPLOAD_FOLDER_IMAGES = 'static/images/uploads'
ALLOWED_EXTENSIONS_IMAGES = {'png', 'jpg', 'jpeg', 'gif'}  # Extensions autorisées pour les images
app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES

# S'assurer que le dossier de téléchargement des images existe
if not os.path.exists(UPLOAD_FOLDER_IMAGES):
    os.makedirs(UPLOAD_FOLDER_IMAGES)

# Fonction pour vérifier les extensions de fichier autorisées pour les images
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGES

# Image upload and manipulation route
@app.route('/Image_manip')
def Image_manip():
    return render_template('image_tool.html')

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
    
    if file and allowed_image_file(file.filename):
        print("File is valid")  # Debugging
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], filename)
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

        # Save the processed image with a different name
        processed_filename = 'processed_' + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], processed_filename)
        cv2.imwrite(processed_filepath, processed_image)
        print(f"Processed image saved to: {processed_filepath}")  # Debugging

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


app.config['UPLOAD_FOLDER_AUDIO'] = 'static/uploads_audio'  # Nouveau nom
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER_AUDIO']):
    os.makedirs(app.config['UPLOAD_FOLDER_AUDIO'])
    print(f"Created uploads folder at: {app.config['UPLOAD_FOLDER_AUDIO']}")

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def normalize_filename(filename):
    """Normalize the filename by removing special characters and spaces."""
    # Remplace les espaces et les caractères spéciaux par des underscores
    normalized = re.sub(r'[^\w\.-]', '_', filename)
    return normalized

def apply_effect(audio, effect):
    """Apply the selected effect to the audio."""
    if effect == 'speed_up':
        return audio.speedup(playback_speed=1.5)
    elif effect == 'slow_down':
        return audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 0.75)
        }).set_frame_rate(audio.frame_rate)
    elif effect == 'fade_in':
        return audio.fade_in(2000)  # 2-second fade-in
    elif effect == 'fade_out':
        return audio.fade_out(2000)  # 2-second fade-out
    else:
        return audio  # No effect applied

@app.route('/audio', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            print("No file part in the request.")
            return redirect(request.url)

        file = request.files['file']
        effect = request.form.get('effect')  # Get the selected effect
        print(f"Selected effect: {effect}")

        if file.filename == '':
            print("No file selected.")
            return redirect(request.url)

        if file:  # Désactivez allowed_file pour le moment
            # Normalize the filename
            normalized_filename = normalize_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], normalized_filename)
            file.save(file_path)
            print(f"File saved at: {file_path}")  # Log pour vérifier le chemin du fichier

            # Vérifiez si le fichier existe après sauvegarde
            if os.path.exists(file_path):
                print("File successfully saved.")
            else:
                print("File was not saved.")

            # Load the audio file using PyDub
            audio = AudioSegment.from_file(file_path)
            print("Audio file loaded successfully.")

            # Apply the selected effect
            processed_audio = apply_effect(audio, effect)
            print("Effect applied successfully.")

            # Save the processed audio
            processed_file_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], 'processed_' + normalized_filename)
            processed_audio.export(processed_file_path, format="mp3")
            print(f"Processed file saved at: {processed_file_path}")

            # Vérifiez si le fichier traité existe
            if os.path.exists(processed_file_path):
                print("Processed file successfully saved.")
            else:
                print("Processed file was not saved.")

            # Render the template with both audio files
            return render_template('audio.html', 
                                 original_audio=normalized_filename, 
                                 processed_audio='processed_' + normalized_filename)

    return render_template('audio.html')

@app.route('/uploads_au/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], filename)
    if os.path.exists(file_path):
        print(f"Serving file: {file_path}")  # Log pour vérifier que le fichier est servi
        return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER_AUDIO']), filename)
    else:
        print(f"File not found: {file_path}")  # Log si le fichier n'existe pas
        return "File not found", 404







# Load pre-trained VGG19 model
vgg = models.vgg19(pretrained=True).features.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

# Image loading and transformation
def image_loader(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')

    if max_size:
        size = max(image.size)
        scale = max_size / float(size)
        new_size = tuple([int(x * scale) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])

    image = preprocessor(image)
    return image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_features(image, model, layers=None):
    layers = layers or {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def content_loss(target, content):
    return torch.nn.functional.mse_loss(target, content)

def gram_matrix(input):
    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    G = torch.mm(features, features.t())
    return G.div(batch_size * channels * height * width)

def style_loss(target, style):
    G_target = gram_matrix(target)
    G_style = gram_matrix(style)
    return torch.nn.functional.mse_loss(G_target, G_style)

# Neural Style Transfer function
def neural_style_transfer(content_image_path, style_image_path, output_image_path, num_iterations=500):
    content_image = image_loader(content_image_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    style_image = image_loader(style_image_path, shape=content_image.shape[-2:]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    content_features = get_features(content_image, vgg)
    style_features = get_features(style_image, vgg)

    target = content_image.clone().requires_grad_(True)

    optimizer = optim.LBFGS([target])

    for i in range(num_iterations):
        def closure():
            target.data.clamp_(0, 1)
            optimizer.zero_grad()

            target_features = get_features(target, vgg)

            c_loss = content_loss(target_features['conv4_2'], content_features['conv4_2'])
            s_loss = 0
            for layer in style_features:
                s_loss += style_loss(target_features[layer], style_features[layer])

            total_loss = c_loss + 1e6 * s_loss
            total_loss.backward(retain_graph=True)  # Conserver le graphe pour la rétropropagation suivante
            return total_loss
        
        optimizer.step(closure)

    target.data.clamp_(0, 1)
    final_img = target.cpu().clone()
    final_img_pil = transforms.ToPILImage()(final_img.squeeze(0))
    final_img_pil.save(output_image_path)


# Ajoute un message de progression dans ton template HTML
@app.route('/style_trans', methods=['GET', 'POST'])
def style_trans():
    if request.method == 'POST':
        # Commencer le traitement en arrière-plan
        content_file = request.files['file']
        style_file = request.files['style_file']

        if content_file and allowed_file(content_file.filename) and style_file and allowed_file(style_file.filename):
            content_filename = os.path.join(app.config['UPLOAD_FOLDER'], content_file.filename)
            style_filename = os.path.join(app.config['UPLOAD_FOLDER'], style_file.filename)

            content_file.save(content_filename)
            style_file.save(style_filename)

            output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'styled_image.jpg')

            # Appeler la fonction de traitement
            neural_style_transfer(content_filename, style_filename, output_filename)
            
            # Retourner la page avec l'image finale traitée
            return render_template('style_trans.html', filename='output_st/styled_image.jpg', message="Image traitée avec succès!")

    return render_template('style_trans.html')


@app.route('/upload_st', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'style_file' not in request.files:
        return redirect(request.url)
    
    content_file = request.files['file']
    style_file = request.files['style_file']

    if content_file and allowed_file(content_file.filename) and style_file and allowed_file(style_file.filename):
        content_filename = os.path.join(app.config['UPLOAD_FOLDER'], content_file.filename)
        style_filename = os.path.join(app.config['UPLOAD_FOLDER'], style_file.filename)

        content_file.save(content_filename)
        style_file.save(style_filename)

        output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'styled_image.jpg')

        # Apply Neural Style Transfer
        neural_style_transfer(content_filename, style_filename, output_filename)

        return render_template('style_trans.html', filename='output_st/styled_image.jpg')
    
    return redirect(request.url)

@app.route('/uploads_st/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output_st/<filename>')
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
  



# Dossier pour enregistrer l'image générée
OUTPUT_FOLDER = 'static/drawings'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Pygame initialization
pygame.init()

WIDTH, HEIGHT = 800, 600
#OUTPUT_FOLDER = "static"  # Dossier de sortie pour les images générées
# Fonction pour calculer les points d'un triangle équilatéral
def calculer_triangle_equilateral(x, y, taille):
    # Calcul des coordonnées des 3 points du triangle équilatéral
    hauteur = math.sqrt(3) / 2 * taille  # hauteur du triangle équilatéral
    point1 = (x, y)  # Point du bas gauche
    point2 = (x + taille, y)  # Point du bas droit
    point3 = (x + taille / 2, y - hauteur)  # Point du sommet
    return [point1, point2, point3]

# Fonction pour dessiner sur le canvas avec Pygame
def draw_canvas(shapes, color, file_name="drawing.png"):
    # Créer la surface de dessin avec pygame
    screen = pygame.Surface((800, 600))  # Dimensions du canvas
    screen.fill((255, 255, 255))  # Fond blanc

    for shape in shapes:
        if shape['type'] == 'circle':
            pygame.draw.circle(screen, color, (shape['x'], shape['y']), shape['size'])
        elif shape['type'] == 'square':
            pygame.draw.rect(screen, color, pygame.Rect(shape['x'] - shape['size'] / 2, shape['y'] - shape['size'] / 2, shape['size'], shape['size']))
        elif shape['type'] == 'rectangle':
            pygame.draw.rect(screen, color, pygame.Rect(shape['x'] - shape['size'] / 2, shape['y'] - shape['size'] / 4, shape['size'], shape['size'] / 2))
        elif shape['type'] == 'triangle':
            points = calculer_triangle_equilateral(shape['x'], shape['y'], shape['size'])
            pygame.draw.polygon(screen, color, points)

    # Enregistrer l'image dans le dossier OUTPUT_FOLDER
    image_path = os.path.join(OUTPUT_FOLDER, file_name)
    pygame.image.save(screen, image_path)
    return image_path


# Route principale des formes dynamiques
@app.route('/dynamic_shape')
def dynamic_shape():
    return render_template('shape_dynamic.html')

# Route pour recevoir les données du frontend et générer l'image
@app.route('/draw_dyna', methods=['POST'])
def draw_dyna():
    data = request.get_json()
    shapes = data['shapes']
    color = data['color']

    # Créer un nom unique pour l'image (ex: drawing_1.png, drawing_2.png, etc.)
    file_name = f"drawing_{len(os.listdir(OUTPUT_FOLDER)) + 1}.png"

    # Dessiner l'image et la sauvegarder dans le dossier spécifié
    image_path = draw_canvas(shapes, color, file_name)

    # Renvoi de l'URL de l'image générée
    return jsonify({
        'image_url': f'/static/drawings/{file_name}',
        'message': 'Image has been successfully saved!'
    })

UPLOAD_FOLDER = 'static/uploads_st'
OUTPUT_FOLDER = 'static/output_st'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



if __name__ == '__main__':
    # Vérifier si les dossiers existent et les créer si nécessaire
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Lancer l'application Flask sur le port 5001
    app.run(debug=True, port=5001)