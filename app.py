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
import seaborn as sns
from transformers import pipeline,GPT2LMHeadModel, GPT2Tokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration

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



# Chemin vers le dossier des images
IMAGE_FOLDER = 'static/images/uploads'
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Charger les données
def load_data():
    try:
        data = pd.read_csv('data/temperatures.csv')
        # Convertir la colonne 'date' en datetime
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Gestion des erreurs de format
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur

# Créer les visualisations
def create_visualizations(data):
    try:
        sns.set(style="darkgrid")
        
        # Plot 1: Line Plot with Artistic Styling
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=data['date'], y=data['temperature'], marker='o', linestyle='-', color='purple')
        plt.title('Temperature Trends Over Time', fontsize=14, fontweight='bold', color='darkblue')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.xticks(rotation=45)
        plot1_path = os.path.join(IMAGE_FOLDER, 'plot1.png')
        plt.savefig(plot1_path)
        plt.close()
        
        # Plot 2: Heatmap
        plt.figure(figsize=(8, 6))
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        pivot_table = data.pivot(index='month', columns='day', values='temperature')
        sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.1f')
        plt.title('Temperature Heatmap')
        plot2_path = os.path.join(IMAGE_FOLDER, 'plot2.png')
        plt.savefig(plot2_path)
        plt.close()
        
        # Plot 3: Artistic Swarmplot
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x=data['date'].dt.strftime('%Y-%m-%d'), y=data['temperature'], palette='inferno')
        plt.title('Temperature Variations', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.xticks(rotation=45)
        plot3_path = os.path.join(IMAGE_FOLDER, 'plot3.png')
        plt.savefig(plot3_path)
        plt.close()
        
        return plot1_path, plot2_path, plot3_path
    except Exception as e:
        print(f"Erreur lors de la création des visualisations : {e}")
        return None, None, None

# Route pour afficher les visualisations
@app.route('/visualization')
def visualization():
    data = load_data()
    if data.empty:
        return render_template('error.html', message="Erreur lors du chargement des données.")
    
    plot1_path, plot2_path, plot3_path = create_visualizations(data)
    if not plot1_path or not plot2_path or not plot3_path:
        return render_template('error.html', message="Erreur lors de la création des graphiques.")
    
    return render_template('visualization.html', 
                           plot1=plot1_path, 
                           plot2=plot2_path, 
                           plot3=plot3_path)

# Route pour afficher le dataset
@app.route('/view_dataset')
def view_dataset():
    data = load_data()
    if data.empty:
        return render_template('error.html', message="Erreur lors du chargement des données.")
    
    return render_template('dataset.html', tables=[data.to_html(classes='data')], titles=['Dataset'])


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




# Dossier pour enregistrer l'image générée
OUTPUT_FOLDER_IM = 'static/images/uploads'
if not os.path.exists(OUTPUT_FOLDER_IM):
    os.makedirs(OUTPUT_FOLDER_IM)

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
    image_path = os.path.join(OUTPUT_FOLDER_IM, file_name)
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
    file_name = f"drawing_{len(os.listdir(OUTPUT_FOLDER_IM)) + 1}.png"

    # Dessiner l'image et la sauvegarder dans le dossier spécifié
    image_path = draw_canvas(shapes, color, file_name)

    # Renvoi de l'URL de l'image générée
    return jsonify({
        'image_url': f'/static/drawings/{file_name}',
        'message': 'Image has been successfully saved!'
    })


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    image_folder = os.path.join(app.static_folder, 'images/uploads')

    # Vérifier si le dossier existe
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Liste des images disponibles
    images = [f'images/uploads/{img}' for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

    if request.method == 'POST':
        # Ajouter une image
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(image_folder, filename)
                file.save(filepath)
                return redirect(url_for('gallery'))  # Recharger la page après l'upload

    return render_template('gallery.html', images=images)

@app.route('/delete_image', methods=['POST'])
def delete_image():
    image_name = request.form.get('image_name')  # Nom de l'image à supprimer
    image_path = os.path.join(app.static_folder, image_name)

    if os.path.exists(image_path):
        os.remove(image_path)
        return jsonify({'message': 'Image supprimée avec succès', 'success': True})
    return jsonify({'message': 'Image introuvable', 'success': False})

#text generation
@app.route('/text_generation')
def text_generation():
    return render_template('text_generation.html')

@app.route('/get_images')
def get_images():
    image_folder = os.path.join(app.static_folder, 'images/uploads')
    images = [url_for('static', filename=f'images/uploads/{img}') for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]
    return jsonify({'images': images})
# Load BLIP model (for generating image descriptions)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load GPT-2 model (for text generation)
model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model.eval()

def generate_image_description(image_path):
    """Uses BLIP to generate a caption for an image."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_gpt2_text(prompt):
    """Uses GPT-2 to generate text based on a given prompt, avoiding repetitions."""
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = gpt2_model.generate(
        inputs, 
        max_length=150, 
        num_return_sequences=1, 
        no_repeat_ngram_size=3,  # Prevents repetitive n-grams
        top_p=0.9,               # Nucleus sampling to avoid generic outputs
        temperature=0.7,         # Controls randomness (lower = more focused)
        repetition_penalty=1.2,  # Penalizes token repetition
        do_sample=True           # Ensures sampling instead of greedy decoding
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    image_name = data['image_name']
    text_type = data['text_type']

    # Construct the full path to the image
    image_path = os.path.join("static/images/uploads", image_name.split("/")[-1])

    # Generate an image description using BLIP
    image_description = generate_image_description(image_path)

    # Construct a prompt based on text type
    if text_type == 'poetry':
        prompt = f" {image_description}"
    elif text_type == 'caption':
        prompt = f"{image_description}"
    elif text_type == 'description':
        prompt = f"{image_description}"

    # Generate text using GPT-2
    generated_text = generate_gpt2_text(prompt)

    return jsonify({'generated_text': generated_text})

#dataset route
@app.route('/')
def dataset():
    file_path = os.path.join('data', 'temperatures.csv')  # Correctement défini
    df = pd.read_csv(file_path)  # Chargement du CSV
    data = df.to_dict(orient='records')  # Convertir en liste de dictionnaires
    
      
    return render_template('dataset.html', dataset=data)

if __name__ == '__main__':
    # Vérifier si les dossiers existent et les créer si nécessaire
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Lancer l'application Flask sur le port 5001
    app.run(debug=True, port=5001)