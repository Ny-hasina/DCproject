import turtle
import random
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Dossier pour les images générées
UPLOAD_FOLDER = 'static/images/shape_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            artist.forward(self.size)  # width
            artist.left(90)
            artist.forward(self.height)  # height
            artist.left(90)
        artist.end_fill()

class Triangle(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(3):
            artist.forward(self.size)
            artist.left(120)
        artist.end_fill()

class Star(Shape):
    def draw(self, artist):
        super().draw(artist)
        artist.begin_fill()
        for _ in range(5):
            artist.forward(self.size)
            artist.right(144)
        artist.end_fill()
# generate_art_turtle.py

import sys
import turtle
from PIL import Image

def generate_shape(shape_type, color, size):
    # Configuration de Turtle
    screen = turtle.Screen()
    screen.bgcolor("white")
    screen.setup(width=600, height=400)
    artist = turtle.Turtle()
    artist.speed(3)
    artist.color(color)

    artist.penup()
    artist.goto(0, 0)
    artist.pendown()

    # Dessiner selon le type de forme
    if shape_type == 'circle':
        artist.begin_fill()
        artist.circle(size)
        artist.end_fill()
    elif shape_type == 'square':
        artist.begin_fill()
        for _ in range(4):
            artist.forward(size)
            artist.left(90)
        artist.end_fill()
    elif shape_type == 'rectangle':
        artist.begin_fill()
        for _ in range(2):
            artist.forward(size)
            artist.left(90)
            artist.forward(size / 2)
            artist.left(90)
        artist.end_fill()
    elif shape_type == 'triangle':
        artist.begin_fill()
        for _ in range(3):
            artist.forward(size)
            artist.left(120)
        artist.end_fill()
    elif shape_type == 'star':
        artist.begin_fill()
        for _ in range(5):
            artist.forward(size)
            artist.right(144)
        artist.end_fill()

    # Sauvegarder l'image avec Turtle et Pillow
    canvas = screen.getcanvas()
    canvas.postscript(file="temp_image.ps", colormode='color')
    image = Image.open("temp_image.ps")
    image_path = "generated_shape.png"
    image.save(image_path)

    turtle.bye()  # Fermer Turtle

    return image_path

# Récupérer les arguments
if __name__ == "__main__":
    shape_type = sys.argv[1]
    color = sys.argv[2]
    size = int(sys.argv[3])

    # Générer l'image et obtenir le chemin du fichier
    result_file = generate_shape(shape_type, color, size)
    print(result_file)  # Afficher le chemin pour Flask

