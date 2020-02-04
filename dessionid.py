# Imports des packages requis
from imutils.video import VideoStream
from imutils.video import FPS
from struct import unpack
import numpy as np
import argparse
import imutils
import time
import cv2
import serial
import adafruit_thermal_printer
import subprocess
import RPi.GPIO as GPIO
import gizeh as gz
import random

uart = 0
ThermalPrinter = 0
printer = 0
prototxt = 0
model = 0
confidence_voulue = 0
CLASSES = 0
COLORS = 0
net = 0

#Configuration générale
def setup():
    global uart
    global ThermalPrinter
    global printer
    global prototxt
    global model
    global confidence_voulue
    global CLASSES
    global COLORS
    global net
    global vs
    
    # Configuration GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    # Allumage LED
    GPIO.setup(40, GPIO.OUT)
    GPIO.output(40, GPIO.HIGH)
    GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Configuration de l'imprimante
    uart = serial.Serial("/dev/serial0", baudrate=9600, timeout=3000)
    ThermalPrinter = adafruit_thermal_printer.get_printer_class(1.11)
    printer = ThermalPrinter(uart)

    # Fichiers d'apprentissage
    prototxt = "/home/pi/Documents/reconnaissance_objets/MobileNetSSD_deploy.prototxt.txt"
    model = "/home/pi/Documents/reconnaissance_objets/MobileNetSSD_deploy.caffemodel"
    confidence_voulue = 0.5

    # Initialisation de la liste des objets entrainés par MobileNet SSD 
    CLASSES = ["arriere-plan", "avion", "velo", "oiseau", "bateau", "bouteille", "autobus", "voiture", "chat", "chaise", "vache", "table", "chien", "cheval", "moto", "personne", "plante en pot", "mouton", "sofa", "train", "moniteur"]

    # Création du contour de détection avec une couleur attribuée au hasard pour chaque objet
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Chargement des fichiers depuis le répertoire de stockage 
    print("Chargement du modèle...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    
    # Initialisation de la caméra, attendre 0.5s pour la mise au point
    print("Démarrage de la PiCamera...")
    vs = VideoStream(usePiCamera=True, resolution=(1024, 768)).start()
    time.sleep(1)

# Lecture d'un dessin
def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }

# Lecture des dessins
def unpack_drawings(path):
    with open(path, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except ValueError as e:
                break

# Récupération du dessin via son nom
def get_drawing(name, index):
    try:
        path = '/home/pi/Documents/cartoonify/cartoonify/downloads/drawing_dataset/'
        itr = unpack_drawings(str(path) + str(name) + '.bin')
        for i in range(index):
            drawing = next(itr)
        return drawing['image']
    except ValueError as e:
        raise e

# Conversion du dessin binaire en dessin python
def convert_quickdraw_strokes_to_gizeh_group(strokes, color=[0, 0, 0], stroke_width=5):
    lines_list = []
    for stroke in strokes:
        x, y = stroke
        points = list(zip(x, y))
        line = gz.polyline(points=points, stroke=color, stroke_width=stroke_width)
        lines_list.append(line)
    return gz.Group(lines_list)

# Spécificité du dessin d'une personne
def draw_person(surface, scale=1.0, position=[0, 0], stroke_width=6):
    body_parts = {'face': [0, 0], 't-shirt': [0, 250], 'pants': [0, 480]}
    gz_body_parts = []
    for name, pos in body_parts.items():
        strokes = get_drawing(name, random.randint(1, 1000))
        strokes_gz = convert_quickdraw_strokes_to_gizeh_group(strokes, stroke_width=stroke_width / scale)
        strokes_gz = strokes_gz.translate(pos)
        gz_body_parts.append(strokes_gz)
    scale *= np.mean([1200, 900]) / 750
    pos[0] = position[0] * 1200 - (scale * (255 / 2))
    pos[1] = position[1] * 900 - (scale * (750 / 2))
    gz_body_parts = gz.Group(gz_body_parts).scale(scale).translate(xy=pos)
    gz_body_parts.draw(surface)
    surface.write_to_png('/home/pi/Documents/reconnaissance_objets/detections/detection.png')
        
#Création du dessin
def draw(surface, strokes, scale=1.0, pos=[0, 0], stroke_width=6, color=[0, 0, 0]):
    try:
        scale *= np.mean([1200, 900]) / 255
        pos[0] = pos[0] * 1200 - (scale * (255 / 2))
        pos[1] = pos[1] * 900 - (scale * (255 / 2))
        lines = convert_quickdraw_strokes_to_gizeh_group(strokes, color, stroke_width=stroke_width / scale)
        lines = lines.scale(scale).translate(xy=pos)
        lines.draw(surface)
        surface.write_to_png('/home/pi/Documents/reconnaissance_objets/detections/detection.png')
    except ValueError as e:
        print(repr(e))

# Traduction de ce qui est detecté en nom de dessin
def get_nom_en(nom_fr):
    switcher = {
        "arriere-plan":"postcard",
        "avion":"airplane",
        "velo":"bicycle",
        "oiseau":"bird",
        "bateau":"speedboat",
        "bouteille":"wine bottle",
        "autobus":"bus",
        "voiture":"car",
        "chat":"cat",
        "chaise":"chair",
        "vache":"cow",
        "table":"table",
        "chien":"dog",
        "cheval":"horse",
        "moto":"motorbike",
        "plante en pot":"house plant",
        "mouton":"sheep",
        "sofa":"couch",
        "train":"train",
        "moniteur":"laptop"
    }
    return switcher.get(nom_fr,"scorpion")

# Prise de photo et détection d'objets
def photo(ev=None):
    global uart
    global ThermalPrinter
    global printer
    global prototxt
    global model
    global confidence_voulue
    global CLASSES
    global COLORS
    global net
    global vs
    
    # Initialisation du compteur FPS
    fps = FPS().start()
    
    # Récupération du flux vidéo, redimension afin d'afficher au maximum 800 pixels 
    print("Lecture du flux...")
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # Récupération des dimensions et transformation en collection d'images
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Détermination de la détection et de la prédiction 
    net.setInput(blob)
    detections = net.forward()
    
    # initialisation de la surface de dessin
    surface = gz.Surface(width=1200, height=900, bg_color=(1, 1, 1))

    # Boucle de détection 
    print("Détection d'objets...")
    for i in np.arange(0, detections.shape[2]):
        # Calcul de la probabilité de l'objet détecté en fonction de la prédiction
        confidence = detections[0, 0, i, 2]
        
        # Suppression des détections faibles (inférieures à la probabilité minimale)
        if confidence > confidence_voulue:
            # Extraction de l'index du type d'objet détecté et calcul des coordonnées de la fenêtre de détection 
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Récupération des coordonnées pour le dessin
            xmin = startX/1024
            xmax = endX/1027
            ymin = startY/768
            ymax = endY/768 
            centre = [np.mean([xmin, xmax]), np.mean([ymin, ymax])]
            size = np.mean([xmax - xmin, ymax - ymin])
            
            # Création du dessin
            print(CLASSES[idx])
            if CLASSES[idx] == 'personne':
                draw_person(surface, scale=size, position=centre)
            else:
                name = get_nom_en(CLASSES[idx])
                draw(surface, get_drawing(name, random.randint(1, 1000)), scale=size, pos=centre)
            
            # Enregistrement de l'image détectée 
            cv2.imwrite("detections/detectionReelle.png", frame)
            frame = 0

    # Impression de l'image
    subprocess.run(["lp","-o","fit-to-page","/home/pi/Documents/reconnaissance_objets/detections/detection.png"])

    # Mise à jour du FPS 
    fps.update()

    # Arrêt du compteur et affichage des informations dans la console
    fps.stop()
    infoFps = "Duree de traitement : {:.2f}".format(fps.elapsed())
    print(infoFps)

# Boucle de detection d'appui sur le bouton
def loop():
    GPIO.add_event_detect(16, GPIO.FALLING, callback=photo, bouncetime=5000)
    while True:
        time.sleep(1)

# Fin de programme
def destroy():
    global vs
    # Arrêt de la PiCamera
    vs.stop()
    GPIO.output(40, GPIO.HIGH)
    GPIO.cleanup()

# Programme principal
if __name__ == '__main__': 
    setup()
    try:
        loop()
    except KeyboardInterrupt: 
        destroy()
