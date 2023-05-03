"""from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')# @app.route('/admin')
def home():
    return render_template('home.html')


def admin():
    return "Bonjour Admin!"


if __name__ == "__main__":
    app.run()
"""
from flask import Flask, render_template, request
import os
import cv2
import boto3

app = Flask(__name__)

# Configuration AWS
s3_client = boto3.client('s3', aws_access_key_id='your_access_key_id', aws_secret_access_key='your_secret_access_key')
rekognition_client = boto3.client('rekognition', region_name='us-east-1', aws_access_key_id='your_access_key_id',
                                  aws_secret_access_key='your_secret_access_key')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Vérifie si une vidéo a été soumise
    if 'video' not in request.files:
        return render_template('home.html', error='Veuillez télécharger une vidéo.')

    # Récupère le fichier vidéo soumis
    video = request.files['video']
    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)

    # Analyse de la vidéo avec OpenCV et Amazon Rekognition
    emotion_counts = {'HAPPY': 0, 'SAD': 0, 'ANGRY': 0, 'CONFUSED': 0, 'DISGUSTED': 0, 'SURPRISED': 0, 'CALM': 0}
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            _, img_encoded = cv2.imencode('.jpg', crop_img)
            response = rekognition_client.detect_faces(Image={'Bytes': img_encoded.tobytes()}, Attributes=['ALL'])
            if len(response['FaceDetails']) > 0:
                emotions = response['FaceDetails'][0]['Emotions']
                for emotion in emotions:
                    emotion_counts[emotion['Type']] += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Suppression du fichier vidéo téléchargé
    os.remove(video_path)

    return render_template('results.html', emotion_counts=emotion_counts)

# Page à propos
@app.route('/about')
def about():
    return render_template('about.html')

# Page contact
@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
