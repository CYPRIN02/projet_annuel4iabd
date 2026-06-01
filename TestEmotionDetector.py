# Import packages
import cv2
import numpy as np
from keras.models import model_from_json
import csv
import time
import keyboard

# Dictionnary of the 7 emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# To use webcam to test the model(optional)
#cap = cv2.VideoCapture(0)

# To use a video to test the model
cap = cv2.VideoCapture("video\\Funny.mp4")

# Set the desired FPS
fps = 10.0
cap.set(cv2.CAP_PROP_FPS, fps)

# Create a csv file---------------------------------------------------------------------------------------
# Create a dictionary to take count of each emotion in CSV file then
emotion_count = {"Angry": 0, "Disgusted": 0, "Fearful": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprised": 0}

# Set the start time
start_time = time.time()

while True:
    # Find haar cascade to draw bounding box around qface
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect all the faces on video
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        # Draw green rectangle
        #cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        rect_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(rect_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        # Write the emotion above the rectangle
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Increment the count of the detected emotion
        emotion_count[emotion_dict[maxindex]] += 1

    # Show the frames of the video and Stop  key 'q'
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    # Check if the 'q' key has been pressed
    if keyboard.is_pressed('q'):
        break

    # Check the elapsed time and stop the loop after 2 minutes
    elapsed_time = time.time() - start_time
    if elapsed_time >= 120: # 2min = 120s
        break

# Save the emotion count to a CSV file
with open('emotion_count.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(list(emotion_count.keys()))
    writer.writerow(list(emotion_count.values()))

# Free the cap
cap.release()
# Close the window of the video
cv2.destroyAllWindows()
