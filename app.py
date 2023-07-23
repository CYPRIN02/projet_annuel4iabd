"""
This is import
"""
import os
import csv
import time
import cv2
import keyboard
import numpy as np
import pandas as pd
from keras.models import model_from_json
import matplotlib
from flask import Flask, render_template, request
from matplotlib import pyplot as plt


matplotlib.use("Agg")
app = Flask(__name__)

@app.route("/")
def home():
    """
    Page Home
    """
    return render_template("home.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Function that analyze the video
    """
    video = request.files["video"]
    video_path = os.path.join("uploads", video.filename)
    if video.save(video_path):
        return render_template("home.html", error="Veuillez télécharger une vidéo.")

    # Dictionnary of the 7 emotions
    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }

    # Load json and create model
    json_file = open("model/emotion_model.json", "r", encoding="utf-8")
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # Load weights into new model
    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded model from disk")

    # To use webcam to test the model
    # cap = cv2.VideoCapture(0)

    # To use a video to test the model
    cap = cv2.VideoCapture(video_path)

    # Set the desired FPS
    fps = 10.0
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Create a csv file-------------------------
    # Create a dictionary to take count of each emotion in CSV file then
    emotion_count = {
        "Angry": 0,
        "Disgusted": 0,
        "Fearful": 0,
        "Happy": 0,
        "Neutral": 0,
        "Sad": 0,
        "Surprised": 0,
    }

    # Create an empty table
    emotion_df = []

    # Set the start time
    start_time = time.time()

    while True:
        # Find haarcascade to draw bounding box around qface
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier(
            "haarcascades/haarcascade_frontalface_default.xml"
        )
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all the faces on video
        num_faces = face_detector.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5
        )

        # Take each face available on the camera and Preprocess it
        for x_coord, y_coord, width, height in num_faces:
            # Draw green rectangle
            cv2.rectangle(frame, (x_coord, y_coord - 50), (x_coord + width, y_coord + height + 10), (0, 255, 0), 4)
            rect_gray_frame = gray_frame[y_coord : y_coord + height, x_coord : x_coord + width]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(rect_gray_frame, (48, 48)), -1), 0
            )

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            print(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            # Write the emotion above the rectangle
            cv2.putText(
                frame,
                emotion_dict[maxindex],
                (x_coord + 5, y_coord - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Increment the count of the detected emotion
            emotion_count[emotion_dict[maxindex]] += 1

            # Append the data to the table for evolution emotion
            emotion_df.append(emotion_prediction[0])

        # Show the frames of the video and Stop  key 'q'
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Check if the 'q' key has been pressed
        if keyboard.is_pressed("q"):
            break

        # Check the elapsed time and stop the loop after 2 minutes
        elapsed_time = time.time() - start_time
        if elapsed_time >= 40:  # 2min = 120s
            break

    # Save the emotion_df as csv file
    emotion_df = pd.DataFrame(emotion_df, columns=emotion_dict.values())
    emotion_df.to_csv("csv_file/emotion_predictions.csv", index=False)

    # Save the emotion count to a CSV file
    with open("csv_file/emotion_count.csv", mode="w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(list(emotion_count.keys()))
        writer.writerow(list(emotion_count.values()))

    # Free the cap
    cap.release()
    # Close the window of the video
    cv2.destroyAllWindows()

    # delete the video
    os.remove(video_path)

    # Read the CSV file into a pandas DataFrame ------------------------------------ plot pie
    df_count = pd.read_csv("csv_file/emotion_count.csv")
    # Convert the dataframe to series
    totals = df_count.iloc[0]  # Assuming your data is on the first row of the df
    # Fill NaN values with 0
    totals = totals.fillna(0)
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 5))
    wedges, texts, autotexts = ax.pie(totals, labels=totals.index, autopct="%1.1f%%")
    plt.title("Emotion Percentages")
    ax.legend(wedges, totals.index,
            title="Emotions",
            loc="lower left",
            bbox_to_anchor=(0, 0, 0.5, 0.5)) # Adjust these values to move your legend
    # Check if the file already exists and delete it
    pie_chart_path = "static/plt_result/emotion_percentages.png"
    if os.path.exists(pie_chart_path):
        os.remove(pie_chart_path)
    # Save the pie chart as a PNG image
    plt.savefig(pie_chart_path)
    plt.close()

    # Read the CSV file into a pandas DataFrame ---- plot of the evolution
    # Read the CSV file into a pandas DataFrame
    df_prediction = pd.read_csv("csv_file/emotion_predictions.csv")

    # Define a list of colors for the plot lines
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Counter for the plot filenames
    plot_counter = 1

    # Create a separate line plot for each emotion
    for i, emotion in enumerate(df_prediction.columns):
        # Create a new figure for this plot
        plt.figure(figsize=(10, 5))
        # Plot the emotion data with color
        plt.plot(df_prediction[emotion], label=emotion, color=colors[i % 7])
        # Set the title including the emotion name
        plt.title(f"Emotion Evolution Over Time - {emotion}")
        plt.xlabel("Number Frame")  # Set the x-axis label
        plt.ylabel("Emotion percen")  # Set the y-axis label
        plt.legend(loc="best")  # Add a legend

        # Define the path for the plot image file
        emotion_line_plot_path = f"static/plt_result/emotion_evolution_{plot_counter}.png"

        # Check if the file already exists and delete it
        if os.path.exists(emotion_line_plot_path):
            os.remove(emotion_line_plot_path)

        # Save the plot as a PNG image
        plt.savefig(emotion_line_plot_path)

        plt.close()  # Close the figure to free up memory

        plot_counter += 1  # Increment the counter for the next plot

    # ---------------------------------------------------------------------
    results_pie_emotion = "/static/plt_result/emotion_percentages.png"
    results_evolution_emotion_1 = "/static/plt_result/emotion_evolution_1.png"
    results_evolution_emotion_2 = "/static/plt_result/emotion_evolution_2.png"
    results_evolution_emotion_3 = "/static/plt_result/emotion_evolution_3.png"
    results_evolution_emotion_4 = "/static/plt_result/emotion_evolution_4.png"
    results_evolution_emotion_5 = "/static/plt_result/emotion_evolution_5.png"
    results_evolution_emotion_6 = "/static/plt_result/emotion_evolution_6.png"
    results_evolution_emotion_7 = "/static/plt_result/emotion_evolution_7.png"

    return render_template(
        "results.html",
        results_pie_emotion=results_pie_emotion,
        results_evolution_emotion_1=results_evolution_emotion_1,
        results_evolution_emotion_2=results_evolution_emotion_2,
        results_evolution_emotion_3=results_evolution_emotion_3,
        results_evolution_emotion_4=results_evolution_emotion_4,
        results_evolution_emotion_5=results_evolution_emotion_5,
        results_evolution_emotion_6=results_evolution_emotion_6,
        results_evolution_emotion_7=results_evolution_emotion_7
    )


@app.route("/about")
def about():
    """
    Page About
    """
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    """
    Page contact
    """
    if request.method == "POST":
        # Récupérez les données envoyées par le formulaire de contact
        name = request.form["name"]
        email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]

        # Créez un dictionnaire avec les données de l'utilisateur
        user_data = {
            "Nom": name,
            "Email": email,
            "Sujet": subject,
            "Message": message
        }

        # Vérifiez si le dossier "user_data" existe, sinon, créez-le
        if not os.path.exists("user_data"):
            os.mkdir("user_data")

        # Définissez le chemin du fichier CSV pour enregistrer les données de l'utilisateur
        csv_file_path = os.path.join("user_data", "user_data.csv")

        # Vérifiez si le fichier CSV existe, sinon, créez-le et ajoutez l'en-tête
        csv_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["Nom", "Email", "Sujet", "Message"])
            if not csv_exists:
                writer.writeheader()          
            # Enregistrez les données de l'utilisateur dans le fichier CSV
            writer.writerow(user_data)

        # Redirigez l'utilisateur vers une page de confirmation ou une autre page de votre choix
        return render_template("thank_you.html")
    else:
        return render_template("contact.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
