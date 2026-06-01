# projet_annuel4iabd
C'est notre repository pour l'année Master 4IABD

# 🎭 Emotion AI – Video Emotion Detection Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge\&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge\&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green?style=for-the-badge\&logo=opencv)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge)

### 🧠 AI-Powered Video Emotion Recognition Platform

*Emotion detection from videos using Deep Learning and Computer Vision.*

</div>

---

# 📌 Project Overview

Emotion AI is a **web-based emotion recognition platform** developed as a **collective academic project**.

The application allows users to upload a video and automatically:

✅ Detect faces
✅ Predict facial emotions using AI
✅ Generate statistics and visual reports
✅ Visualize emotional evolution over time

The goal is to automate emotional analysis using **Computer Vision** and **Deep Learning**.

---

# 🎯 Problem Statement

Manual emotion analysis in videos is:

* Time consuming
* Subjective
* Difficult to scale

This project addresses these limitations by providing an **automated AI-powered emotion analysis system** capable of processing videos and producing interpretable results.

---

# 💡 Proposed Solution

The platform combines:

* **Flask** → Web application & routing
* **OpenCV** → Face detection
* **TensorFlow / Keras** → Emotion classification
* **Matplotlib** → Statistical visualizations

Pipeline:

Video Upload → Face Detection → Emotion Prediction → Graph Generation → Results Dashboard

---

# 🚀 Features

### 🎥 Video Upload

Users can upload video files directly from the web interface.

### 👤 Face Detection

Automatic face localization using Haar Cascade.

### 😊 Emotion Classification

Deep Learning CNN predicts:

* Angry 😠
* Disgusted 🤢
* Fearful 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprised 😲

### 📊 Emotion Analytics

Automatic generation of:

* Pie charts
* Emotion evolution curves
* Emotion percentage statistics

### 🌐 Interactive Web Interface

Simple and intuitive UI built with Flask + HTML/CSS.

---

# 🏗️ Technical Architecture

```text
User
  ↓
Web Interface (HTML/CSS/JS)
  ↓
Flask Backend (Python)
  ↓
OpenCV Face Detection
  ↓
TensorFlow CNN Emotion Model
  ↓
Statistics + Graphs
  ↓
Results Page
```

---

# 🧰 Tech Stack

| Layer              | Technologies                       |
| ------------------ | ---------------------------------- |
| Front-End          | HTML5, CSS3, JavaScript            |
| Back-End           | Python, Flask, Jinja2              |
| AI / Deep Learning | TensorFlow, Keras                  |
| Computer Vision    | OpenCV, Haar Cascade               |
| Data Processing    | NumPy                              |
| Visualization      | Matplotlib                         |
| Model              | CNN (Convolutional Neural Network) |

---

# 🧠 AI Model

The emotion recognition model is based on a **Convolutional Neural Network (CNN)**.

Architecture:

* Conv2D – 32 filters
* Conv2D – 64 filters
* MaxPooling
* Dropout
* Conv2D – 128 filters
* Flatten
* Dense – 1024 neurons
* Softmax – 7 emotion classes

---

# 📷 Project Screenshots

## 🏠 Home Page

Add screenshot here:

```bash
static/screenshots/home.png
```

---

## 😊 Emotion Detection

```bash
static/screenshots/detection.png
```

---

## 📊 Analytics Dashboard

```bash
static/screenshots/results.png
```

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/your-username/emotion-ai.git
```

Go to project folder:

```bash
cd emotion-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Flask server:

```bash
python app.py
```

Open:

```bash
http://127.0.0.1:5000
```

---

# 📁 Project Structure

```bash
EmotionAI/
│
├── static/
├── templates/
├── model/
├── haarcascades/
├── uploads/
├── app.py
├── requirements.txt
├── README.md
└── emotion_model.h5
```

---

# 📈 Results & Impact

The project successfully demonstrates:

✅ Automated face detection
✅ Multi-class emotion recognition
✅ Real-time video processing
✅ Automatic report generation
✅ End-to-end AI + Web integration

Impact:

* 7 emotions detected
* Video analysis fully automated
* Significant reduction in manual analysis effort

---

# 👥 Collective Work

Academic collective project developed in **Artificial Intelligence & Big Data**.

Duration:

📅 Approximately **4–6 months**

Main domains:

* AI / Deep Learning
* Computer Vision
* Full-Stack Development
* Data Visualization

---

# 🔮 Future Improvements

Possible next steps:

* Real-time webcam analysis
* PDF report export
* Cloud deployment (AWS / Render)
* Better UI/UX design
* More accurate emotion models

---

# 📜 License

Educational / Academic Project.

---

<div align="center">

### ⭐ If you like this project, consider starring the repository ⭐

Made with ❤️ using Python, Flask & AI

</div>
