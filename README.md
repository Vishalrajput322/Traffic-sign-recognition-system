Traffic Sign Recognition System
Introduction
This project is a Traffic Sign Recognition System that classifies various traffic signs using a Convolutional Neural Network (CNN) model. The system is trained on the German Traffic Sign Benchmark (GTSRB) dataset, which contains images of traffic signs from different categories. This project utilizes deep learning techniques to achieve high accuracy in recognizing and classifying traffic signs.

Project Structure
Dataset: GTSRB dataset, which contains over 50,000 images of 43 different classes of traffic signs.
Model: A deep learning model built using TensorFlow and Keras with several layers including convolutional, pooling, and fully connected layers.
GUI: A simple GUI using Tkinter for users to upload images and get real-time predictions on traffic signs.
Files
traffic_sign_recognition.py: The main Python script to train, test, and predict traffic signs.
model.h5: Pre-trained model saved after training for traffic sign classification.
gui.py: The Tkinter-based GUI allowing users to load an image and receive the traffic sign classification.
README.md: This file, describing the project details.
Prerequisites
Before running the project, make sure you have the following dependencies installed:

Python 3.x
TensorFlow
Keras
OpenCV
Numpy
Pandas
Matplotlib
Tkinter
PIL (Pillow)
You can install the required packages using pip:


pip install tensorflow keras opencv-python numpy pandas matplotlib pillow tkinter
Dataset
The German Traffic Sign Benchmark (GTSRB) dataset can be downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign. After downloading, place the dataset into the appropriate folder and load it in the script.

Dataset Structure
Training images are organized into different class folders (from 0 to 42).
Each folder contains images of traffic signs that belong to that specific class.
Training the Model
The model is a Convolutional Neural Network (CNN) built using the following layers:

Convolutional Layers for feature extraction.
Max-Pooling Layers to reduce dimensionality.
Flatten Layer to transform the feature map into a 1D array.
Dense Layers for final classification.
