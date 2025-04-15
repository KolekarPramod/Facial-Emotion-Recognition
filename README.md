# **DeepFER: Facial Emotion Recognition Using Deep Learning**

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## **Overview**
**DeepFER** is a facial emotion recognition system that leverages advanced deep learning techniques, including Convolutional Neural Networks (CNNs) and Transfer Learning, to classify emotions from human facial expressions. The model recognizes multiple emotions such as happiness, sadness, anger, surprise, and more. This project focuses on creating an efficient, high-performance system for real-time emotion classification, which can be applied in areas like human-computer interaction, mental health monitoring, and customer service.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## **Project Background**
Facial Emotion Recognition (FER) has gained widespread attention due to its vast potential in improving human-computer interactions and enhancing user experiences across various industries. The goal of this project is to create an emotion recognition system that can efficiently classify human emotions based on facial expressions using deep learning techniques.

The project utilizes Convolutional Neural Networks (CNNs) to automatically extract features from facial images and classify them into one of seven emotion categories. We also implement **Transfer Learning** to fine-tune pre-trained models, speeding up the training process and improving performance.

## **Dataset Overview**
The dataset used in this project consists of facial images categorized into seven emotion classes:
- **Angry**
- **Sad**
- **Happy**
- **Fear**
- **Neutral**
- **Disgust**
- **Surprise**

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

### **Key Dataset Features:**
- High-quality facial images with diverse backgrounds and lighting conditions.
- Includes both posed and spontaneous expressions.
- Data augmentation techniques such as rotation, scaling, and flipping were applied to increase dataset variability.
- Annotated labels corresponding to each image’s emotion class.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## **Project Goal**
The primary goal of **DeepFER** is to create a reliable system capable of real-time facial emotion recognition. By using CNNs and Transfer Learning, the project aims to handle variability in facial expressions, lighting conditions, and image qualities, achieving high accuracy for real-time applications.

## **Specific Objectives**
- **Data Collection & Preprocessing**: Collect and preprocess a diverse dataset of facial images, applying augmentation techniques for better model generalization.
- **Model Development**: Implement a CNN architecture for emotion classification, using Transfer Learning to fine-tune pre-trained models.
- **Training & Evaluation**: Train the model, optimize hyperparameters, and evaluate its performance on validation data.
- **Real-Time Processing**: Implement algorithms for real-time emotion recognition from live video feeds or images.
- **Performance Optimization**: Improve model efficiency and reduce latency to meet real-time processing requirements.

## **Model Development**

### **Keras CNN Model from Scratch:**

A Convolutional Neural Network (CNN) was built from scratch using Keras to recognize emotions from facial images. Below is the architecture of the model:


### **Model Performance (Scratch CNN Model):**

- **Accuracy**: 63.39%
- **Precision**: 63.01%
- **Recall**: 63.39%
- **F1 Score**: 62.85%

This model showed reasonable performance but required further enhancement, which led to the use of **Transfer Learning** to improve results.

### **Transfer Learning with YOLO:**

To further improve the model's performance, I employed **Transfer Learning** using pre-trained weights with the YOLO (You Only Look Once) architecture. Here's how the process was carried out:

### **Model Performance (Transfer Learning with YOLO):**

- **Top-1 Accuracy**: 56%
- **Top-5 Accuracy**: 98.6%

While the transfer learning approach improved top-5 accuracy significantly, further refinements in the model are still being explored to achieve higher top-1 accuracy.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


## **Usage**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DeepFER.git
   cd DeepFER
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   For local development, use the following command:
   ```bash
   python app.py
   ```

4. **Real-time Emotion Recognition**:
   The model can process live video feeds or images and classify the emotions from the faces in real-time.

## **Contributing**
Contributions are welcome! If you'd like to enhance this project, feel free to fork the repository and create a pull request.
