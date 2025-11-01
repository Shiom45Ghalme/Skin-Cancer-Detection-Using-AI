# 🔬 Skin Cancer Detection using Deep Learning

An AI-powered web application for detecting and classifying skin lesions using deep learning. Built with TensorFlow/Keras and Flask.

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) using Transfer Learning with MobileNetV2 to classify skin lesions into 7 different categories. The system achieved **71.38% accuracy** on the test dataset.

## 📊 Dataset

- **Source**: HAM10000 (Human Against Machine with 10000 training images)
- **Total Images**: 10,015 dermatoscopic images
- **Classes**: 7 types of skin lesions

### Class Distribution:
1. **Melanocytic nevi** (Benign moles)
2. **Melanoma** (Malignant)
3. **Benign keratosis** (Benign)
4. **Basal cell carcinoma** (Malignant)
5. **Actinic keratoses** (Pre-cancerous)
6. **Vascular lesions** (Benign)
7. **Dermatofibroma** (Benign)

## 🏗️ Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Custom Layers**: Dense layers with Dropout and Batch Normalization
- **Input Size**: 224x224x3
- **Total Parameters**: 3,050,055
- **Trainable Parameters**: 790,535

## 📈 Performance

- **Test Accuracy**: 71.38%
- **Test Precision**: 85.77%
- **Test Recall**: 60.42%

## 🚀 Features

- ✅ Deep Learning model with Transfer Learning
- ✅ Web-based interface for easy image upload
- ✅ Real-time predictions with confidence scores
- ✅ Detailed probability distribution for all classes
- ✅ Risk assessment (Malignant/Pre-cancerous/Benign)
- ✅ Medical recommendations based on diagnosis
- ✅ Downloadable analysis reports
- ✅ Responsive design for mobile/desktop

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Flask
- **Data Processing**: NumPy, Pandas, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS, JavaScript

## 📁 Project Structure

