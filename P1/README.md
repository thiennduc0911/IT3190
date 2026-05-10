# Vietnamese News Classification

This project focuses on Vietnamese text classification using traditional Machine Learning methods and TF-IDF feature extraction.

## Project Overview

The goal of this project is to classify Vietnamese news articles into different categories automatically.  
The dataset is collected from VnExpress news articles.

The project includes:

- Vietnamese text preprocessing
- Word segmentation using PyVi
- Stopword removal
- TF-IDF feature extraction
- Training and evaluation of multiple Machine Learning models
- Model comparison and selection

## Technologies Used

- Python
- Scikit-learn
- PyVi
- Matplotlib
- Jupyter Notebook

## Machine Learning Models

The following models were implemented and evaluated:

- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

## Features

- Vietnamese tokenization
- TF-IDF vectorization
- Hyperparameter tuning using GridSearchCV
- Accuracy evaluation
- Confusion matrix visualization

## Dataset

The dataset is not included in this repository due to size limitations.

Dataset source:
- VnExpress Vietnamese news dataset
- https://users.soict.hust.edu.vn/khoattq/ml-dm-course

After downloading the dataset, place it in:

```text
data/news_vnexpress/
```
## Project Structure

text_classification/
│
├── main.py
├── experiment.ipynb
├── requirements.txt
├── README.md
└── data/

## Requirements
Install required libraries:

pip install -r requirements.txt

## How to run
Run the Python script:

python main.py

Or open the notebook:

jupyter notebook experiment.ipynb

## Results
The project evaluates different Machine Learning algorithms and selects the best-performing model based on classification accuracy.
