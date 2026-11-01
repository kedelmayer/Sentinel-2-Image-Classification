# Sentinel-2 Image Classification Model

A small image classification model in Python for Sentinel-2 satellite data, accompanied by several demo modules showcasing different machine learning techniques.

## Dataset

Data is provided on Kaggle by Salma Adel Saleh: https://www.kaggle.com/datasets/salmaadell/eurosat-rgb.
- **Sample Description**: 64 x 64 pixel RGB images in .jpg format, sourced from Sentinel-2 satellite imagery.
- **Classes**: 10 land types - forests, residential areas, herbaceous vegetation, water bodies (seas/lakes) annual farmland, permanent farmland, industrial areas, rivers, highways and pastures.
- **Images Per Class**: 2000 images for each land type.


## Repository Structure

```text
sentinel_2_image_classification/
├── model.py                  # Core image classification model [Work in Progress].
├── demo1.py                  # Data Preparation and Feature Extraction demonstration.
├── demo2.py                  # Model Performance Evaluation demonstration.
├── config.json               # Model configuration file, used to adjust randomly chosen class sets.
├── EuroSAT_RGB               # Extracted satellite data, classified into folders.
├── text                      # Sample tweet data, used in the demos to display machine learning concepts with text.
├── requirements.txt
└── README.md
```


## Quick Start

Download the data from Kaggle: https://www.kaggle.com/datasets/salmaadell/eurosat-rgb

Extract archive.zip and move EuroSAT_RGB into sentinel_2_image_classification. Check that the folder structure matches below:

```text
sentinel_2_image_classification/
├── EuroSAT_RGB               # Subfolder
├── text                      # Subfolder 
├── config.json
├── demo1.py
├── demo2.py
├── model.py
├── README.md
└── requirements.txt                  
```


Set up a Python environment and install dependencies from `requirements.txt`:

```bash
cd ~/sentinel_2_image_classification      # Navigate to sentinel_2_image_classification
python3 -m venv .venv
.venv\Scripts\activate.ps1                
# For Linux, use:  source .venv/bin/activate
pip3 install -r requirements.txt
```

Run the first demo:

```bash
python3 demo1.py
```


## Using the Config File and Model.py

The image classification model is currently a work in progress that will be built upon throughout the semester. Currently, it:
- **Selects a Sample Image Set**: Randomly selects a set of images based on the classes and images per class specified in config.json.
- **Extracts Edge Histograms**: Extracts each image's edge histogram.
- **Compares 2D-Reduced Histogram Data**: Reduces each image's edge histogram to 2 dimensions using PCA, then plots the points to compare the values.

To change the model's output, change the following values in config.json:
- ```images_per_land```: Changes the number of images chosen from each land type specified.
- ```bins```: Specifies the number of bins used to generate each image's edge histogram.
- ```Lands```: Specifies the pool of land types to gather images from. The total collection of land cover types is listed in ```AllLands``` as a reference point.

## Demo Overview
The core model is accompanied by several demo modules showcasing different machine learning techniques. The techniques showcased in each demo are as follows:
**Demo 1 - Data Preparation and Feature Extraction**: 
- *Edge Histogram and Similarity Measurements:* Images are converted to grayscale, and an angle is obtained for each pixel in the image. These angles are translated to a 36-dimensional edge histogram for each image. The edge histograms are then compared using Euclidean, Manhattan, and Cosine distance.
- *Histogram of Oriented Gradient (HOG) Feature Descriptor:* HOG discriptors, the counted occurrences of gradient orientation in each pixel of the image, are calculated and displayed for a single image.
- *Dimensionality Reduction using Principal Component Analysis (PCA):* Edge histograms are reduced from 36 dimensions to 2 dimensions using Principal Component Analysis. Each histogram is then plotted on the 2D coordinate plane to examine if any of the classes are visually separable.
- *Text Processing and Dimensionality Reduction* A multiclass dataset of tweets is processed and token counts and TF-IDF feature counts are extracted, with both feature counts amounting to 9728. These samples are then dimensionally reduced to 2 dimensions using PCA and plotted on the 2D coordinate plane to examine if any of the classes are visually separable.
- **Demo 2 - Model Performance Evaluation**:
- *Data Preparation:* The first 200 images from the AnnualCrop, Forest, Residential, and River classes are converted to 36-dimensional edge histograms. The dataset is then split 80/20 into a training set and a test set and standardized based on the training data.
- *Performance Comparison:* The dataset is fitted to a Naive Bayes classifier, a Neural Network classifier, and a Random Forest classifier to examine the effectiveness of each model for the classification task. For each classifier, stratified 5-fold cross validation is performed, and the accuracy and variability of each method are compared. Then, the classifiers are trained using the entire training set and their performance in classification is analyzed. Two misclassified images are examined visually.
- *Model Selection and Model Complexity:* Images from AnnualCrop and PermanentCrop, two classes which may be hard to distinguish, are classified using a support vector classifier. 6 different values of C (the regularization parameter) are used to examine its impact om model performance and complexity. Standard and stratified 5-fold cross validation are performed on the dataset, and the training and validation error of each method are plotted against C. The value of C with the lowest mean validation error from the stratified 5-fold cross-validation is selected and its test and cross-validation errors are compared to examine if they are consistent.
