# Sentinel-2 Image Classification Model

A small image classification model in Python for Sentinel-2 satellite data, accompanied by several demo modules showcasing different machine learning techniques across image, text, and transactional datasets.

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
├── demo3.py                  # Association Analysis, Deep Learning, and LLM-Based Classification demonstration.
├── config.json               # Model configuration file, used to adjust randomly chosen class sets.
├── EuroSAT_RGB               # Extracted satellite data, classified into folders.
├── text                      # Sample tweet and transaction data, used for demonstrations.
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
├── demo3.py
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

Model.py is currently a work in progress that will be built upon throughout the semester. Currently, it:
- **Prepares an Image Dataset**: Creates a training and test dataset, including only the classes specified in config.json.
- **Creates and Compiles the Model**: Constructs an image classification model using the following CNN architecture:
  - Standardize RGB values to range [0,1]
  - 3 rounds of convolutional layers with 8 filters of size 3x3 followed by 2x2 max pooling layers
  - Flatten the layer
  - 3 Fully connected hidden layers with 128, 64, and 16 neurons and ReLU activation
  - Output layer with softmax activation
- **Visualizes Accuracy and Loss**: Plots model training and validation accuracy and loss by epoch, then prints the final training accuracy, validation accuracy, training loss, and validation loss.

To change the model's output, change the following values in config.json:
- ```epochs```: Changes the number of epochs that the model is set to train for.
- ```Lands```: Specifies the pool of land types to gather images from. The total collection of land cover types is listed in ```AllLands``` as a reference point.

## Demo Overview
The core model is accompanied by several demo modules showcasing different machine learning techniques. The techniques showcased in each demo are as follows:
### Demo 1: Data Preparation and Feature Extraction 
- **Edge Histogram and Similarity Measurements:** Images are converted to grayscale, and an angle is obtained for each pixel in the image. These angles are translated to a 36-dimensional edge histogram for each image. The edge histograms are then compared using Euclidean, Manhattan, and Cosine distance.
- **Histogram of Oriented Gradient (HOG) Feature Descriptor:** HOG discriptors, the counted occurrences of gradient orientation in each pixel of the image, are calculated and displayed for a single image.
- **Dimensionality Reduction using Principal Component Analysis (PCA):** Edge histograms are reduced from 36 dimensions to 2 dimensions using Principal Component Analysis. Each histogram is then plotted on the 2D coordinate plane to examine if any of the classes are visually separable.
- **Text Processing and Dimensionality Reduction** A multiclass dataset of tweets is processed and token counts and TF-IDF feature counts are extracted, with both feature counts amounting to 9728. These samples are then dimensionally reduced to 2 dimensions using PCA and plotted on the 2D coordinate plane to examine if any of the classes are visually separable.
### Demo 2: Model Performance Evaluation
- **Data Preparation:** The first 200 images from the AnnualCrop, Forest, Residential, and River classes are converted to 36-dimensional edge histograms. The dataset is then split 80/20 into a training set and a test set and standardized based on the training data.
- **Performance Comparison:** The dataset is fitted to a Naive Bayes classifier, a Neural Network classifier, and a Random Forest classifier to examine the effectiveness of each model for the classification task. For each classifier, stratified 5-fold cross validation is performed, and the accuracy and variability of each method are compared. Then, the classifiers are trained using the entire training set and their performance in classification is analyzed. Two misclassified images are examined visually.
- **Model Selection and Model Complexity:** Images from AnnualCrop and PermanentCrop, two classes which may be hard to distinguish, are classified using a support vector classifier. 6 different values of C (the regularization parameter) are used to examine its impact om model performance and complexity. Standard and stratified 5-fold cross validation are performed on the dataset, and the training and validation error of each method are plotted against C. The value of C with the lowest mean validation error from the stratified 5-fold cross-validation is selected and its test and cross-validation errors are compared to examine if they are consistent.
### Demo 3: Association Analysis, Deep Learning, and LLM-Based Text Classification
- **Association Rule Mining:** Association rules are generated for a given transaction dataset. The effect of changing the minimum support and minimum confidence parameters on the amount of rules generated is compared using a heatmap.
- **Image Classification Using CNN:** An image classification model for the classes Annual Crop, Forest, Residential, and River is constructed using two convolutional layers, two max pooling layer, and a neural network with 16 hidden nodes. To examine the effect of the convolutional layers on model accuracy, two new models are created, with the number of filters in the second convolutional layer being replaced with 4 and 16. The accuracies for all three models are then compared.
- **Multi-Label Text Classification Using BERT:** The BERT model for text emotion classification is fine-tuned for the text training dataset over 5 epochs. The fine-tuned model is evaluated on the test set using exact-match accuracy, at-least-one-label accuracy, and micro-averaged F1 score.
