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
├── demo1.py                  # Small example demonstration of model concepts with visual output.
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

Run the demo:

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

