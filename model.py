import os, random, json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import filters, exposure
from skimage.color import rgb2gray
from sklearn.utils import Bunch
from sklearn.decomposition import PCA
     
# Outputs n randomly selected images from each land specified in config.json.
#   Return: An array of images.
def randomSet():
    with open("config.JSON", "r") as f:         #Loads data from config.json
        config = json.load(f)

    images = []
    for i in config['Lands']:                   # Randomly selects images
        set = "EuroSAT_RGB/"+i+"/"
        for j in range(0, config['images_per_land']):
            img = mpimg.imread(set+random.choice(os.listdir(set)))
            images.append(img)
    return images


# Randomly selects n images each from each land specified in config.json
# and extracts their edge histograms and target classes.
#   Return: A Bunch object containing an array of histograms, an array 
#       of classes, a Pandas DataFrame containing the histograms and classes,
#       and an array containing the names of the target classes.
def randomHistBunch():
    with open("config.json", "r") as f:                                 # Loads data from config.json
        config = json.load(f)

    data = []
    target = []
    count = 0
    for i in config['Lands']:                                           # Iterate through lands
        set = "EuroSAT_RGB/"+i+"/"
        for j in range(0, config['images_per_land']):
            img = mpimg.imread(set+random.choice(os.listdir(set)))      # Randomly select images
            hist, c = histogram(img, config['bins'])                    # Extract image's edge histogram and bin centers
            data.append(hist)                                           # Add histogram to array
            target.append(count)                                        # Add class to array
        count = count+1

    data_df = pd.DataFrame(data, columns=list(range(config['bins'])), copy=False)   # Assemble histogram DataFrame
    target_df = pd.DataFrame(target, columns=["target"])                # Assemble target DataFrame
    combined_df = pd.concat([data_df, target_df], axis=1)               # Combine DataFrames
    X = combined_df[list(range(config['bins']))]
    y = combined_df[["target"]]

    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    return Bunch( data=X, target=y, frame=combined_df, target_names=config['Lands'])


# Calculates the angles between horizontal and vertical operators.
#   Return: Calculated angle.
def angle(dx, dy):
    return np.mod(np.arctan2(dy, dx), np.pi)


# Extracts an image's edge histogram.
#   img: The image to extract the histogram from.
#   Return: An array containing an array of histogram 
#       values and an array of bin center values.
def histogram(img, bins):
    gray = rgb2gray(img)                                                # Convert color images to grayscale
    angles = angle(filters.sobel_h(gray), filters.sobel_v(gray))        # Obtain an angle for each pixel in the images
    hist = exposure.histogram(angles, nbins=bins, source_range='image', 
                                   normalize=False, channel_axis=None)  # Obtain edge histograms
    return(hist)


# Plots the 2D points from a dimensionally-reduced edge histogram, using different
# colors for data from the different classes.
#   histogramBunch: A Bunch object containing the histogram and class data.
#   reducedSet: An array containing the PCA-Reduced set of edge histograms.
def plot_PCA(histogramBunch, reducedSet):
    fig = plt.figure(1, figsize=(8, 6))                                 
    ax = fig.add_subplot()
    scatter = ax.scatter(
        reducedSet[:, 0],                                   # Plot the points
        reducedSet[:, 1],
        c=histogramBunch.target,                            # Use target array to color points
        s=40
    )

    ax.set(                                                 # Add titles
        title="PCA-Reduced Feature Visualization",
        xlabel="1st Principal Component",
        ylabel="2nd Principal Component",
    )
    ax.xaxis.set_ticklabels([])                             # Add Tick Labels
    ax.yaxis.set_ticklabels([])

    legend1 = ax.legend(                                    # Add a legend
        scatter.legend_elements()[0],
        histogramBunch.target_names,
        loc="upper right",
        title="Land Cover",
    )
    ax.add_artist(legend1)
    plt.show()                                              # Show the plot



## ------------------------------------------------------------------------------------------------ ##
## Dimensionality Reduction using Principal Component Analysis (PCA) ##

histogramBunch = randomHistBunch()                                      # Randomly select an image set and extract edge histograms
reducedSet = PCA(n_components=2).fit_transform(histogramBunch.data)     # Reduce the histogram set to 2 dimensions

plot_PCA(histogramBunch, reducedSet)                                    # Plot the 2D points using a different color for each class


