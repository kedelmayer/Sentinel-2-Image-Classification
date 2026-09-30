import os, random, json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import filters, exposure
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.metrics import pairwise
from sklearn.utils import Bunch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Randomly selects 50 images each from Annual Crop, Herbaceous Vegetation, Residential,
# and River classes and extracts their edge histograms and target classes.
#   Return: A Bunch object containing an array of histograms, an array 
#       of classes, a Pandas DataFrame containing the histograms and classes,
#       and an array containing the names of the target classes and the column header name.
def randomHistBunch():
    lands = ["AnnualCrop", "HerbaceousVegetation", "Residential", "River"]
    data = []
    target = []
    count = 0
    for i in lands:                                                     # Iterate through lands
        set = "EuroSAT_RGB/"+i+"/"
        for j in range(0, 50):
            img = mpimg.imread(set+random.choice(os.listdir(set)))      # Randomly select images
            hist, c = histogram(img)                                    # Extract image's edge histogram and bin centers
            data.append(hist)                                           # Add histogram to array
            target.append(count)                                        # Add class to array
        count = count+1

    return convertBunch(data, target, lands, 36, "Land Cover")


# Converts a data array and its class information into a Bunch object.
#   data: An array of data.
#   target: An array containing the class for each data sample.
#   names: An array containing the class names corresponding to the target data.
#   data_cols: The number of features for the data array.
#   target_head: A string representing the name of the target array.
#   sparse_data: A boolean indicating whether the data array is sparse. False by default.
#   Return: A Bunch object containing the data array, the class array, a Pandas DataFrame
#       containing the data and classes, an array containing the names of the target 
#       classes, and the target column header name.
def convertBunch(data, target, names, data_cols, target_head, sparse_data=False):

    if not sparse_data:                                                             # Assemble data DataFrame
        data_df = pd.DataFrame(data, columns=list(range(data_cols)), copy=False)    
    else:
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=list(range(data_cols)))

    target_df = pd.DataFrame(target, columns=[target_head])                         # Assemble target DataFrame
    combined_df = pd.concat([data_df, target_df], axis=1)                           # Combine DataFrames
    X = combined_df[list(range(data_cols))]
    y = combined_df[[target_head]]

    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    return Bunch( data=X, target=y, frame=combined_df, target_names=names, target_head=target_head)


# Calculates the angles between horizontal and vertical operators.
#   Return: Calculated angle.
def angle(dx, dy):
    return np.mod(np.arctan2(dy, dx), np.pi)


# Extracts an image's edge histogram.
#   img: The image to extract the histogram from.
#   Return: An array containing an array of histogram 
#       values and an array of bin center values.
def histogram(img):
    gray = rgb2gray(img)                                                # Convert color images to grayscale
    angles = angle(filters.sobel_h(gray), filters.sobel_v(gray))        # Obtain an angle for each pixel in the images
    hist = exposure.histogram(angles, nbins=36, source_range='image', 
                                   normalize=False, channel_axis=None)  # Obtain edge histograms
    return(hist)


# Plots an image and its corresponding edge histogram.
#   image: The image to plot.
#   hist: An array containing the edge histogram data.
#   axes: The subplot to plot the image and histogram in.
#   Return: The plotted image and histogram.
def plot_img_and_hist(image, hist, axes):
    ax_img, ax_hist = axes

    ax_img.imshow(image)                    # Display image
    ax_img.set_axis_off()

    counts, bin_centers = hist              # Display histogram
    ax_hist.plot(bin_centers, counts)
    ax_hist.set_xlabel('Bins')
    ax_hist.set_ylabel('Pixel Count')

    return ax_img, ax_hist


# Plots the 2D points from a dimensionally-reduced set of data on a figure subplot, 
# using different colors for data from the different classes.
#   bunch: A Bunch object containing the original data and and target information.
#   reducedSet: An array containing the PCA-Reduced set of edge histograms.
#   ax: The subplot to plot the data on.
#   title: A string representing the title of the subgraph.
def plot_PCA(bunch, reducedSet, ax, title):                                
    scatter = ax.scatter(
        reducedSet[:, 0],                                   # Plot the points
        reducedSet[:, 1],
        c=bunch.target,                                     # Use target array to color points
        s=40
    )

    ax.set(                                                 # Add titles
        title=title,
        xlabel="1st Principal Component",
        ylabel="2nd Principal Component",
    )
    ax.xaxis.set_ticklabels([])                             # Add Tick Labels
    ax.yaxis.set_ticklabels([])

    legend1 = ax.legend(                                    # Add a legend
        scatter.legend_elements()[0],
        bunch.target_names,
        loc="upper right",
        title=bunch.target_head,
    )
    ax.add_artist(legend1)


## ------------------------------------------------------------------------------------------------ ##
## Feature Extraction: Edge histogram and Similarity Measurements ##
# Choose one example image from each class.
i1 = mpimg.imread('EuroSAT_RGB/AnnualCrop/AnnualCrop_30.jpg')
i2 = mpimg.imread('EuroSAT_RGB/HerbaceousVegetation/HerbaceousVegetation_68.jpg')
i3 = mpimg.imread('EuroSAT_RGB/Residential/Residential_43.jpg')
i4 = mpimg.imread('EuroSAT_RGB/River/River_41.jpg')
images = [i1, i2, i3, i4]

# Extract edge histograms.
histograms = []
for i in images:
    h = histogram(i)
    histograms.append(h)

# Plot images with their corresponding edge histogram values.
fig, axes = plt.subplots(2, 4, figsize=(15, 5))
fig.suptitle('Images and Corresponding Edge Histogram Values', fontsize=16)
titles = ['Annual Crop', 'HerbaceousVegetation', 'Residential', 'River']

for i in range(0, 4):
    ax_img, ax_hist = plot_img_and_hist(images[i], histograms[i], axes[:, i])
    ax_img.set_title(titles[i])

plt.tight_layout()
plt.show()

# Perform histogram comparison between the Annual Crop and River edge histograms.
h1 = histograms[0][0].reshape(1, -1)    # Reshape the Annual Crop histogram array to (1,36)
h4 = histograms[3][0].reshape(1, -1)    # Reshape the River histogram array to (1,36)

eDistance = pairwise.euclidean_distances(h1, h4)        # Calculate Euclidean Distance
mDistance = pairwise.manhattan_distances(h1, h4)        # Calculate Manhattan Distance
cDistance = pairwise.cosine_distances(h1, h4)           # Calculate Cosine Distance

print("--------------------------------------------------------------")
print("     Histogram Comparison Analysis: Annual Crop and River     ")
print("--------------------------------------------------------------")
print(" Euclidean Distance:  "+str(eDistance[0][0]))
print(" Manhattan Distance:  "+str(mDistance[0][0]))
print(" Cosine Distance:  "+str(cDistance[0][0]))
print("--------------------------------------------------------------")


## ------------------------------------------------------------------------------------------------ ##
## Histogram of Oriented Gradient (HOG) Feature Descriptor ##
# Compute the HOG feature descriptors for the Annual Crop example image.
image = images[0]

fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=-1,
)

# Visualize the image and the HOG discriptors.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.set_title('Annual Crop Image')                      # Plot image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis('off')

ax2.set_title('Histogram of Oriented Gradients')        # Plot HOG descriptors
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.axis('off')

plt.show()


## ------------------------------------------------------------------------------------------------ ##
## Dimensionality Reduction using Principal Component Analysis (PCA) ##

# - Histograms - #
# Plot the dimensionally reduced data for a set of histograms.
histogramBunch = randomHistBunch()                                      # Randomly select an image set and extract edge histograms
reducedSet = PCA(n_components=2).fit_transform(histogramBunch.data)     # Reduce the histogram set from 36 to 2 dimensions

# Plot the 2D points using four different colors for data from the four classes 
fig = plt.figure(1, figsize=(8, 6))                                                   
plot_PCA(histogramBunch, reducedSet, fig.add_subplot(), "PCA-Reduced Feature Visualization")
plt.show()


# - Text Data - #
# Plot the dimensionally reduced data for an array of text data.
f =  open("text/train.JSON", "r")        # Load data from train.json
json_array = json.load(f)

tweets = []                              # Collect text data from JSON objects
for i in json_array:
    tweets.append(i['Tweet'])

# Extract token counts.
cvectorizer = CountVectorizer()
cX = cvectorizer.fit_transform(tweets)
print("Token count dimensionality:  "+str(cX.shape))              # Print the dimensionality of the vector representation

# Extract TF-IDF feature counts.
tvectorizer = TfidfVectorizer()
tX = tvectorizer.fit_transform(tweets)
print("TF-IDF feature count dimensionality:  "+str(tX.shape))     # Print the dimensionality of the vector representation

# Choose 4 sample classes (Anger, Fear, Optimism, Trust) and process the array to collect text and class information.
text = []
targets = []
emotions = ["anger", "fear", "optimism", "trust"]

for i in json_array:
    if i['anger'] ==  True:
        text.append(i['Tweet'])
        targets.append(0)
    elif i['fear'] ==  True:
        text.append(i['Tweet'])
        targets.append(1)
    elif i['optimism'] ==  True:
        text.append(i['Tweet'])
        targets.append(2)
    elif i['trust'] ==  True:
        text.append(i['Tweet'])
        targets.append(3)

# Extract token counts for the sample pool.
cX = cvectorizer.fit_transform(text)
cSamples, cFeatures = cX.shape

# Extract TF-IDF feature counts for the sample pool.
tX = tvectorizer.fit_transform(text)
tSamples, tFeatures = tX.shape

# Convert the feature count data to Bunch objects.
countBunch = convertBunch(cX, targets, emotions, cFeatures, "Emotion", sparse_data=True)
tfidfBunch = convertBunch(tX, targets, emotions, tFeatures, "Emotion", sparse_data=True)

# Dimensionally reduce the objects to 2 dimensions using PCA.
c_reduced = PCA(n_components=2).fit_transform(countBunch.data)
t_reduced = PCA(n_components=2).fit_transform(tfidfBunch.data)

# Plot the 2D points using 4 different colors for data from the 4 classes.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('PCA-Reduced Feature Visualizations', fontsize=16)
plot_PCA(countBunch, c_reduced, axes[0], "Token Count Features")
plot_PCA(tfidfBunch, t_reduced, axes[1], "TF-IDF Features")

plt.tight_layout()
plt.show()