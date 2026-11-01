import os, random, json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import filters, exposure
from skimage.color import rgb2gray
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# Randomly selects images from a given set of land types and extracts their edge histograms and target classes.
#   lands: An array containing the classes from which to select images.
#   num_images: an integer indicating the number of images to select from each class.
#   rand: A bool indicating whether or not to randomize chosen samples. True by default.
#   Return: A Bunch object containing an array of histograms, an array 
#       of classes, a Pandas DataFrame containing the histograms and classes,
#       and an array containing the names of the target classes and the column header name.
def randomHistBunch(lands, num_images, rand=True):
    data = []
    target = []
    count = 0
    if random:
        for i in lands:                                                     # Iterate through lands
            set = "EuroSAT_RGB/"+i+"/"
            for j in range(0, num_images):
                img = mpimg.imread(set+random.choice(os.listdir(set)))      # Randomly select images
                hist, c = histogram(img)                                    # Extract image's edge histogram and bin centers
                data.append(hist)                                           # Add histogram to array
                target.append(count)                                        # Add class to array
            count = count+1
    else:
        for i in lands:                                                     # Iterate through lands
            for j in range(1, num_images+1):
                img = mpimg.imread(f"EuroSAT_RGB/"+i+"/"+i+"_"+str(j)+".jpg")    # Sequentially select images
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

# Prints a summary of an input dataset.
#   split: A string denoting the name of the split.
#   data: A pandas dataframe holding the dataset data.
#   target: A pandas dataframe holding the dataset targets.
#   target_names: An array containing the dataset target names.
def print_dataset_summary(split, data, target, target_names):
    print("\n===================== DATASET SUMMARY ======================")
    print(f"Split:                              {split}")
    print(f"Samples:                            {len(data)}")
    print(f"Classes:                            {len(target_names)}")
    print(f"Images per class:")
    feature = 0
    for i in target_names:
        count = 0
        for j in target:
            if j == feature: count = count + 1
        print(f"    {i}:        {count}")
    print(f"Feature vector dimensionality:      {(data.shape[1])}")
    print("==============================================================\n")

## ------------------------------------------------------------------------------------------------ ##
## Data Preparation ##
# Prepare edge histogram dataset.
dataset = randomHistBunch(["AnnualCrop", "Forest", "Residential", "River"], 200, rand=False)    # Select an image set and extract edge histograms 
labels = dataset.target_names
print_dataset_summary("Full Dataset", dataset.data, dataset.target, labels)

# Split the dataset into a training and test set.
trainData, testData, trainTarget, testTarget = train_test_split(                                  # Split the dataset using an 80/20 split
    dataset.data, dataset.target, test_size=0.2)

print_dataset_summary("Train", trainData, trainTarget, labels)                                    # Show training split summary
print_dataset_summary("Test", testData, testTarget, labels)                                       # Show testing split summary

# Standardize the dataset.
scaler = preprocessing.StandardScaler().fit(trainData)
trainData = scaler.transform(trainData)                                                           # Standardize training data
testData = scaler.transform(testData)                                                             # Standardize test data

## ------------------------------------------------------------------------------------------------ ##
## Classification Method Performance Evaluation ##

skf = StratifiedKFold(n_splits=5, shuffle=False)                    # Prepare stratified 5-fold cross-validation object

# Use a Naive Bayes Classifier to classify the data.
NBclf = GaussianNB()                                                # GaussianNB with default parameters
NBclf.fit(trainData, trainTarget)                                   # Fit the model to the training data

NBresults = cross_validate(NBclf, trainData, trainTarget, cv=skf,   # Perform cross-validation on the naive bayes classifier model
    scoring=["accuracy"], n_jobs=-1, return_train_score=True)

NB_test_pred = NBclf.predict(testData)                              # Generate a model prediction for held-out test data

index = 0
misclass_indices = []
misclass_labels = []
for item, label in zip(trainData, trainTarget):
    result = NBclf.predict([item])
    if result != label:                                             # Collect misclassified samples
        misclass_indices.append(index)                              # Collect sample index
        misclass_labels.append((result, label))                     # Collect labels (predicted, actual)
    index = index + 1

# Use a neural network to classify the data.
NNclf = MLPClassifier(hidden_layer_sizes=(10,10,10))                # MLPClassifier with 3 hidden layers of 10 nodes each
NNclf.fit(trainData, trainTarget)                                   # Fit the model to the training data

NNresults = cross_validate(NNclf, trainData, trainTarget, cv=skf,   # Perform cross-validation on the neural network model
    scoring=["accuracy"], n_jobs=-1, return_train_score=True)

NN_test_pred = NNclf.predict(testData)                              # Generate a model prediction for held-out test data


# Use a random forest classifier to classify the data.
RFclf = RandomForestClassifier()                                    # RandomForestClassifier with default parameters
RFclf.fit(trainData, trainTarget)                                   # Fit the model to the training data

RFresults = cross_validate(RFclf, trainData, trainTarget, cv=skf,   # Perform cross-validation on the random forest model
    scoring=["accuracy"], n_jobs=-1, return_train_score=True)

RF_test_pred = RFclf.predict(testData)                              # Generate a model prediction for held-out test data


# Summarize cross-validation results.
print("\n======== CLASSIFIER CROSS-VALIDATION RESULT SUMMARY ========")
print(f"Naive Bayes Classifier:")
print(f"    Mean Validation Accuracy:       {NBresults["test_accuracy"].mean()}") 
print(f"    Standard Deviation:             {NBresults["test_accuracy"].std()}")
print(f"Neural Network Classifier:")
print(f"    Mean Validation Accuracy:       {NNresults["test_accuracy"].mean()}") 
print(f"    Standard Deviation:             {NNresults["test_accuracy"].std()}")
print(f"Random Forest Classifier:")
print(f"    Mean Validation Accuracy:       {RFresults["test_accuracy"].mean()}") 
print(f"    Standard Deviation:             {RFresults["test_accuracy"].std()}")
print("==============================================================\n")

# Summarize model evaluation results.
print("\n=========== CLASSIFIER EVALUATION RESULT SUMMARY ===========")
print(f"Naive Bayes Classifier:")
print(f"    Test Accuracy:                   {accuracy_score(testTarget, NB_test_pred)}") 
print(f"    Macro F-Measure:                 {f1_score(testTarget, NB_test_pred, average='macro')}")
print(f"Neural Network Classifier:")
print(f"    Test Accuracy:                   {accuracy_score(testTarget, NN_test_pred)}") 
print(f"    Macro F-Measure:                 {f1_score(testTarget, NN_test_pred, average='macro')}")
print(f"Random Forest Classifier:")
print(f"    Test Accuracy:                   {accuracy_score(testTarget, RF_test_pred)}") 
print(f"    Macro F-Measure:                 {f1_score(testTarget, RF_test_pred, average='macro')}")
print("==============================================================\n")

# Show confusion matrices for the three classification methods.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices', fontsize=16)
classifiers = [NBclf, NNclf, RFclf]
titles = ['Naive Bayes Classifier', 'Neural Network', 'Random Forest']

for i in range(0, 3):
    disp = ConfusionMatrixDisplay.from_estimator(               # Plot confusion matrix
        classifiers[i],
        testData,
        testTarget,
        display_labels=labels,
        cmap=plt.cm.Blues,
        ax=axes[i]
    )
    disp.ax_.set_title(titles[i])                               # Set subplot title

    print(titles[i])
    print(disp.confusion_matrix)                                # Print confusion matrix to terminal

plt.tight_layout()
plt.show()

# Display an image from AnnualCrop and River, the most commonly confused classes.
imgIDs = [misclass_indices[0], misclass_indices[len(misclass_indices)-1]]
images = []
for id in imgIDs:                                                                       # Correlate histograms back to their images
    if id in range(0, 199):
        img = mpimg.imread(f"EuroSAT_RGB/AnnualCrop/AnnualCrop_"+str(id+1)+".jpg")
    elif id in range(200, 399):
        img = mpimg.imread(f"EuroSAT_RGB/Forest/Forest_"+str(id+1)+".jpg")
    elif id in range(400, 599):
        img = mpimg.imread(f"EuroSAT_RGB/Residential/Residential_"+str(id+1)+".jpg")
    elif id in range(600, 799):
        img = mpimg.imread(f"EuroSAT_RGB/River/River_"+str(id+1)+".jpg")
    images.append(img)

label1 = misclass_labels[0]
label2 = misclass_labels[len(misclass_labels)-1]
lands = ["AnnualCrop", "Forest", "Residential", "River"]
[x] = label1[0]                                                                          # Contextualize image 1 labels 
l1p = lands[x]
l1a = lands[label1[1]]
title1 = "Predicted as " + str(l1p) +", Actually " + str(l1a)
[x] = label2[0]                                                                          # Contextualize image 2 labels 
l2p = lands[x]
l2a = lands[label2[1]]
title2 = "Predicted as " + str(l2p) +", Actually " + str(l2a)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

ax1.set_title(title1)                                  # Plot image 1
ax1.imshow(images[0], cmap=plt.cm.gray)
ax1.axis('off')
ax2.set_title(title2)                                  # Plot image 2
ax2.imshow(images[1], cmap=plt.cm.gray)
ax2.axis('off')

plt.show()

## ------------------------------------------------------------------------------------------------ ##
## Model Selection and Model Complexity ##
# Select images from two classes that may be easily confused.
dataset = randomHistBunch(["AnnualCrop", "PermanentCrop"], 400)    # Randomly select an image set and extract edge histograms 
labels = dataset.target_names

# Split and standardize the dataset.
trainData, testData, trainTarget, testTarget = train_test_split(                    # Split the dataset using an 80/20 split
    dataset.data, dataset.target, test_size=0.2)

scaler = preprocessing.StandardScaler().fit(trainData)
trainData = scaler.transform(trainData)                                             # Standardize training data
testData = scaler.transform(testData)                                               # Standardize test data

# Use a Support Vector classifiers across 6 different regularization parameters to classify the data.
C1svc = LinearSVC(C=0.01).fit(trainData, trainTarget)               # LinearSVC with C = 0.01
C2svc = LinearSVC(C=0.1).fit(trainData, trainTarget)                # LinearSVC with C = 0.1
C3svc = LinearSVC(C=1).fit(trainData, trainTarget)                  # LinearSVC with C = 1
C4svc = LinearSVC(C=10).fit(trainData, trainTarget)                 # LinearSVC with C = 10
C5svc = LinearSVC(C=100).fit(trainData, trainTarget)                # LinearSVC with C = 100
C6svc = LinearSVC(C=1000).fit(trainData, trainTarget)               # LinearSVC with C = 1000
svcs = [C1svc, C2svc, C3svc, C4svc, C5svc, C6svc]
cvals = [0.01,0.1,1,10,100,1000]

# Evaluate the models using standard cross-validation.
kf = KFold(n_splits=5, shuffle=True)                                # Prepare standard 5-fold cross-validation object
stdTrain = []
stdTest = []
for i in svcs:
    result = cross_validate(i, trainData, trainTarget, cv=kf, scoring=["accuracy"], n_jobs=-1, return_train_score=True)
    stdTrain.append(1-(result["train_accuracy"].mean()))                 # Calculate training error
    stdTest.append(1-(result["test_accuracy"].mean()))                   # Calculate test error

# Evaluate the models using stratified cross-validation.
skf = KFold(n_splits=5, shuffle=True)                                # Prepare standard 5-fold cross-validation object
strTrain = []
strTest = []
for i in svcs:
    result = cross_validate(i, trainData, trainTarget, cv=skf, scoring=["accuracy"], n_jobs=-1, return_train_score=True)
    strTrain.append(1-(result["train_accuracy"].mean()))                 # Calculate training error
    strTest.append(1-(result["test_accuracy"].mean()))                   # Calculate test error

# Plot an error graph.
x = [0.01,0.1,1,10,100,1000]
plt.xscale('log')
plt.plot(x, stdTrain, label = "Standard 5-fold CV training error")
plt.plot(x, stdTest, label = "Standard 5-fold CV validation error")
plt.plot(x, strTrain, label = "Stratified 5-fold CV training error")
plt.plot(x, strTest, label = "Stratified 5-fold CV validation error")
plt.xlabel("Regularization Parameter C")
plt.ylabel("Mean Error %")
plt.legend()
plt.show()

# Evaluate the test accuracy of a LinearSVC using the value of C with the lowest mean stratified 5-fold CV error.
bestIndex = 0
lowest = strTest[0]
i = 0
for j in strTest:
    if j < lowest:
        lowest = j
        bestIndex = i
    i = i + 1
bestC = svcs[bestIndex]

SVC_test_pred = bestC.predict(testData)                              # Generate a model prediction for held-out test data
print(f"Test Error at C = {str(cvals[bestIndex])}: {1-(accuracy_score(testTarget, SVC_test_pred))}")
print(f"Cross-Validation Error at C = {str(cvals[bestIndex])}: {str(strTest[bestIndex])}")