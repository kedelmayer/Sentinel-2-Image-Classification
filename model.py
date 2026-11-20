import json, os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
     
# Plots the training and validation loss or accuracy of a model.fit history object.
#   ax: The axis to plot on.
#   output: A model.fit history object containing a model's training and validation loss by epoch.
#   metric: A string containing the metric to plot. May be "Loss" or "Accuracy". Default "Loss".
def plot_output(ax, output, metric="Loss"):
    if metric == "Loss":
        ax.plot(output.history['loss'], label="Training Loss", color="#747575", linewidth=2, linestyle='-')
        ax.plot(output.history['val_loss'], label="Validation Loss", color="#c70c1f", linewidth=2, linestyle='-')
        ax.legend(loc='upper right', fontsize='large')
    if metric == "Accuracy":
        ax.plot(output.history['accuracy'], label="Training Accuracy", color="#747575", linewidth=2, linestyle='-')
        ax.plot(output.history['val_accuracy'], label="Validation Accuracy", color="#c70c1f", linewidth=2, linestyle='-')
        ax.legend(loc='lower right', fontsize='large')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)

## ------------------------------------------------------------------------------------------------ ##
with open("config.JSON", "r") as f:                                                     # Loads data from config.json
    config = json.load(f)

num_classes = len(config['Lands'])
num_epochs = config['epochs']

# Prepare training dataset.
train_ds = tf.keras.utils.image_dataset_from_directory(
  'EuroSAT_RGB',                                                                        # Image directory
  class_names=config['Lands'],                                                          # Classes to select from
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(64, 64),
  batch_size=64)

# Prepare validation dataset.
val_ds = tf.keras.utils.image_dataset_from_directory(
  'EuroSAT_RGB',                                                                        # Image directory
  class_names=config['Lands'],                                                          # Classes to select from
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(64, 64),
  batch_size=64)

# Initialize the model.
model = Sequential([
    Rescaling(1./255),                                                                  # Standardize RGB values to range [0,1]
    Conv2D(8, 3, activation='relu'),                                                    # Convolutional layer with 8 filters of size 3x3
    MaxPooling2D(pool_size=2),                                                          # Max pooling layer with pool size 2x2
    Conv2D(8, 3, activation='relu'),                                                    # Convolutional layer with 8 filters of size 3x3
    MaxPooling2D(pool_size=2),                                                          # Max pooling layer with pool size 2x2
    Conv2D(8, 3, activation='relu'),                                                    # Convolutional layer with 8 filters of size 3x3
    MaxPooling2D(pool_size=2),                                                          # Max pooling layer with pool size 2x2
    Flatten(),                                                                          # Flatten the layer
    Dense(128, activation='relu'),                                                      # Fully connected hidden layer with 128 neurons
    Dense(64, activation='relu'),                                                       # Fully connected hidden layer with 64 neurons
    Dense(16, activation='relu'),                                                       # Fully connected hidden layer with 16 neurons
    Dense(num_classes, activation='softmax'),                                           # Output layer with softmax activation
])

# Compile the model using Adam optimizer and categorical cross-entropy loss.
model.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])

# Fit the model.
print(f"Training Model")
output1 = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

# Plot training results and report the final training and validation accuracies and losses.
fig, axs = plt.subplots(2,1,figsize=(16, 6))
fig.suptitle('Training Result Comparison', fontsize=16)
plot_output(axs[0], output1, "Accuracy")
plot_output(axs[1], output1, "Loss")
plt.tight_layout()
plt.show()

# Show final training and validation accuracy and loss.
table = [[output1.history['accuracy'][num_epochs-1], output1.history['val_accuracy'][num_epochs-1], 
          output1.history['loss'][num_epochs-1], output1.history['val_loss'][num_epochs-1]]]
df = pd.DataFrame(table, columns =['Final Training Accuracy', 'Final Validation Accuracy', 'Final Training Loss', 'Final Validation Loss'])
print(df)


