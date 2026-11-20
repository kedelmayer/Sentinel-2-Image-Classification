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
     
# Plots the training and validation accuracies of a model.fit history object.
#   ax: The axis to plot on.
#   output: A model.fit history object containing a model's training and validation accuracies by epoch.
def plot_accuracy(ax, output):
    ax.plot(output.history['accuracy'], label="Training Accuracy", color="#747575", linewidth=2, linestyle='-')
    ax.plot(output.history['val_accuracy'], label="Validation Accuracy", color="#c70c1f", linewidth=2, linestyle='-')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='lower right', fontsize='large')

# Plots the training and validation loss of a model.fit history object.
#   ax: The axis to plot on.
#   output: A model.fit history object containing a model's training and validation loss by epoch.
def plot_loss(ax, output):
    ax.plot(output.history['loss'], label="Training Loss", color="#747575", linewidth=2, linestyle='-')
    ax.plot(output.history['val_loss'], label="Validation Loss", color="#c70c1f", linewidth=2, linestyle='-')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='lower right', fontsize='large')

## ------------------------------------------------------------------------------------------------ ##
with open("config.JSON", "r") as f:                                                     # Loads data from config.json
    config = json.load(f)

names = config['Lands']
num_classes = len(config['Lands'])
num_epochs = config['epochs']

# Prepare training dataset.
train_ds = tf.keras.utils.image_dataset_from_directory(
  'EuroSAT_RGB',                                                                        # Image directory
  class_names=names,                                                                    # Classes to select from
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(64, 64),
  batch_size=64)

# Prepare validation dataset.
val_ds = tf.keras.utils.image_dataset_from_directory(
  'EuroSAT_RGB',                                                                        # Image directory
  class_names=names,                                                                    # Classes to select from
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(64, 64),
  batch_size=64)

# Initialize the model.
model1 = Sequential([
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

model2 = Sequential([
    Rescaling(1./255),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

model3 = Sequential([
    Rescaling(1./255),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

# Compile the models using Adam optimizer and categorical cross-entropy loss.
model1.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])
model2.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])
model3.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])

# Fit the models.
print(f"Training Model 1...")
output1 = model1.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
print(f"Training Model 2...")
output2 = model2.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
print(f"Training Model 3...")
output3 = model3.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
print()

# Plot training results and report the final training and validation accuracies.
fig, (axs) = plt.subplots(2,3,figsize=(16, 6))
fig.suptitle('Training Result Comparison', fontsize=16)
axs[0,0].set_title('Model 1')
axs[0,1].set_title('Model 2')
axs[0,2].set_title('Model 3')

plot_accuracy(axs[0,0], output1)
plot_accuracy(axs[0,1], output2)
plot_accuracy(axs[0,2], output3)
plot_loss(axs[1,0], output1)
plot_loss(axs[1,1], output2)
plot_loss(axs[1,2], output3)
plt.tight_layout()
plt.show()

# Show final training and validation accuracy comparisons.
table = [[output1.history['accuracy'][num_epochs-1], output1.history['val_accuracy'][num_epochs-1], 
          output1.history['loss'][num_epochs-1], output1.history['val_loss'][num_epochs-1]],
         [output2.history['accuracy'][num_epochs-1], output2.history['val_accuracy'][num_epochs-1], 
          output2.history['loss'][num_epochs-1], output2.history['val_loss'][num_epochs-1]], 
         [output3.history['accuracy'][num_epochs-1], output3.history['val_accuracy'][num_epochs-1], 
          output3.history['loss'][num_epochs-1], output3.history['val_loss'][num_epochs-1]]]
df = pd.DataFrame(table, columns =['Final Training Accuracy', 'Final Validation Accuracy', 'Final Training Loss', 'Final Validation Loss'])
print(df)


