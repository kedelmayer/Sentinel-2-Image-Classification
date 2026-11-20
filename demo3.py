import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset, Features, Sequence, Value
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Prints a summary of an input transaction dataset.
#   items: A pandas dataframe representation of the full transaction list. Contains a header.
#   item_counts: An array of tuples of the format (item, count). Sorted in descending order.
def print_transaction_summary(items, item_counts):
    print("\n=================== TRANSACTION SUMMARY ====================")
    print(f"Number of transactions:             {len(items)-1}")
    print(f"Number of unique items:             {len(item_counts)}")
    print(f"Most purchased items:")
    for (item, count) in item_counts[:3]:
        print(f"    {item}:  {count}")
    print("==============================================================\n")

# Plots the training and validation accuracies of a model.fit history object.
#   ax: The axis to plot on.
#   output: A model.fit history object containing a model's training and validation accuracies by epoch.
def plot_output(ax, output):
    ax.plot(output.history['accuracy'], label="Training Accuracy", color="#747575", linewidth=2, linestyle='-')
    ax.plot(output.history['val_accuracy'], label="Validation Accuracy", color="#c70c1f", linewidth=2, linestyle='-')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='lower right', fontsize='large')

# Encodes a dataset for evaluation by the Bert model
#   examples: The text data to be encoded.
#   return: The encoded dataset.
def preprocess_data(examples):
  text = examples["text"]
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  labels_matrix = np.zeros((len(text), len(labels)), dtype=np.float32)
  for idx, label_ids in enumerate(examples["labels"]):
    for label_id in label_ids:
      labels_matrix[idx, label_id] = 1.0

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

# Computes multilabel metrics for a test evaluation.
#   predictions: The model's output data.
#   labels: The true answers for the model's data.
#   threshold: The delineator for when a model's response should be counted as a prediction. Default 0.5.
#   return: The computed F1-score, ROC AUC, and accuracy.
def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()                                                        # Apply sigmoid on predictions (shape (batch_size, num_labels))
    probs = sigmoid(torch.Tensor(predictions))                                          # Use threshold to convert to integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels                                                                     # Compute metrics
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,                                                  # Return as dictionary
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

# Parses the predictions for a model's evaluation output.
#   p: The model's predictions.
#   return: The computed F1-score, ROC AUC, and accuracy.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


## ------------------------------------------------------------------------------------------------ ##
## Association Rule Mining ##
# Summarize the transaction dataset.
items = pd.read_csv("text/Grocery_Items.csv")                                           # Read in transaction dataset
item_counts = []                                                                        # Initialize array for compiling item counts
for i, row in items.iterrows():
    for col in row:
        if isinstance(col, str):
            try:
                [idx] = [i for i, val in enumerate(item_counts) if val[0] == col]       # Find index of item in item_counts
                item_counts[idx] = (col, (item_counts[idx][1]+1))                       # Increment count
            except:
                item_counts.append((col, 1))                                            # Add new item to count array if it can't be found
item_counts.sort(key=lambda tup: tup[1], reverse=True)                                  # Sort list by most-bought items
print_transaction_summary(items, item_counts)                                           # Print summary

# Extract frequent itemsets with minimum support of 0.005.
transactions = [
    [str(item) for item in row if pd.notna(item) and str(item).strip()]
    for row in items.to_numpy()
]
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions, sparse=True)                       
items_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)          # Encode the data as a one-hot encoded pandas DataFrame
frequent_itemsets = apriori(items_encoded, min_support=0.005, use_colnames=True)        # Extract frequent itemsets with minimum support of 0.005

# Generate association rules using a minimum confidence threshold of 0.10.
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)    # Generate association rules with a minimum confidence of 0.10

# Display the total number of rules and the five rules with the highest lift.
rules = rules.sort_values('lift', ascending=False)
print(f"Total number of Association Rules: {len(rules)}")                               # Print total number of rules
print(rules[['support', 'confidence', 'lift']].head(5))                                 # Display rules

# Analyze parameter sensitivity.
FI_1 = apriori(items_encoded, min_support=0.002, use_colnames=True)                     # Frequent itemsets with minimum support of 0.002
FI_2 = apriori(items_encoded, min_support=0.005, use_colnames=True)                     # Frequent itemsets with minimum support of 0.005
FI_3 = apriori(items_encoded, min_support=0.01, use_colnames=True)                      # Frequent itemsets with minimum support of 0.01

r_11 = association_rules(FI_1, metric="confidence", min_threshold=0.05)                 # min support = 0.002, min confidence = 0.05
r_21 = association_rules(FI_2, metric="confidence", min_threshold=0.05)                 # min support = 0.005, min confidence = 0.05
r_31 = association_rules(FI_3, metric="confidence", min_threshold=0.05)                 # min support = 0.01, min confidence = 0.05
r_12 = association_rules(FI_1, metric="confidence", min_threshold=0.1)                  # min support = 0.002, min confidence = 0.1
r_22 = association_rules(FI_2, metric="confidence", min_threshold=0.1)                  # min support = 0.005, min confidence = 0.1
r_32 = association_rules(FI_3, metric="confidence", min_threshold=0.1)                  # min support = 0.01, min confidence = 0.1
r_13 = association_rules(FI_1, metric="confidence", min_threshold=0.15)                 # min support = 0.002, min confidence = 0.15
r_23 = association_rules(FI_2, metric="confidence", min_threshold=0.15)                 # min support = 0.005, min confidence = 0.15
r_33 = association_rules(FI_3, metric="confidence", min_threshold=0.15)                 # min support = 0.01, min confidence = 0.15

data = np.array([[len(r_11),len(r_21),len(r_31)],[len(r_12),len(r_22),len(r_32)],[len(r_13),len(r_23),len(r_33)]])
rules = pd.DataFrame(data, index=[0.05,0.1,0.15], columns=[0.002,0.005,0.01])

ax = sns.heatmap(rules, annot=True)                                                     # Plot number of association rules on heatmap                         
ax.set(xlabel="Minimum Support", ylabel="Minimum Confidence")
ax.set_title("Association Rules by Minimum Support and Minimum Confidence")
plt.tight_layout()
plt.show()

## ------------------------------------------------------------------------------------------------ ##
## Image Classification using CNN ##
# Prepare training dataset.
train_ds = tf.keras.utils.image_dataset_from_directory(
  'EuroSAT_RGB',                                                                        # Image directory
  class_names=["AnnualCrop", "Forest", "Residential", "River"],                         # Classes to select from
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(64, 64),
  batch_size=64)

# Prepare validation dataset.
val_ds = tf.keras.utils.image_dataset_from_directory(
  'EuroSAT_RGB',                                                                        # Image directory
  class_names=["AnnualCrop", "Forest", "Residential", "River"],                         # Classes to select from
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
    Flatten(),                                                                          # Flatten the layer
    Dense(16, activation='relu'),                                                       # Fully connected hidden layer with 16 neurons
    Dense(4, activation='softmax'),                                                     # Output layer with 4 neurons and softmax activation
])

# Compile the model using Adam optimizer and categorical cross-entropy loss.
model1.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])
num_epochs = 20                                                                         # Train for 20 epochs

# Fit the model.
print(f"Training Model 1...")
output1 = model1.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

# Plot training results and report the final training and validation accuracies.
fig, ax = plt.subplots()
plot_output(ax, output1)
ax.set_title('Model Accuracy by Epoch', fontsize=16)
plt.show()

print(f"Model 1 Final Accuracies:")
print(f"Final training accuracy: {output1.history['accuracy'][num_epochs-1]}")          # Output final training accuracy
print(f"Final validation accuracy: {output1.history['val_accuracy'][num_epochs-1]}")    # Output final validation accuracy
print()

# Change the number of filters in the second convolutional layer to 4 and 16.
model2 = Sequential([
    Rescaling(1./255),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(4, 3, activation='relu'),                                                    # Convolutional layer with 4 filters of size 3x3
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax'),
])

model3 = Sequential([
    Rescaling(1./255),
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(16, 3, activation='relu'),                                                   # Convolutional layer with 16 filters of size 3x3
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax'),
])

# Compile the models using Adam optimizer and categorical cross-entropy loss.
model2.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])
model3.compile('adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])

# Fit the models.
print(f"Training Model 2...")
output2 = model2.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
print()
print(f"Training Model 3...")
output3 = model3.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
print()

# Plot training results and report the final training and validation accuracies.
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16, 6))
fig.suptitle('Training Result Comparison: Number of Filters in Convolutional Layer 2', fontsize=16)
ax1.set_title('Model Accuracy with 4 filters')
ax2.set_title('Model Accuracy with 16 filters')

plot_output(ax1, output2)
plot_output(ax2, output3)
plt.tight_layout()
plt.show()

# Show final training and validation accuracy comparisons.
table = [[4, output2.history['accuracy'][num_epochs-1], output2.history['val_accuracy'][num_epochs-1]],
         [8, output1.history['accuracy'][num_epochs-1], output1.history['val_accuracy'][num_epochs-1]], 
         [16, output3.history['accuracy'][num_epochs-1], output3.history['val_accuracy'][num_epochs-1]]]
df = pd.DataFrame(table, columns =['Filters in Second Convolutional Layer', 'Final Training Accuracy', 'Final Validation Accuracy'])
print(df)

## ------------------------------------------------------------------------------------------------ ##
## Multi-Label Text Classification Using BERT ##

dataset = load_dataset('json', data_files={'train': 'text/s_train.JSON',                # Load simplified text dataset
            'validation': 'text/s_validation.JSON', 'test': 'text/s_test.JSON'})

labels = ['anger','anticipation','disgust','fear','joy','love','optimism',
          'pessimism','sadness','surprise','trust']
id2label = {idx: label for idx, label in enumerate(labels)}                             # Create id -> label map
label2id = {label: idx for idx, label in enumerate(labels)}                             # Create label -> id map

# Use autotokenizer to preprocess the text.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")                          # Initialize tokenizer

new_features = Features({                                                               # Define new features with labels as floats
    'input_ids': Sequence(Value('int64')),
    'token_type_ids': Sequence(Value('int64')),
    'attention_mask': Sequence(Value('int64')),
    'labels': Sequence(Value('float32'))
})

encoded_dataset = dataset.map(preprocess_data, batched=True,                            # Encode the dataset using the new features
                              remove_columns=dataset['train'].column_names, 
                              features=new_features)
encoded_dataset.set_format("torch")                                                     # Set data format to PyTorch Tensor


# Define and train the model.
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                problem_type="multi_label_classification",num_labels=len(labels),
                id2label=id2label, label2id=label2id)

batch_size = 8
metric_name = "f1"

args = TrainingArguments(
    f"bert-finetuned-go-emotions",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,                                                                  # Fine-tune the model for 5 epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()                                                                          # Train the model

# Plot the training and validation loss curves for all epochs.
trainLosses = []
evalLosses = []
log_history = trainer.state.log_history
for i in log_history:                                                                    # Compile training and validation loss from log history
    if 'loss' in i: trainLosses.append(i['loss'])
    if 'eval_loss' in i: evalLosses.append(i['eval_loss'])

fig, ax = plt.subplots()                                                                 # Plot training and validation loss
ax.set_title('Loss by Epoch', fontsize=16)
ax.plot(trainLosses, label="Training Loss", color="#747575", linewidth=2, linestyle='-')
ax.plot(evalLosses, label="Validation Loss", color="#c70c1f", linewidth=2, linestyle='-')
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.legend(loc='lower right', fontsize='large')
plt.show()

# Make model predictions using the test set to assess model accuracy.
preds = trainer.predict(encoded_dataset["test"])                                         # Make predictions on the test set
reals = encoded_dataset["test"]['labels']

def label_match(pred, real):
    all = True
    any = False
    for j in range(0,10):
        if pred[j] != real[j]: all = False                                               # If labels don't match, exact match is false
        if pred[j] == 1.0 and real[j] == 1.0: any = True                                 # If both labels are true, any match is true
    return all, any

count = 0
e_accurate = 0
o_accurate = 0
preds_converted = []
sigmoid = torch.nn.Sigmoid()
for i in range(0,len(preds.predictions)):
    probs = sigmoid(torch.from_numpy(preds.predictions[i]))                              # Turn every score into a number between 0 and 1
    p = np.zeros(probs.shape)
    p[np.where(probs >= 0.5)] = 1                                                        # Convert probability to a prediction at a threshold of 0.5
    preds_converted.append(p)
    r = reals[i].numpy()                                                                 # Convert encoded test dataset tensor to array

    all, any = label_match(p, r)                                                         # Determine whether prediction is an exact o at-least-one match
    if all: e_accurate = e_accurate + 1
    if any: o_accurate = o_accurate + 1
    count = count + 1

f1_micro_average = f1_score(y_true=encoded_dataset["test"]['labels'], y_pred=preds_converted, average='micro')

print(f"Exact Match Test Accuracy:              {e_accurate/(count)}")                   # Display exact match test accuracy
print(f"At-Least-One-Match Test Accuracy:       {o_accurate/(count)}")                   # Display at-least-one match test accuracy
print(f"Micro-Averaged F1 Score for Test Set:   {f1_micro_average}")                     # Display micro-averaged F1 score for test set
