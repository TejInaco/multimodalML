import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from math import ceil

BATCH_SIZE = 32
NUM_CLASSES = 5
DROPOUT_PROB = 0.2
DATASET_PATH = "data/"

print("DATASET_PATH content")
print(os.listdir(DATASET_PATH))

# Read CSV file
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=200, error_bad_lines=False)
df['usage'] = df['usage'].astype('str')
df = df.sample(frac=1).reset_index(drop=True)

print("Head styles.csv")
print(df.head(10))



X = df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year']]
y = df['usage']

X = pd.get_dummies(X)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Attributes")
print(X)

print("Labels")
print(y)

model = Sequential()
model.add(Dense(12, input_dim=len(X.columns), activation='relu'))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(8, activation='relu'))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print ("Final Model summary")
model.summary()


model.fit(
    X_train,
    y_train,
    steps_per_epoch = ceil(len(X_train) / BATCH_SIZE),

    validation_data=(X_test, y_test),
    validation_steps= ceil(len(X_test) / BATCH_SIZE),
    
    epochs=1,
    verbose=1
)

model.save('weights/tabular_model.h5')