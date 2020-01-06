import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from math import ceil


BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 5
DROPOUT_PROB = 0.2
DATASET_PATH = "data/"

print("DATASET_PATH content")
print(os.listdir(DATASET_PATH))

# Read CSV file
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=20000, error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df['usage'] = df['usage'].astype('str')
df['season'] = df['season'].astype('str')
df = df.sample(frac=1).reset_index(drop=True)

print("Head styles.csv")
print(df.head(10))


# Image classification
image_generator = ImageDataGenerator(validation_split = 0.2)

training_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory=DATASET_PATH + "images",
    x_col="image",
    y_col="season",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

validation_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory=DATASET_PATH + "images",
    x_col="image",
    y_col="season",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

#Load the VGG model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model.layers:
    layer.trainable = False

print ("Base Model summary")
print(base_model.summary())

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_PROB)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(DROPOUT_PROB)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print ("Final Model summary")
model.summary()


# Fit model
model.fit_generator(
    generator=training_generator,
    steps_per_epoch=ceil(training_generator.samples / BATCH_SIZE),

    validation_data=validation_generator,
    validation_steps=ceil(validation_generator.samples / BATCH_SIZE),
    
    epochs=2,
    verbose=1
)

model.save('weights/image_model.h5')