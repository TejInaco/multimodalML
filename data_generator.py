import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array

class MultimodalDataGenerator(keras.utils.Sequence) :
    def __init__(self, images, tabular, labels, batch_size, target_size, directory) :
        self.images = images
        self.tabular = tabular
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.directory = directory

    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_files = self.images.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_image = np.array(
            [ img_to_array(load_img(self.directory + file, target_size=self.target_size)) for file in batch_files ],
            dtype=np.uint8
        )
        batch_tabular = self.tabular.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_y = self.labels.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        return [batch_image, batch_tabular], batch_y


class TextMultimodalDataGenerator(keras.utils.Sequence) :
    def __init__(self, images, tabular, text, labels, batch_size, target_size, directory) :
        self.images = images
        self.tabular = tabular
        self.text = text
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.directory = directory

    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_files = self.images.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_image = np.array(
            [ img_to_array(load_img(self.directory + file, target_size=self.target_size)) for file in batch_files ],
            dtype=np.uint8
        )
        batch_tabular = self.tabular.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_text = self.text.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_y = self.labels.iloc[idx * self.batch_size : (idx+1) * self.batch_size]

        return [batch_image, batch_tabular, batch_text], batch_y