# data_preprocessing.py

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_cifar100_data(batch_size=64):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    train_datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_gen = val_datagen.flow(x_test, y_test, batch_size=batch_size)

    return train_gen, val_gen