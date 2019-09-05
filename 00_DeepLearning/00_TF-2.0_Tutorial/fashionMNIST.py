from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
# import matplotlib.pyplot as plt
# print(tf.__version__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
print(train_images.shape)
print(train_labels)
print(train_labels.shape)
 
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir='./tensorboard'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/fashionMNIST.h5',
        monitor='accuracy',
        # save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=1,
    )
    # tf.keras.callbacks.BaseLogger.on_train_begin()

]

model.fit(train_images, train_labels, epochs=5, callbacks=callbacks)

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test accuracy :', test_acc)
