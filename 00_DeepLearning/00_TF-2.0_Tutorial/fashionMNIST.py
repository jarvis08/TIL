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

# # Base Logger
# train_his = model.fit(train_images, train_labels, epochs=5, callbacks=None)
# print(train_his.history)

# # Early Stopping
# callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3)
# train_his = model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

# # Reduce LearningRate on Plateau
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='acc', verbose=1, factor=0.2, patience=5, min_lr=0.001)
# model.fit(train_images, train_labels, epochs=500, callbacks=[reduce_lr])



# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test accuracy :', test_acc)



# callbacks = [
#     tf.keras.callbacks.TensorBoard(
#         log_dir='./tensorboard'
#     ),
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath='./checkpoints/fashionMNIST.h5',
#         monitor='val_loss',
#         # save_best_only=True
#     ),
#     tf.keras.callbacks.EarlyStopping(
#         monitor='accuracy',
#         patience=1,
#     )
# ]