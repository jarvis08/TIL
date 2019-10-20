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



######################################################
# Base Logger
# train_his = model.fit(train_images, train_labels, epochs=5, callbacks=None)
# print(train_his.history)



######################################################
# Early Stopping
# callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3)
# train_his = model.fit(train_images, train_labels, epochs=5, callbacks=[callback])



######################################################
# Reduce LearningRate on Plateau
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='acc', verbose=1, factor=0.2, 
#     patience=5, min_lr=0.001
# )
# model.fit(train_images, train_labels, epochs=500, callbacks=[reduce_lr])



######################################################
# Model Check Point
# moodelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=weights.{epoch:02d}-{val_loss:.2f}.hdf5
# )

# checkPoint = tf.keras.callbacks.ModelCheckpoint(
#         filepath='./checkpoints/fashionMNIST.epoch5.h5',
#         monitor='acc',
#         save_best_only=True
# )
# model.fit(train_images, train_labels, epochs=5, callbacks=[checkPoint])



######################################################
# TensorBoard
# tensorboard = tf.keras.callbacks.TensorBoard(
#     log_dir='./tensorboard'
# )
# model.fit(train_images, train_labels, epochs=5, callbacks=[tensorboard])



######################################################
# Lambda Callback
# batch_print = tf.keras.callbacks.LambdaCallback(
#     on_batch_begin=lambda batch,logs: print(batch)
# )

# import json
# json_log = open('loss_log.json', mode='wt', buffering=1)
# json_logging = tf.keras.callbacks.LambdaCallback(
#     on_epoch_end=lambda epoch, logs: json_log.write(
#         json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
#     on_train_end=lambda logs: json_log.close()
# )
# model.fit(train_images, train_labels, epochs=2, callbacks=[batch_print, json_logging])



######################################################
# Lambda Callback
# import datetime


# class CustomCallback(tf.keras.callbacks.Callback):

#   def on_train_batch_begin(self, batch, logs=None):
#     print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

#   def on_train_batch_end(self, batch, logs=None):
#     print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

#   def on_test_batch_begin(self, batch, logs=None):
#     print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

#   def on_test_batch_end(self, batch, logs=None):
#     print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


# model.fit(train_images, train_labels, epochs=3, callbacks=[CustomCallback()])



######################################################
# Terminate On NaN
# model.fit(train_images, train_labels, epochs=5,
#     callbacks=[tf.keras.callbacks.TerminateOnNaN])



######################################################
# Remote Monitor
# remoteMonitor = tf.keras.callbacks.RemoteMonitor(
#     root='http://localhost:9000', path='/publish/epoch/end/', 
#     field='data', headers=None, send_as_json=False
# )



######################################################
# Progress Bar Logger
# progbar = tf.keras.callbacks.ProgbarLogger(count_mode='steps')
# model.fit(train_images, train_labels, epochs=5, callbacks=[progbar])



######################################################
# CSV Logger
csv_logger = tf.keras.callbacks.CSVLogger('csv.log')
csv_logger2 = tf.keras.callbacks.CSVLogger('csv.csv')
model.fit(train_images, train_labels, epochs=5, callbacks=[csv_logger, csv_logger2])



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test accuracy :', test_acc)