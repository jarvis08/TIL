import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

w = tf.Variable(1)
w.assign_add(1)
print(w)