# -*- coding: utf-8 -*-
from __future__ import absolute_import
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras import backend as K
import tensorflow as tf
from keras.legacy import interfaces
import numpy as np

class cust_act (Layer):
#Intializing constants if any
    def __init__(self, alpha=1, **kwargs):
        super(cust_act, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
#Define the function and include the return parameters
    def call(self, inputs):
       return tf.exp(-1*inputs*inputs) #gaussian
       #return tf.log(1+tf.exp(inputs)) # softplus
       #return (K.sigmoid(alpha*inputs) * inputs) #swish

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(cust_act, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

