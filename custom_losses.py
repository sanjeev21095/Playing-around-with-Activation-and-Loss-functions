from __future__ import absolute_import
import six
from . import backend as K
from .utils.generic_utils import deserialize_keras_object


def stock_loss(y_true, y_pred):
    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0), \
        alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), \
        K.abs(y_true - y_pred)
        )
    return K.mean(loss, axis=-1) 

def mse_log(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(K.log(y_true) - K.log(y_pred))
    return K.mean(loss, axis=-1)

def negative_log_likelihood(y_true, y_pred):
    return -1*K.mean(K.log(y_pred), axis=-1)

def tanimoto_loss(y_true, y_pred):
    val1 = -1*K.sum((K.sigmoid(y_pred))*y_true)
    val2 = K.l2_normalize(y_true, axis=-1) + K.l2_normalize(y_pred, axis=-1) + K.sum(y_pred*y_true)
    return val1/val2

def cubed_hinge_loss(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.))*K.maximum(1. - y_true * y_pred, 0.), axis=-1)

   

def serialize(loss):
    return loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
