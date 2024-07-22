import numpy as np
import cv2
import os
from tqdm import tqdm as tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from keras import backend as K
import segmentation_models as sm
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import generic_utils
from segmentation_models.losses import CategoricalFocalLoss
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
# from keras.optimizers import Adam,SGD
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from glob import glob
from segmentation_models.metrics import  f1_score
from tensorflow.keras.callbacks import TensorBoard
import datetime
from segmentation_models.metrics import  f1_score,precision,recall
from segmentation_models.metrics import  f1_score
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import time


#main utility functions for model
# def dsc(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return score
""" Here we have dice loss we can use when needed"""
def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

""" Here we have bce loss and we can use when needed"""
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

""" Here we have iou_score metrices we can use when needed"""
def iou_score(gt, pr, class_weights=1., smooth=1, per_image=True, threshold=None):
    '''
    input：
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), 
        if ``None`` prediction prediction will not be round
    output：
        IoU/Jaccard score in range [0, 1]
    '''
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]
        
    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    intersection = tf.reduce_sum(gt * pr, axis=axes)
    union = tf.reduce_sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = tf.reduce_mean(iou, axis=0)

    # weighted mean per class
    iou = tf.reduce_mean(iou * class_weights)
    return iou

"""importing libraries"""


import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import MovingAverageQuantizer, LastValueQuantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from keras.layers import Layer


""" here we are defining the ConfigQuantzefig to quantize the perticular layer inside the model """
# Custom QuantizeConfig for Conv2D layers
class ConvQuantizeConfig(QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False))]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []

    def get_config(self):
        return {}
    

""" here we are registering the quanization """
tf.keras.utils.register_keras_serializable(package='QuantizeConfig')(ConvQuantizeConfig)


#build model
import segmentation_models as sm
from keras.layers import *
from keras import layers
from keras.models import Model


""" Simpliy defining the model architecture"""

def SEModule(input, ratio, out_dim):
    # bs, c, h, w
    x = GlobalAveragePooling2D()(input)
    excitation = Dense(units=out_dim // ratio)(x)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)
    scale = multiply([input, excitation])
    return scale


# def SEUnet(nClasses, input_height=224, input_width=224):
#     inputs = Input(shape=(input_height, input_width, 3))
#     conv1 = Conv2D(16,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(inputs)
#     conv1 = GroupNormalization(groups=8, axis=-1)(conv1)

#     conv1 = Conv2D(16,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv1)
#     conv1 = GroupNormalization(groups=8, axis=-1)(conv1)

#     # se
#     conv1 = SEModule(conv1, 4, 16)

#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(32,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool1)
#     conv2 = GroupNormalization(groups=8, axis=-1)(conv2)

#     conv2 = Conv2D(32,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv2)
#     conv2 = GroupNormalization(groups=8, axis=-1)(conv2)

#     # se
#     conv2 = SEModule(conv2, 8, 32)

#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(64,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool2)
#     conv3 = GroupNormalization(groups=8, axis=-1)(conv3)

#     conv3 = Conv2D(64,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv3)
#     conv3 = GroupNormalization(groups=8, axis=-1)(conv3)

#     # se
#     conv3 = SEModule(conv3, 8, 64)

#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(128,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool3)
#     conv4 = GroupNormalization(groups=8, axis=-1)(conv4)

#     conv4 = Conv2D(128,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv4)
#     conv4 = GroupNormalization(groups=8, axis=-1)(conv4)

#     # se
#     conv4 = SEModule(conv4, 16, 128)

#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(256,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool4)
#     conv5 = GroupNormalization(groups=8, axis=-1)(conv5)
#     conv5 = Conv2D(256,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv5)
#     conv5 = GroupNormalization(groups=8, axis=-1)(conv5)

#     # se
#     conv5 = SEModule(conv5, 16, 256)

#     up6 = Conv2D(128,
#                  2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2,
#                                                                     2))(conv5))
#     up6 = GroupNormalization(groups=8, axis=-1)(up6)

#     merge6 = concatenate([conv4, up6], axis=3)
#     conv6 = Conv2D(128,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge6)
#     conv6 = GroupNormalization(groups=8, axis=-1)(conv6)

#     conv6 = Conv2D(128,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv6)
#     conv6 = GroupNormalization(groups=8, axis=-1)(conv6)

#     # se
#     conv6 = SEModule(conv6, 16, 128)

#     up7 = Conv2D(64,
#                  2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2,
#                                                                     2))(conv6))
#     up7 = GroupNormalization(groups=8, axis=-1)(up7)

#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(64,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge7)
#     conv7 = GroupNormalization(groups=8, axis=-1)(conv7)

#     conv7 = Conv2D(64,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv7)
#     conv7 = GroupNormalization(groups=8, axis=-1)(conv7)

#     # se
#     conv7 = SEModule(conv7, 8, 64)

#     up8 = Conv2D(32,
#                  2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2,
#                                                                     2))(conv7))
#     up8 = GroupNormalization(groups=8, axis=-1)(up8)

#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(32,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge8)
#     conv8 = GroupNormalization(groups=8, axis=-1)(conv8)

#     conv8 = Conv2D(32,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv8)
#     conv8 = GroupNormalization(groups=8, axis=-1)(conv8)

#     # se
#     conv8 = SEModule(conv8, 4, 32)

#     up9 = Conv2D(16,
#                  2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2,
#                                                                     2))(conv8))
#     up9 = GroupNormalization(groups=8, axis=-1)(up9)

#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(16,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge9)
#     conv9 = GroupNormalization(groups=8, axis=-1)(conv9)

#     conv9 = Conv2D(16,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv9 = GroupNormalization(groups=8, axis=-1)(conv9)

#     # se
#     conv9 = SEModule(conv9, 2, 16)

#     conv10 = Conv2D(nClasses, (3, 3), padding='same')(conv9)
#     conv10 = GroupNormalization(groups=8, axis=-1)(conv10)

#     outputHeight = Model(inputs, conv10).output_shape[1]
#     outputWidth = Model(inputs, conv10).output_shape[2]

#     out = (Reshape((outputHeight * outputWidth, nClasses)))(conv10)
#     out = Activation('softmax')(out)

#     model = Model(inputs,out)
#     model.outputHeight = outputHeight
#     model.outputWidth = outputWidth

#     return model

""" defining the model architecture by making use of quantization annoate layer and here we are quantizing the specific
layer by annotating the perticular layer to 8-bit """

def SEUnet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    
    conv1 = quantize_annotate_layer(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(inputs)
    
    conv1 = BatchNormalization()(conv1)
    conv1 = quantize_annotate_layer(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv1)
    conv1 = BatchNormalization()(conv1)
    # SE
    conv1 = SEModule(conv1, 4, 16)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = quantize_annotate_layer(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = quantize_annotate_layer(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv2)
    conv2 = BatchNormalization()(conv2)
    
    # SE
    conv2 = SEModule(conv2, 8, 32)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = quantize_annotate_layer(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = quantize_annotate_layer(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv3)
    conv3 = BatchNormalization()(conv3)
    
    # SE
    conv3 = SEModule(conv3, 8, 64)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = quantize_annotate_layer(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = quantize_annotate_layer(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv4)
    conv4 = BatchNormalization()(conv4)
    
    # SE
    conv4 = SEModule(conv4, 16, 128)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = quantize_annotate_layer(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = quantize_annotate_layer(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # SE
    conv5 = SEModule(conv5, 16, 256)

    up6 = quantize_annotate_layer(
        Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(UpSampling2D(size=(2, 2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = quantize_annotate_layer(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = quantize_annotate_layer(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv6)
    conv6 = BatchNormalization()(conv6)
    
    # SE
    conv6 = SEModule(conv6, 16, 128)

    up7 = quantize_annotate_layer(
        Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = quantize_annotate_layer(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = quantize_annotate_layer(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # SE
    conv7 = SEModule(conv7, 8, 64)

    up8 = quantize_annotate_layer(
        Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = quantize_annotate_layer(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = quantize_annotate_layer(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv8)
    conv8 = BatchNormalization()(conv8)
    
    # SE
    conv8 = SEModule(conv8, 4, 32)

    up9 = quantize_annotate_layer(
        Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = quantize_annotate_layer(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = quantize_annotate_layer(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        quantize_config=ConvQuantizeConfig()
    )(conv9)
    conv9 = BatchNormalization()(conv9)
    
    # SE
    conv9 = SEModule(conv9, 2, 16)

    conv10 = quantize_annotate_layer(
        Conv2D(nClasses, (3, 3), padding='same'),
        quantize_config=ConvQuantizeConfig()
    )(conv9)
    conv10 = BatchNormalization()(conv10)

    outputHeight = Model(inputs, conv10).output_shape[1]
    outputWidth = Model(inputs, conv10).output_shape[2]

    out = Reshape((outputHeight * outputWidth, nClasses))(conv10)
    out = Activation('softmax')(out)

    model = Model(inputs, out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model

""" here we are defining the model output layer"""
def my_model():
    model1 = SEUnet(nClasses=5)

    x = model1.get_layer(index=-3).output

    # folder=["Sprout","Black_smut","Rotten","Background"]
    out0 = Conv2D(2, (1, 1), activation='softmax',name='sprout')(x)
    out1 = Conv2D(2, (1, 1), activation='softmax',name='peeled')(x)
    out2 = Conv2D(2, (1, 1), activation='softmax',name='rotten')(x)
    out3 = Conv2D(2, (1, 1), activation='softmax',name='black_smut')(x)
    out4 = Conv2D(2, (1, 1), activation='softmax',name='background')(x)
  


    model_new = Model(inputs = model1.input,outputs = [out0,out1,out2,out3,out4])
    #print(model_new.summary())
    
    quantized_model = tfmot.quantization.keras.quantize_apply(model_new) ### applying quantization by calling the finction


    model_name = 'onin_10th_jul_spplrobg'
    date = '10thjuly2024'
    learning_rate = 0.001
 

    opt = Adam(lr=learning_rate)

    """ here we are defining the confusion metrices and loss finctions """

        # Define losses, metrics, and loss weights
    losses = {"sprout": sm.losses.bce_jaccard_loss,"peeled": sm.losses.bce_jaccard_loss,"rotten": sm.losses.bce_jaccard_loss, "black_smut": sm.losses.bce_jaccard_loss,"background": sm.losses.bce_jaccard_loss}
    metrics = {"sprout": [iou_score, f1_score,precision,recall],"peeled": [iou_score, f1_score,precision,recall],"rotten": [iou_score, f1_score,precision,recall],"black_smut": [iou_score, f1_score,precision,recall], "background": [iou_score, f1_score,precision,recall]}
    loss_weights = {"sprout": 1.0,"peeled": 1.0,"rotten":1.0, "black_smut": 0.8, "background": 3.0}

    # LR_Reduce_callback = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, mode='auto')
    # model_ckpt1 = ModelCheckpoint(model_name+'_'+date+'.h5', monitor='loss', save_weights_only=False,save_best_only=True, period=1) ## WEIGHTS WITH MODEL
    # model_ckpt2 = ModelCheckpoint(model_name+'_'+date+'_weights.h5', monitor='loss', save_weights_only=True,save_best_only=True, period=1) ##ONLY WEIGHTS

    quantized_model.compile(optimizer=opt, loss=losses, metrics=metrics) ## 

    return quantized_model


""" These 2 functions will be giving usique name to the model and saving the model """
def get_unique_filename(file_name):
    unique_file_name = time.strftime(f"%Y%m%d_%H%M%S_{file_name}")
    return unique_file_name

def save_model(model,model_name,model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir,unique_filename)
    model.save(path_to_model)
    











