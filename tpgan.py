from keras import backend as K
from keras.layers import Input, Add, Maximum, Dense, Activation, BatchNormalization, Conv2D, Conv2DTranspose, Reshape, Flatten, Concatenate, Lambda, MaxPooling2D, ZeroPadding2D, Dropout, AveragePooling2D, Average ,GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard, Callback
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.initializers import Constant, RandomNormal, TruncatedNormal, Zeros
from keras import losses
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import re
import cv2
from PIL import Image
from tensorflow.python.ops.image_ops import resize_images
from keras.preprocessing.image import array_to_img
from functools import partial


import face_alignment
import multipie_gen
from numba import jit, vectorize, prange
import multiprocessing
import time


import face_alignment

import multipie_gen

tf.logging.set_verbosity(tf.logging.WARN) # record warnning message

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

EYE_H, EYE_W = 40, 40
NOSE_H, NOSE_W = 32, 40
MOUTH_H, MOUTH_W = 32, 48

class TPGAN():

    def __init__(self,
                 base_filters=64, gpus=1,
                 facenet_weights='',
                 generator_weights='',
                 classifier_weights='',
                 discriminator_weights=''):
        """ 
        initialize TP-GAN network with given weights file. if weights file is None, the weights are initialized by default initializer.
        
        Args:
            base_filters (int): base filters count of TP-GAN. default 64.
            gpus (int): number of gpus to use.
            generator_weights (str): trained generator weights file path. it is used to resume training. not required when train from scratch.
            classifier_weights (str): trained classifier weights file path. it is used to resume training. not required when training from scratch.
            discriminator_weights (str): trained discriminator weights file path. it is used to resume training. not required when training from scratch.
        """

        
        self.gpus = gpus
        self.base_filters = base_filters
        self.generator_weights = generator_weights
        self.discriminator_weights = discriminator_weights
        self.classifier_weights = classifier_weights
        

        self._discriminator = None
        self._generator = None
        self._classifier = None
        self._parts_rotator = None  #進入臉部
        
        self.generator_train_model = None
        self.discriminator_train_model = None
        
        self.gen_current_epochs = self.current_epoch_from_weights_file(self.generator_weights)
        self.disc_current_epochs = self.current_epoch_from_weights_file(self.discriminator_weights)
        self.class_epochs = self.current_epoch_from_weights_file(self.classifier_weights)
        
        self.facenet = FaceNet(extractor_weights=facenet_weights) 

            
    def current_epoch_from_weights_file(self, weights_file):

        if weights_file is not None:
            try:
                ret_epochs = int(re.match(r'.*epoch([0-9]+).*.hdf5', weights_file).groups()[0])
            except:
                ret_epochs = 0
        else:
            ret_epochs = 0
            
        return ret_epochs
                
    def discriminator(self):
        """ 
        getter of singleton discriminator
        """
        
        if self._discriminator is None:
           self._discriminator = self.build_discriminator(base_filters=self.base_filters)
           #self._discriminator = load_model(self.discriminator_weights)
           #if self.discriminator_weights is not None:
           #self._discriminator.load_weights(self.discriminator_weights)
            
        return self._discriminator

    def generator(self):
        """ 
        getter of singleton generator
        """
        
        if self._generator is None:
            self._generator = self.build_generator(base_filters=self.base_filters)
            #self._generator = load_model(self.generator_weights,custom_objects= {'CloudableModel':CloudableModel,'tf':tf,'multipie_gen':multipie_gen})
            
        return self._generator
    
    def classifier(self):
        """ 
        getter of singleton classifier
        """
        
        if self._classifier is None:
            self._classifier = self.build_classifier()
           #self._classifier = load_model(self.classifier_weights,custom_objects= {'CloudableModel':CloudableModel,'tf':tf,'multipie_gen':multipie_gen})
        return self._classifier
    
    def parts_rotator(self):
        """ 
        getter of singleton part rotator for each part; left eye, right eye, nose, and mouth.
        """
        
        if self._parts_rotator is None:
            self._parts_rotator = self.build_parts_rotator(base_filters=self.base_filters)
                        
        return self._parts_rotator
    
    def _add_activation(self, X, func='relu'):
        """
        private func to add activation layer
        """

        if func is None:
            return X
        elif func == 'relu':
            return Activation('relu')(X)
        elif func == 'lrelu':
            return LeakyReLU()(X)
        else:
            raise Exception('Undefined function for activation: ' + func)

    def _res_block(self, X, kernel_size, batch_norm=False, activation=None, name=None):
        """
        private func to add residual block
        """
        
        X_shortcut = X
        
        if batch_norm:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name+'_c1_0', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(X)
            X = BatchNormalization(epsilon=1e-5, name=name+'_c1_0_bn')(X)
        else:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name+'_c1_1', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(X)
        self._add_activation(X, activation)
        
        if batch_norm:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False, name=name+'_c2_0', kernel_initializer=TruncatedNormal(stddev=0.02))(X)
            X = BatchNormalization(epsilon=1e-5, name=name+'_c2_0_bn')(X)
        else:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name+'_c2_1', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(X)

        self._add_activation(X, activation)
            
        X = Add()([X_shortcut, X])
        
        return X
    
    def build_generator(self, name="generator", base_filters=64):

        '''
        def combine_parts(size_hw, leye, reye, nose, mouth):
            
            img_h, img_w = size_hw
        
            leye_img = ZeroPadding2D(padding=((int(multipie_gen.LEYE_Y - multipie_gen.EYE_H/2), img_h - int(multipie_gen.LEYE_Y + multipie_gen.EYE_H/2)),
            (int(multipie_gen.LEYE_X - multipie_gen.EYE_W/2), img_w - int(multipie_gen.LEYE_X + multipie_gen.EYE_W/2))))(leye)
            
            reye_img = ZeroPadding2D(padding=((int(multipie_gen.REYE_Y - multipie_gen.EYE_H/2), img_h - int(multipie_gen.REYE_Y + multipie_gen.EYE_H/2)),
            (int(multipie_gen.REYE_X - multipie_gen.EYE_W/2), img_w - int(multipie_gen.REYE_X + multipie_gen.EYE_W/2))))(reye)
    
            nose_img = ZeroPadding2D(padding=((int(multipie_gen.NOSE_Y - multipie_gen.NOSE_H/2), img_h - int(multipie_gen.NOSE_Y + multipie_gen.NOSE_H/2)),
            (int(multipie_gen.NOSE_X - multipie_gen.NOSE_W/2), img_w - int(multipie_gen.NOSE_X + multipie_gen.NOSE_W/2))))(nose)
    
            mouth_img = ZeroPadding2D(padding=((int(multipie_gen.MOUTH_Y - multipie_gen.MOUTH_H/2), img_h - int(multipie_gen.MOUTH_Y + multipie_gen.MOUTH_H/2)),
            (int(multipie_gen.MOUTH_X - multipie_gen.MOUTH_W/2), img_w - int(multipie_gen.MOUTH_X + multipie_gen.MOUTH_W/2))))(mouth)
    
            return Maximum()([leye_img, reye_img, nose_img, mouth_img])
        '''
            
           
        def combine_parts(size_hw, leye, reye, nose, mouth, d128r_img):
            img_h, img_w = size_hw
            lm_img = d128r_img,cv2.COLOR_BGR2RGB
            
            try:
                landmark = fa.get_landmarks(cv2.cvtColor(lm_img))
            
                reye_points = landmark[36:42]
                reye_center = (np.max(reye_points, axis=0) + np.min(reye_points, axis=0)) / 2
                reye_left = int(reye_center[0] - EYE_W / 2 + 0.5)
                reye_up = int(reye_center[1] - EYE_H / 2 + 0.5)
                
                reye_img = ZeroPadding2D(padding=((int(reye_up - EYE_H/2), img_h - int(reye_up + EYE_H/2)),
                (int(reye_left - EYE_W/2), img_w - int(reye_left + EYE_W/2))))(reye)
                
                leye_points = landmark[42:48]
                leye_center = (np.max(leye_points, axis=0) + np.min(leye_points, axis=0)) / 2
                leye_left = int(leye_center[0] - EYE_W / 2 + 0.5)
                leye_up = int(leye_center[1] - EYE_H / 2 + 0.5)
                
                leye_img = ZeroPadding2D(padding=((int(leye_up - EYE_H/2), img_h - int(leye_up + EYE_H/2)),
                (int(leye_left - EYE_W/2), img_w - int(leye_left + EYE_W/2))))(leye)
                
                nose_points = landmark[31:36]
                nose_center = (np.max(nose_points, axis=0) + np.min(nose_points, axis=0)) / 2
                nose_left = int(nose_center[0] - NOSE_W / 2 + 0.5)
                nose_up = int(nose_center[1] - 10 - NOSE_H / 2 + 0.5)
    
                nose_img = ZeroPadding2D(padding=((int(nose_up -NOSE_H/2), img_h - int(nose_up + NOSE_H/2)),
                (int(nose_left - NOSE_W/2), img_w - int(nose_left + NOSE_W/2))))(nose)
           
                mouth_points = landmark[48:60]
                mouth_center = (np.max(mouth_points, axis=0) + np.min(mouth_points, axis=0)) / 2
                mouth_left = int(mouth_center[0] - MOUTH_W / 2 + 0.5)
                mouth_up = int(mouth_center[1] - MOUTH_H / 2 + 0.5)
    
                mouth_img = ZeroPadding2D(padding=((int(mouth_up - MOUTH_H/2), img_h - int(mouth_up + MOUTH_H/2)),
                (int(mouth_left - MOUTH_W/2), img_w - int(mouth_left + MOUTH_W/2))))(mouth)
                
                img = Maximum()([leye_img, reye_img, nose_img, mouth_img])
                return  img
            except:
                img_h, img_w = size_hw
            
                leye_img = ZeroPadding2D(padding=((int(multipie_gen.LEYE_Y - multipie_gen.EYE_H/2), img_h - int(multipie_gen.LEYE_Y + multipie_gen.EYE_H/2)),
                (int(multipie_gen.LEYE_X - multipie_gen.EYE_W/2), img_w - int(multipie_gen.LEYE_X + multipie_gen.EYE_W/2))))(leye)
                
                reye_img = ZeroPadding2D(padding=((int(multipie_gen.REYE_Y - multipie_gen.EYE_H/2), img_h - int(multipie_gen.REYE_Y + multipie_gen.EYE_H/2)),
                (int(multipie_gen.REYE_X - multipie_gen.EYE_W/2), img_w - int(multipie_gen.REYE_X + multipie_gen.EYE_W/2))))(reye)
        
                nose_img = ZeroPadding2D(padding=((int(multipie_gen.NOSE_Y - multipie_gen.NOSE_H/2), img_h - int(multipie_gen.NOSE_Y + multipie_gen.NOSE_H/2)),
                (int(multipie_gen.NOSE_X - multipie_gen.NOSE_W/2), img_w - int(multipie_gen.NOSE_X + multipie_gen.NOSE_W/2))))(nose)
        
                mouth_img = ZeroPadding2D(padding=((int(multipie_gen.MOUTH_Y - multipie_gen.MOUTH_H/2), img_h - int(multipie_gen.MOUTH_Y + multipie_gen.MOUTH_H/2)),
                (int(multipie_gen.MOUTH_X - multipie_gen.MOUTH_W/2), img_w - int(multipie_gen.MOUTH_X + multipie_gen.MOUTH_W/2))))(mouth)
        
                return Maximum()([leye_img, reye_img, nose_img, mouth_img])
         
        
        full_name = name
        # shorten name
        name = name[0]
    
        in_img = Input(shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 3))
        #mc_in_img128 = Concatenate()([in_img, Lambda(lambda x: x[:,:,::-1,:])(in_img)]) #[::-1]將字串或陣列倒序排列
        mc_in_img128 = in_img #[::-1]將字串或陣列倒序排列

        mc_in_img64 = Lambda(lambda x: tf.image.resize_bilinear(x, [multipie_gen.IMG_H//2, multipie_gen.IMG_W//2]))(mc_in_img128)
        mc_in_img32 = Lambda(lambda x: tf.image.resize_bilinear(x, [multipie_gen.IMG_H//4, multipie_gen.IMG_W//4]))(mc_in_img64)
        
        
        c128 = Conv2D(base_filters, (7, 7), padding='same', strides=(1, 1), name=name+'_c128', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(mc_in_img128)
        c128 = self._add_activation(c128, 'lrelu')
        c128r = self._res_block(c128, (7, 7), batch_norm=True, activation='lrelu', name=name+'_c128_r')
        
        c64 = Conv2D(base_filters, (5, 5), padding='same', strides=(2, 2), name=name+'_c64', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c128r)
        c64 = BatchNormalization(epsilon=1e-5, name=name+'_c64_bn')(c64)
        c64 = self._add_activation(c64, 'lrelu')
        c64r = self._res_block(c64, (5, 5), batch_norm=True, activation='lrelu', name=name+'_c64_r')
        
        c32 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_c32', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c64r)
        c32 = BatchNormalization(epsilon=1e-5, name=name+'_c32_bn')(c32)
        c32 = self._add_activation(c32, 'lrelu')
        c32r = self._res_block(c32, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c32_r')
        
        c16 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_c16', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c32r)
        c16 = BatchNormalization(epsilon=1e-5, name=name+'_c16_bn')(c16)
        c16 = self._add_activation(c16, 'lrelu') 
        c16r = self._res_block(c16, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c16_r')
        
        c8 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c8', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c16r)
        c8 = BatchNormalization(epsilon=1e-5, name=name+'_c8_bn')(c8)
        c8 = self._add_activation(c8, 'lrelu')
        
        c8r = self._res_block(c8, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r')     
        c8r2 = self._res_block(c8r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r2')     
        c8r3 = self._res_block(c8r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r3')     
        c8r4 = self._res_block(c8r3, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r4')     

        fc1 = Dense(512, name=name+'_fc1', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.005))(Flatten()(c8r4))
        fc2 = Maximum()([Lambda(lambda x: x[:, :256])(fc1), Lambda(lambda x: x[:, 256:])(fc1)])
        
        in_noise = Input(shape=(100,))
        fc2_with_noise = Concatenate()([fc2, in_noise])
        
        fc3 = Dense(8*8*base_filters, name=name+'_fc3', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Constant(0.1))(fc2_with_noise)
        
        f8 = Conv2DTranspose(base_filters, (8, 8), padding='valid', strides=(1, 1), name=name+'_f8', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(Reshape((1, 1, fc3.shape[-1].value))(fc3))
        f32 = Conv2DTranspose(base_filters//2, (3, 3), padding='same', strides=(4, 4), name=name+'_f32', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(f8)
        f64 = Conv2DTranspose(base_filters//4, (3, 3), padding='same', strides=(2, 2), name=name+'_f64', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(f32)
        f128 = Conv2DTranspose(base_filters//8, (3, 3), padding='same', strides=(2, 2), name=name+'_f128', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(f64)
           
        # size8
        d8 = Concatenate(name=name+'_d8')([c8r4, f8])
        d8r = self._res_block(d8, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d8_r')
        d8r2 = self._res_block(d8r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d8_r2')
        d8r3 = self._res_block(d8r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d8_r3')
      
        # size16
        d16 = Conv2DTranspose(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_d16', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d8r3)
        d16 = BatchNormalization(epsilon=1e-5, name=name+'_d16_bn')(d16)
        d16 = self._add_activation(d16, 'relu')
        d16r = self._res_block(c16r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d16_r')
        d16r2 = self._res_block(Concatenate()([d16, d16r]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d16_r2')
        d16r3 = self._res_block(d16r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d16_r3')
        
        # size32
        d32 = Conv2DTranspose(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_d32', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d16r3)
        d32 = BatchNormalization(epsilon=1e-5, name=name+'_d32_bn')(d32)
        d32 = self._add_activation(d32, 'relu')
        d32r = self._res_block(Concatenate()([c32r, mc_in_img32, f32]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d32_r')
        d32r2 = self._res_block(Concatenate()([d32, d32r]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d32_r2')
        d32r3 = self._res_block(d32r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d32_r3')
        img32 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_img32', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d32r3)
        
        # size64
        d64 = Conv2DTranspose(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_d64', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d32r3)
        d64 = BatchNormalization(epsilon=1e-5, name=name+'_d64_bn')(d64)
        d64 = self._add_activation(d64, 'relu')
        d64r = self._res_block(Concatenate()([c64r, mc_in_img64, f64]), (5, 5), batch_norm=True, activation='lrelu', name=name+'_d64_r')
        
        interpolated64 = Lambda(lambda x: tf.image.resize_bilinear(x, [64, 64]))(img32) # Use Lambda layer to wrap tensorflow func, resize_bilinear
        
        d64r2 = self._res_block(Concatenate()([d64, d64r, interpolated64]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d64_r2')
        d64r3 = self._res_block(d64r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d64_r3')
        img64 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_img64', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d64r3)
                
        # size128
        d128 = Conv2DTranspose(base_filters, (3, 3), padding='same', strides=(2, 2), name=name+'_d128', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d64r3)
        d128 = BatchNormalization(epsilon=1e-5, name=name+'_d128_bn')(d128)
        d128 = self._add_activation(d128, 'relu')
        d128r = self._res_block(Concatenate()([c128r, mc_in_img128, f128]), (5, 5), batch_norm=True, activation='lrelu', name=name+'_d128_r')

        interpolated128 = Lambda(lambda x: tf.image.resize_bilinear(x, [128, 128]))(img64) # Use Lambda layer to wrap tensorflow func, resize_bilinear
        
        
        in_leye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_reye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_nose = Input(shape=(multipie_gen.NOSE_H, multipie_gen.NOSE_W, 3))
        in_mouth = Input(shape=(multipie_gen.MOUTH_H, multipie_gen.MOUTH_W, 3))
        
        front_leye_img, front_leye_feat, front_reye_img, front_reye_feat, front_nose_img, front_nose_feat, front_mouth_img, front_mouth_feat\
        = self.parts_rotator()([in_leye, in_reye, in_nose, in_mouth])
        
        d128r_img =  Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_d128r_img', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d128r)
        
        combined_parts_img = combine_parts([128, 128], front_leye_img, front_reye_img, front_nose_img, front_mouth_img, d128r_img)
        combined_parts_feat = combine_parts([128, 128], front_leye_feat, front_reye_feat, front_nose_feat, front_mouth_feat,d128r_img)
        '''
        combined_parts_img = combine_parts([128, 128], front_leye_img, front_reye_img, front_nose_img, front_mouth_img)
        combined_parts_feat = combine_parts([128, 128], front_leye_feat, front_reye_feat, front_nose_feat, front_mouth_feat)
        '''
        d128r2 = self._res_block(Concatenate()([d128, d128r, interpolated128,combined_parts_feat, combined_parts_img]), (3, 3), batch_norm=False, activation='lrelu', name=name+'_d128_r2')
        d128r2c = Conv2D(base_filters, (5, 5), padding='same', strides=(1, 1), name=name+'_d128_r2c', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(d128r2)
        d128r2c = BatchNormalization(epsilon=1e-5, name=name+'_d128_r2c_bn')(d128r2c)
        d128r2c = self._add_activation(d128r2c, 'lrelu')
        d128r3c = self._res_block(d128r2c, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d128_r3c')
        d128r3c2 = Conv2D(base_filters//2, (3, 3), padding='same', strides=(1, 1), name=name+'_d128_r3c2', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(d128r3c)
        d128r3c2 = BatchNormalization(epsilon=1e-5, name=name+'_d128_r3c2_bn')(d128r3c2)
        d128r3c2 = self._add_activation(d128r3c2, 'lrelu')
        img128 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_img128', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d128r3c2)
        ret_model = CloudableModel(inputs=[in_img, in_leye, in_reye, in_nose, in_mouth, in_noise], outputs=[img128, img64, img32, fc2, front_leye_img, front_reye_img, front_nose_img, front_mouth_img], name=full_name)          
      
        return ret_model
    
    def build_classifier(self, name='classifier'):
        """
        build classifier model.
        """
        
        full_name = name
        # shorten name
        name = name[0]
        
        in_feat = Input(shape=(256,))
        X = Dropout(0.7)(in_feat)
        clas = Dense(multipie_gen.NUM_SUBJECTS, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=regularizers.l2(0.005),
                     use_bias=False, name=name+'_dense')(X)
        
        ret_classifier = CloudableModel(inputs=in_feat, outputs=clas, name=full_name)
        
        #ret_classifier.summary()
        return ret_classifier    
                    
    def build_train_generator_model(self):
        """
        build train model for generator.
        this model wraps generator and classifier, adds interface for loss functions.
        """
        
        in_img = Input(shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 3))
        in_leye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_reye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_nose = Input(shape=(multipie_gen.NOSE_H, multipie_gen.NOSE_W, 3))
        in_mouth = Input(shape=(multipie_gen.MOUTH_H, multipie_gen.MOUTH_W, 3))
        in_noise = Input(shape=(100,))
        
        img128, img64, img32, fc2, front_leye_img, front_reye_img, front_nose_img, front_mouth_img\
        = self.generator()([in_img, in_leye, in_reye, in_nose, in_mouth, in_noise]) 
        
        subject_id = self.classifier()(fc2)
                
        # add name label to connect with each loss functions
        img128_px = Lambda(lambda x:x, name = "00img128px")(img128)
        img128_sym = Lambda(lambda x:x, name = "01img128sym")(img128)
        img128_ip = Lambda(lambda x:x, name = "02ip")(img128)
        img128_adv = Lambda(lambda x:x, name = "03adv")(img128)
        img128_tv = Lambda(lambda x:x, name = "04tv")(img128)
        img64_px = Lambda(lambda x:x, name = "05img64px")(img64)
        img64_sym = Lambda(lambda x:x, name = "06img64sym")(img64)
        img32_px = Lambda(lambda x:x, name = "07img32px")(img32)
        img32_sym = Lambda(lambda x:x, name = "08img32sym")(img32)
        subject_id = Lambda(lambda x:x, name = "09classify")(subject_id)
        leye = Lambda(lambda x:x, name = "10leye")(front_leye_img)
        reye = Lambda(lambda x:x, name = "11reye")(front_reye_img)
        nose = Lambda(lambda x:x, name = "12nose")(front_nose_img)
        mouth = Lambda(lambda x:x, name = "13mouth")(front_mouth_img)
        
        ret_model = CloudableModel(inputs=[in_img, in_leye, in_reye, in_nose, in_mouth, in_noise],
                          outputs=[img128_px, img128_sym, img128_ip, img128_adv, img128_tv, img64_px, img64_sym, img32_px, img32_sym, subject_id, leye, reye, nose, mouth],
                          name='train_genarator_model')
        #ret_model.summary()
        
        return ret_model
    
    def build_parts_rotator(self, base_filters=64):
        """
        build models for all each part rotator.
        """       
        
        leye_rotator = self.build_part_rotator('leye', base_filters=base_filters, in_h=multipie_gen.EYE_H , in_w=multipie_gen.EYE_W)
        reye_rotator = self.build_part_rotator('reye', base_filters=base_filters, in_h=multipie_gen.EYE_H , in_w=multipie_gen.EYE_W)
        nose_rotator = self.build_part_rotator('nose', base_filters=base_filters, in_h=multipie_gen.NOSE_H , in_w=multipie_gen.NOSE_W)
        mouth_rotator = self.build_part_rotator('mouth', base_filters=base_filters, in_h=multipie_gen.MOUTH_H , in_w=multipie_gen.MOUTH_W)
            
        in_leye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_reye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_nose = Input(shape=(multipie_gen.NOSE_H, multipie_gen.NOSE_W, 3))
        in_mouth = Input(shape=(multipie_gen.MOUTH_H, multipie_gen.MOUTH_W, 3))
        
        out_leye_img, out_leye_feat = leye_rotator(in_leye)
        out_reye_img, out_reye_feat = reye_rotator(in_reye)
        out_nose_img, out_nose_feat = nose_rotator(in_nose)
        out_mouth_img, out_mouth_feat = mouth_rotator(in_mouth)
        
        ret_model = CloudableModel(inputs=[in_leye, in_reye, in_nose, in_mouth],
                            outputs=[out_leye_img, out_leye_feat, out_reye_img, out_reye_feat, out_nose_img, out_nose_feat, out_mouth_img, out_mouth_feat], name='parts_rotator')
        #ret_model.summary()
        
        return ret_model
    
    def build_part_rotator(self, name, in_h, in_w, base_filters=64):
        """
        build model for one part rotator.
        """   
        
        in_img = Input(shape=(in_h, in_w, 3))
        
        c0 = Conv2D(base_filters, (3, 3), padding='same', strides=(1, 1), name=name+'_c0', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(in_img)
        c0r = self._res_block(c0, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c0_r')
        c1 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_c1', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c0r)
        c1 = BatchNormalization(name=name+'_c1_bn')(c1)
        c1 = self._add_activation(c1, 'lrelu')
        
        c1r = self._res_block(c1, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c1_r')
        c2 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_c2', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c1r)
        c2 = BatchNormalization(name=name+'_c2_bn')(c2)
        c2 = self._add_activation(c2, 'lrelu')
        
        c2r = self._res_block(c2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c2_r')
        c3 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c3', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c2r)
        c3 = BatchNormalization(name=name+'_c3_bn')(c3)
        c3 = self._add_activation(c3, 'lrelu')
        
        c3r = self._res_block(c3, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c3_r')
        c3r2 = self._res_block(c3r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c3_r2')
        
        d1 = Conv2DTranspose(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_d1', use_bias=True, kernel_initializer=RandomNormal(stddev=0.02))(c3r2)
        d1 = BatchNormalization(name=name+'_d1_bn')(d1)
        d1 = self._add_activation(d1, 'lrelu')
        
        after_select_d1 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(1, 1), name=name+'_asd1', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(Concatenate()([d1, c2r]))
        after_select_d1 = BatchNormalization(name=name+'_asd1_bn')(after_select_d1)
        after_select_d1 = self._add_activation(after_select_d1, 'lrelu')
        d1r = self._res_block(after_select_d1, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d1_r')
        d2 = Conv2DTranspose(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_d2', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d1r)
        d2 = BatchNormalization(name=name+'_d2_bn')(d2)
        d2 = self._add_activation(d2, 'lrelu')
        
        after_select_d2 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(1, 1), name=name+'_asd2', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(Concatenate()([d2, c1r]))
        after_select_d2 = BatchNormalization(name=name+'_asd2_bn')(after_select_d2)
        after_select_d2 = self._add_activation(after_select_d2, 'lrelu')
        d2r = self._res_block(after_select_d2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d2_r')
        d3 = Conv2DTranspose(base_filters, (3, 3), padding='same', strides=(2, 2), name=name+'_d3', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d2r)
        d3 = BatchNormalization(name=name+'_d3_bn')(d3)
        d3 = self._add_activation(d3, 'lrelu')
        
        after_select_d3 = Conv2D(base_filters, (3, 3), padding='same', strides=(1, 1), name=name+'_asd3', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(Concatenate()([d3, c0r]))
        after_select_d3 = BatchNormalization(name=name+'_asd3_bn')(after_select_d3)
        after_select_d3 = self._add_activation(after_select_d3, 'lrelu')
        part_feat = self._res_block(after_select_d3, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d3_r')
        
        part_img = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_c4', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(part_feat)

        ret_model = CloudableModel(inputs=[in_img], outputs=[part_img, part_feat], name= name + '_rotator')
        
        #ret_model.summary()
        
        return ret_model
    
    def build_discriminator(self, name='discriminator', base_filters=64):
        """
        build model for discriminator.
        """   
        
        full_name = name
        # shorten name
        name = name[0]
        
        in_img = Input(shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 3))
        
        c64 = Conv2D(base_filters, (3, 3), padding='same', strides=(2, 2), name=name+'_c64', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(in_img)
        c64 = self._add_activation(c64, 'lrelu')
        
        c32 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_c32', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c64)
        c32 = BatchNormalization(center=True, scale=True, name=name+'_c32_bn')(c32)
        c32 = self._add_activation(c32, 'lrelu')
        
        c16 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_c16', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c32)
        c16 = BatchNormalization(center=True, scale=True, name=name+'_c16_bn')(c16)
        c16 = self._add_activation(c16, 'lrelu')
        
        c8 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c8', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c16)
        c8 = BatchNormalization(center=True, scale=True, name=name+'_c8_bn')(c8)
        c8 = self._add_activation(c8, 'lrelu')
        c8r = self._res_block(c8, (3, 3), batch_norm=False, activation='lrelu', name=name+'_c8_r')
        
        c4 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c4', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c8r)
        c4 = BatchNormalization(center=True, scale=True, name=name+'_c4_bn')(c4)
        c4 = self._add_activation(c4, 'lrelu')
        c4r = self._res_block(c4, (3, 3), batch_norm=False, activation='lrelu', name=name+'_c4_r')
        
        Layer_0 = MaxPooling2D(pool_size=(2, 2),padding='same') (c4r)
        Layer_1 = LeakyReLU(alpha=0.1) (Layer_0)
        Layer_2 = Dense(256, activation='relu') (Layer_1)
        Layer_3 = Dense(128, activation='relu') (Layer_2)
        Layer_4 = Dense(64, activation='relu') (Layer_3)
        
        
        #feat = Conv2D(1, (1, 1), padding='same', strides=(1, 1), name=name+'_c4_r_c', activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(c4r)
        #ret_model = CloudableModel(inputs=in_img, outputs=feat, name=full_name)
        
        
        ret_model = CloudableModel(inputs=in_img, outputs=Layer_4, name=full_name)
        #ret_model.summary()
        
        return ret_model
        
    class SaveWeightsCallback(Callback):
    
        def __init__(self, target_models, out_dir, period):
            """
            Args:
                target_models (list): list of save target models
                out_dir (str): output dir
                period (int): save interval epochs
            """
            self.target_models = target_models
            self.out_dir = out_dir
            self.period = period
    
        def on_epoch_end(self, epoch, logs):
            if (epoch + 1) % self.period == 0:
                for target_model in self.target_models:
                    out_model_dir = '{}{}/'.format(self.out_dir, target_model.name)
                    tf.gfile.MakeDirs(out_model_dir)

                    target_model.save(out_model_dir + 'epoch{epoch:04d}_loss{loss:.3f}.hdf5'.format(epoch=epoch + 1, loss=logs['loss']), overwrite=True)
    
    def train_gan(self, gen_datagen_creator, gen_train_batch_size, gen_valid_batch_size,
                  disc_datagen_creator, disc_batch_size, disc_gt_shape,
                  optimizer,
                  gen_steps_per_epoch=300, disc_steps_per_epoch=10, epochs=100, 
                  out_dir='../out/', out_period=5, is_output_img=False,
                  lr=0.001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1.5,
                  lambda_sym=3e-1, lambda_ip=1e1, lambda_adv=2e1, lambda_tv=1e-3,
                  lambda_class=4e-1, lambda_parts=3):
        
        """
        train both generator and discriminator as GAN.
        
        Args:
            gen_datagen_creator (func): function for create generator datagen
            gen_train_batch_size (int): batch size for training generator
            gen_valid_batch_size (int): batch size for validate generator
            disc_datagen_creator (func): function for create discriminator datagen
            disc_batch_size (int): batch size for training/validate discriminator
            disc_gt_shape (int): dicriminator output size    
            optimizer (Optimizer): keras optimizer for training generator. (currently, training discriminator always use Adam)
            gen_steps_per_epoch (int): steps per epoch for training generator
            disc_steps_per_epoch (int): steps per epoch for training discriminator
            epochs (int): train epochs
            out_dir (str): out_dir for weights, logs, sample images
            out_period (int): output interval epochs.
            is_output_img (bool): if True, output sample images each out period epochs.
            lr (float): learning rate for training generator optimizer
            lambda_128 (func): func returns the coefficient of 128px image output
            lambda_64 (func): func returns the coefficient of 64px image output
            lambda_32 (func): func returns the coefficient of 32px image output
            lambda_sym (func): func returns the coefficient of symmetricity loss
            lambda_ip (func): func returns the coefficient of identity preserve loss
            lambda_adv (func): func returns the coefficient of adversarial loss
            lambda_tv (func): func returns the coefficient of total variation loss
            lambda_class (func): func returns the coefficient of classification loss
            lambda_parts (func): func returns the coefficient of part img loss
        """
        for i in range(epochs):
            print('train generator {}/{}'.format(i+1, epochs))

            lambda_128_i = lambda_128 
            lambda_64_i = lambda_64
            lambda_32_i = lambda_32
            lambda_sym_i = lambda_sym
            lambda_ip_i = lambda_ip
            lambda_adv_i =lambda_adv
            lambda_tv_i = lambda_tv
            lambda_class_i = lambda_class
            lambda_parts_i=lambda_parts
            print('params for this epoch\n\
                  lambda_128:{}\n\
                  lambda_64:{}\n\
                  lambda_32:{}\n\
                  lambda_sym:{}\n\
                  lambda_ip:{}\n\
                  lambda_adv:{}\n\
                  lambda_tv:{}\n\
                  lambda_class:{}\n\
                  lambda_parts:{}\n'.format(
                  lambda_128_i, 
                  lambda_64_i, 
                  lambda_32_i,
                  lambda_sym_i, 
                  lambda_ip_i, 
                  lambda_adv_i, 
                  lambda_tv_i, 
                  lambda_class_i,
                  lambda_parts_i))
        
            gen_train_datagen = gen_datagen_creator(batch_size=gen_train_batch_size, setting='train')
            gen_valid_datagen = gen_datagen_creator(batch_size=gen_valid_batch_size, setting='valid')
            g_history = self.train_generator(train_gen=gen_train_datagen, valid_gen=gen_valid_datagen, optimizer=optimizer, steps_per_epoch=gen_steps_per_epoch,
                                           epochs=1, is_output_img=is_output_img, out_dir=out_dir, out_period=out_period,
                                           lr=lr, lambda_128=lambda_128_i, lambda_64=lambda_64_i, lambda_32=lambda_32_i,
                                           lambda_sym=lambda_sym_i, lambda_ip=lambda_ip_i, lambda_adv=lambda_adv_i, lambda_tv=lambda_tv_i,
                                           lambda_class=lambda_class_i, lambda_parts=lambda_parts_i)
            print('epoch:{} generator model trained. loss is as follows.\n{}'.format(self.gen_current_epochs, g_history.history['loss']))


            print('train discriminator {}/{}'.format(i+1, epochs))
            disc_datagen = disc_datagen_creator(self.generator(), batch_size=disc_batch_size, setting='train', gt_shape=disc_gt_shape)
            d_history = self.train_discriminator(train_gen=disc_datagen, valid_gen=disc_datagen, steps_per_epoch=disc_steps_per_epoch,
                                               epochs=1, out_dir=out_dir, out_period=out_period)
            print('epoch:{} discriminator model trained. binary_accuracy is as follows.\n{}'.format(self.disc_current_epochs, d_history.history['binary_accuracy']))


            
            
            
    def train_generator(self, train_gen, valid_gen, optimizer=Adam(lr=0.0001), steps_per_epoch=100, epochs=1, is_output_img=False, out_dir='D:/desktop/tpgan_keras/keras_tpgan/out/', out_period=1,
              lr=0.001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1.5,
              lambda_sym=3e-1, lambda_ip=1e1, lambda_adv=2e1, lambda_tv=1e-3,
              lambda_class=4e-1, lambda_parts=3):
        
        #asfasf = 0
        """
        train generator.
        
        Args:
            train_gen (generator): generator which provides mini-batch train data
            valid_gen (generator): generator which provides mini-batch validation data 
            optimizer (Optimizer): keras optimizer for training generator.
            steps_per_epoch (int): steps per epoch for training generator
            epochs (int): train epochs
            is_output_img (bool): if True, output sample images each out period epochs.
            out_dir (str): out_dir for weights, logs, sample images
            out_period (int): output interval epochs.
            lr (float): learning rate for training generator optimizer
            decay (float): learning rate decay for training generator optimizer
            lambda_128 (float): coefficient of 128px image output
            lambda_64 (float): coefficient of 64px image output
            lambda_32 (float): coefficient of 32px image output
            lambda_sym (float): coefficient of symmetricity loss
            lambda_ip (float): coefficient of identity preserve loss
            lambda_adv (float): coefficient of adversarial loss
            lambda_tv (float): coefficient of total variation loss
            lambda_class (float): coefficient of classification loss
            lambda_parts (float): coefficient of part img loss
        """
        
        class SaveSampleImageCallback(Callback):
            """
            this callback save sample images generated by current trained model.
            """
            
            def __init__(self, generator, out_dir, period): 
                """
                Args:
                    generator (generator): generator which provides input profile images
                    out_dir (str): output dir
                    period (int): save interval epochs
                """
                
                self.generator = generator
                self.out_dir = out_dir
                self.period = period
                
                tf.gfile.MakeDirs('{}img128/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}img64/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}img32/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}leye/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}reye/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}nose/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}mouth/'.format(self.out_dir))
        
            def on_epoch_end(self, epoch, logs):
                
                def imsave(path, imarray):
                    
                    imarray[np.where(imarray<0)] = 0
                    imarray[np.where(imarray>1)] = 1
                    image = Image.fromarray((imarray*np.iinfo(np.uint8).max).astype(np.uint8))
                    
                    with open(path, 'wb') as f:
                        image.save(f)

                if (epoch + 1) % self.period == 0:
                    inputs, _ = next(self.generator)
                    img128, _, _, _, _, img64, _, img32, _, subject_id, front_leye_img, front_reye_img, front_nose_img, front_mouth_img = self.model.predict(inputs)
                    for i in range(len(img128)):
                        sub_id = np.argmax(subject_id[i])
                        imsave('{}img128/epoch{:04d}_img128_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), img128[i])
                        imsave('{}img64/epoch{:04d}_img64_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), img64[i])
                        imsave('{}img32/epoch{:04d}_img32_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), img32[i])
                        imsave('{}leye/epoch{:04d}_leye_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), front_leye_img[i])
                        imsave('{}reye/epoch{:04d}_reye_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), front_reye_img[i])
                        imsave('{}nose/epoch{:04d}_nose_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), front_nose_img[i])
                        imsave('{}mouth/epoch{:04d}_mouth_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id+1, logs['loss'], i), front_mouth_img[i])

        def _loss_img128_px(y_true, y_pred):
            """
            pixel loss for size 128x128
            """
            loss_pixel = K.mean(losses.mean_absolute_error(y_true, y_pred))
            return lambda_128 * loss_pixel
        
        def _loss_img128_sym(y_true, y_pred):
            """
            symmetricity loss for size 128x128
            """
            loss_sym = K.mean(losses.mean_absolute_error(y_pred[:,:,::-1,:], y_pred))
            return lambda_128 * lambda_sym * loss_sym
                
        def _loss_ip(y_true, y_pred):
            
            '''
            sess = tf.Session()
            
            # In[]:   tenosr2numpy
            array_true = sess.run(y_true)
            array_pred = sess.run(y_pred)
            # In[]:   numpy2uint8            
            y_true_img=Image.fromarray(array_true)
            y_pred_img=Image.fromarray(array_pred)
            # In[]:   facenet.predict            
            y_true_emb = self.facenet.embeddings(y_true_img)
            y_pred_emb = self.facenet.embeddings(y_pred_img)
            '''
            y_true_img = Lambda(lambda x:resize_images(x, (160, 160)))(y_true)
            y_true_emb = self.facenet.extractor()(y_true_img)

            y_pred_img = Lambda(lambda x:resize_images(x, (160, 160)))(y_pred)
            y_pred_emb = self.facenet.extractor()(y_pred_img)
            

            
            return lambda_ip * (K.mean(K.abs(y_pred_emb - y_true_emb)))
            
        def _loss_adv(y_true, y_pred):
            """
            adversarial loss computed from img128
            """
      
            disc_score = self.discriminator()(y_pred) #disc_score.shape =(4,4) 
            
            return lambda_adv * K.mean(losses.binary_crossentropy(K.ones_like(disc_score)*0.9, disc_score))
        
        def _loss_tv(y_true, y_pred):
            """
            total variation loss computed from img128
            """
            return lambda_tv * K.mean(tf.image.total_variation(y_pred))
        
        def _loss_img64_px(y_true, y_pred):
            """
            pixel loss for size 64x64
            """
            loss_pixel = K.mean(losses.mean_absolute_error(y_true, y_pred))
            return lambda_64 * loss_pixel
        
        def _loss_img64_sym(y_true, y_pred):
            """
            symmetricity loss for size 64x64
            """
            loss_sym = K.mean(losses.mean_absolute_error(y_pred[:,:,::-1,:], y_pred))
            return lambda_64 * lambda_sym * loss_sym
       
        def _loss_img32_px(y_true, y_pred):
            """
            pixel loss for size 32x32
            """
            loss_pixel = K.mean(losses.mean_absolute_error(y_true, y_pred))
            return lambda_32 * loss_pixel
        
        def _loss_img32_sym(y_true, y_pred):
            """
            symmetricity loss for size 32x32
            """
            loss_sym = K.mean(losses.mean_absolute_error(y_pred[:,:,::-1,:], y_pred))
            return lambda_32 * lambda_sym * loss_sym
            
        def _loss_classify(y_true, y_pred):
            """
            classification loss
            """
            return lambda_class * K.mean(losses.categorical_crossentropy(y_true, y_pred))

        def _loss_part(y_true, y_pred):
            """
            rotated part image loss
            """
            return lambda_parts * K.mean(losses.mean_absolute_error(y_true, y_pred))
              
        
        if self.generator_train_model is None:
            self.generator_train_model = self.build_train_generator_model()
            if self.gpus > 1:
                self.generator_train_model = multi_gpu_model(self.generator_train_model, gpus=self.gpus)
            
            # set trainable flag and recompile               
            self.generator().trainable = True
            self.classifier().trainable = True
            self.discriminator().trainable = False
            self.generator_train_model.compile(optimizer=optimizer,
                                loss={'00img128px': _loss_img128_px,
                                      '01img128sym': _loss_img128_sym,
                                      '02ip': _loss_ip,
                                      '03adv': _loss_adv,
                                      '04tv': _loss_tv,
                                      '05img64px': _loss_img64_px,
                                      '06img64sym': _loss_img64_sym,
                                      '07img32px': _loss_img32_px,
                                      '08img32sym': _loss_img32_sym,
                                      '09classify': _loss_classify,
                                      '10leye': _loss_part,
                                      '11reye': _loss_part,
                                      '12nose': _loss_part,
                                      '13mouth': _loss_part},
                                metrics={'02ip':'acc'})
            
        callbacks = []
        callbacks.append(TensorBoard(log_dir=out_dir+'logs/generator/'))
        callbacks.append(self.SaveWeightsCallback(target_models=[self.generator(), self.classifier()], out_dir=out_dir+'weights/', period=out_period))
        if is_output_img:
            callbacks.append(SaveSampleImageCallback(generator=valid_gen, out_dir=out_dir+'images/', period=out_period))
        history = self.generator_train_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs+self.gen_current_epochs,
                                            callbacks=callbacks, workers=0, validation_data=valid_gen, validation_steps=1,
                                            shuffle=False, initial_epoch=self.gen_current_epochs)
        
        #callbacks.append(TensorBoard(log_dir=out_dir+'logs/generator/'))
        #callbacks.append(self.SaveWeightsCallback(target_models=[self.generator(), self.classifier()], out_dir=out_dir+'weights/', period=out_period))
        #self.generator_train_model.SaveWeightsCallback(target_models=[self.generator(), self.classifier()], out_dir=out_dir+'weights/', period=out_period)        
        
        self.gen_current_epochs += epochs
        
        #self.generator().save('generator_train_model.hdf5')#Test
 
        return history
    
    def train_discriminator(self, train_gen, valid_gen, steps_per_epoch=100, epochs=1, out_dir='../out/', out_period=1):
        """
        train discriminator
        
        Args:
            train_gen (generator): generator which provides mini-batch train data
            valid_gen (generator): generator which provides mini-batch validation data 
            steps_per_epoch (int): steps per epoch for training discriminator
            epochs (int): train epochs
            out_dir (str): out_dir for weights, logs, sample images
            out_period (int): output interval epochs.
        """
        
        def loss_disc(y_true, y_pred):
            """
            binary_crossentropy considering one-side label smoothing
            """
            return losses.binary_crossentropy(y_true*0.9, y_pred)
        
        if self.discriminator_train_model is None:
            if self.gpus > 1:
                self.discriminator_train_model = multi_gpu_model(self.discriminator(), gpus=self.gpus)
            else:
                self.discriminator_train_model = self.discriminator()
        
            # set trainable flag and recompile        
            self.generator().trainable = False
            self.classifier().trainable = False
            self.discriminator().trainable = True
            self.discriminator_train_model.compile(optimizer=Adam(lr=0.0001), loss=loss_disc, metrics=['binary_accuracy'])
                
        callbacks = []
        callbacks.append(TensorBoard(log_dir=out_dir+'logs/discriminator/'))
        callbacks.append(self.SaveWeightsCallback(target_models=[self.discriminator()], out_dir=out_dir+'weights/', period=out_period))
        
        history = self.discriminator_train_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                    epochs=epochs+self.disc_current_epochs, callbacks=callbacks,
                                    workers=0, validation_data=valid_gen, validation_steps=100,
                                    shuffle=False, initial_epoch=self.disc_current_epochs)
        self.disc_current_epochs += epochs

        return history
    
    def generate(self, inputs):
        """
        generate frontal image
        """
        
        if self.generator is None:
            self._init_generator()
        
        img128, img64, img32, fc2, front_leye_img, front_reye_img, front_nose_img, front_mouth_img\
        = self.generator().predict(inputs)
        
        
        img128 = (img128*np.iinfo(np.uint8).max).astype(np.uint8)
        img64 = (img64*np.iinfo(np.uint8).max).astype(np.uint8)
        img32 = (img32*np.iinfo(np.uint8).max).astype(np.uint8)
        front_leye_img = (front_leye_img*np.iinfo(np.uint8).max).astype(np.uint8)
        front_reye_img = (front_reye_img*np.iinfo(np.uint8).max).astype(np.uint8)
        front_nose_img = (front_nose_img*np.iinfo(np.uint8).max).astype(np.uint8)
        front_mouth_img = (front_mouth_img*np.iinfo(np.uint8).max).astype(np.uint8)
        
        return img128, img64, img32, front_leye_img, front_reye_img, front_nose_img, front_mouth_img
    
    def rotate_parts(self, inputs):
        """
        generate rotated part images
        """
        
        out_leyes, _, out_reyes, _, out_noses, _, out_mouthes, _ = self.parts_rotator().predict(inputs)
        
        out_leyes1 = (out_leyes*np.iinfo(np.uint8).max).astype(np.uint8)
        r_image, g_image, b_image = cv2.split(out_leyes1)
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        out_leyes = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        
        
        out_reyes1 = (out_reyes*np.iinfo(np.uint8).max).astype(np.uint8)
        r_image, g_image, b_image = cv2.split(out_reyes1)
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        out_reyes = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        
        out_noses1 = (out_noses*np.iinfo(np.uint8).max).astype(np.uint8)
        r_image, g_image, b_image = cv2.split(out_noses1)
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        out_noses = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        
        out_mouthes1 = (out_mouthes*np.iinfo(np.uint8).max).astype(np.uint8)
        r_image, g_image, b_image = cv2.split(out_mouthes1)
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        out_mouthes = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        
        return out_leyes, out_reyes, out_noses, out_mouthes
    
    def discriminate(self, frontal_img):
        """
        discriminate frontal image.
        
        Returns: discriminated score map
        """
        
        if self.discriminator is None:
            self._init_discriminator()
        
        out_img = self.discriminator().predict(frontal_img[np.newaxis, ...])[0]
        out_img = (out_img*np.iinfo(np.uint8).max).astype(np.uint8)
        print('out_img',out_img)
        out_img1 = cv2.resize(out_img, (frontal_img.shape[1], frontal_img.shape[0]))
        
        r_image, g_image, b_image = cv2.split(out_img1)

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        out_img = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        
        return out_img
    
class FaceNet():
    def __init__(self, extractor_weights=None, in_size_hw=(128, 128)):
        self.in_size_hw = in_size_hw
        self.extractor_weights = extractor_weights
        self._extractor = None
        
    def extractor(self):

        self._extractor = self.InceptionResNetV1(
                input_shape=(None, None, 3)
        )
        
        self._extractor.load_weights(self.extractor_weights)
            
        return self._extractor
    
    def scaling(self,x, scale):
        return x * scale

    def conv2d_bn(self,x,
                  filters,
                  kernel_size,
                  strides=1,
                  padding='same',
                  activation='relu',
                  use_bias=False,
                  name=None):
        x = Conv2D(filters,
                   kernel_size,
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias,
                   name=name)(x)
        if not use_bias:
            bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
            bn_name = self._generate_layer_name('BatchNorm', prefix=name)
            x = BatchNormalization(axis=bn_axis,
                                   momentum=0.995,
                                   epsilon=0.001,
                                   scale=False,
                                   name=bn_name)(x)
        if activation is not None:
            ac_name = self._generate_layer_name('Activation', prefix=name)
            x = Activation(activation, name=ac_name)(x)
        return x
    
    
    def _generate_layer_name(self,name, branch_idx=None, prefix=None):
        if prefix is None:
            return None
        if branch_idx is None:
            return '_'.join((prefix, name))
        return '_'.join((prefix, 'Branch', str(branch_idx), name))
    
    
    def _inception_resnet_block(self,x, scale, block_type, block_idx,
                                activation='relu'):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        if block_idx is None:
            prefix = None
        else:
            prefix = '_'.join((block_type, str(block_idx)))
        name_fmt = partial(self._generate_layer_name, prefix=prefix)
    
        if block_type == 'Block35':
            branch_0 = self.conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
            branch_1 = self.conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
            branch_1 = self.conv2d_bn(branch_1,
                                 32,
                                 3,
                                 name=name_fmt('Conv2d_0b_3x3', 1))
            branch_2 = self.conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
            branch_2 = self.conv2d_bn(branch_2,
                                 32,
                                 3,
                                 name=name_fmt('Conv2d_0b_3x3', 2))
            branch_2 = self.conv2d_bn(branch_2,
                                 32,
                                 3,
                                 name=name_fmt('Conv2d_0c_3x3', 2))
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'Block17':
            branch_0 = self.conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))
            branch_1 = self.conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
            branch_1 = self.conv2d_bn(branch_1,
                                 128, [1, 7],
                                 name=name_fmt('Conv2d_0b_1x7', 1))
            branch_1 = self.conv2d_bn(branch_1,
                                 128, [7, 1],
                                 name=name_fmt('Conv2d_0c_7x1', 1))
            branches = [branch_0, branch_1]
        elif block_type == 'Block8':
            branch_0 = self.conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
            branch_1 = self.conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
            branch_1 = self.conv2d_bn(branch_1,
                                 192, [1, 3],
                                 name=name_fmt('Conv2d_0b_1x3', 1))
            branch_1 = self.conv2d_bn(branch_1,
                                 192, [3, 1],
                                 name=name_fmt('Conv2d_0c_3x1', 1))
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "Block35", "Block17" or "Block8", '
                             'but got: ' + str(block_type))
    
        mixed = Concatenate(axis=channel_axis,
                            name=name_fmt('Concatenate'))(branches)
        up = self.conv2d_bn(mixed,
                       K.int_shape(x)[channel_axis],
                       1,
                       activation=None,
                       use_bias=True,
                       name=name_fmt('Conv2d_1x1'))
        up = Lambda(self.scaling,
                    output_shape=K.int_shape(up)[1:],
                    arguments={'scale': scale})(up)
        x = Add()([x, up])
        if activation is not None:
            x = Activation(activation, name=name_fmt('Activation'))(x)
        return x
    
    
    def InceptionResNetV1(self,input_shape=(160, 160, 3),
                          classes=128,
                          dropout_keep_prob=0.8,
                          weights_path=None):
        inputs = Input(shape=input_shape)
        x = self.conv2d_bn(inputs,
                      32,
                      3,
                      strides=2,
                      padding='valid',
                      name='Conv2d_1a_3x3')
        x = self.conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
        x = self.conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        x = self.conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
        x = self.conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
        x = self.conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')
    
        # 5x Block35 (Inception-ResNet-A block):
        for block_idx in range(1, 6):
            x = self._inception_resnet_block(x,
                                        scale=0.17,
                                        block_type='Block35',
                                        block_idx=block_idx)
    
        # Mixed 6a (Reduction-A block):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        name_fmt = partial(self._generate_layer_name, prefix='Mixed_6a')
        branch_0 = self.conv2d_bn(x,
                             384,
                             3,
                             strides=2,
                             padding='valid',
                             name=name_fmt('Conv2d_1a_3x3', 0))
        branch_1 = self.conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = self.conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_1 = self.conv2d_bn(branch_1,
                             256,
                             3,
                             strides=2,
                             padding='valid',
                             name=name_fmt('Conv2d_1a_3x3', 1))
        branch_pool = MaxPooling2D(3,
                                   strides=2,
                                   padding='valid',
                                   name=name_fmt('MaxPool_1a_3x3', 2))(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)
    
        # 10x Block17 (Inception-ResNet-B block):
        for block_idx in range(1, 11):
            x = self._inception_resnet_block(x,
                                        scale=0.1,
                                        block_type='Block17',
                                        block_idx=block_idx)
    
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        name_fmt = partial(self._generate_layer_name, prefix='Mixed_7a')
        branch_0 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
        branch_0 = self.conv2d_bn(branch_0,
                             384,
                             3,
                             strides=2,
                             padding='valid',
                             name=name_fmt('Conv2d_1a_3x3', 0))
        branch_1 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = self.conv2d_bn(branch_1,
                             256,
                             3,
                             strides=2,
                             padding='valid',
                             name=name_fmt('Conv2d_1a_3x3', 1))
        branch_2 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = self.conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = self.conv2d_bn(branch_2,
                             256,
                             3,
                             strides=2,
                             padding='valid',
                             name=name_fmt('Conv2d_1a_3x3', 2))
        branch_pool = MaxPooling2D(3,
                                   strides=2,
                                   padding='valid',
                                   name=name_fmt('MaxPool_1a_3x3', 3))(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)
    
        # 5x Block8 (Inception-ResNet-C block):
        for block_idx in range(1, 6):
            x = self._inception_resnet_block(x,
                                        scale=0.2,
                                        block_type='Block8',
                                        block_idx=block_idx)
        x = self._inception_resnet_block(x,
                                    scale=1.,
                                    activation=None,
                                    block_type='Block8',
                                    block_idx=6)
    
        # Classification block
        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
        # Bottleneck
        x = Dense(classes, use_bias=False, name='Bottleneck')(x)
        bn_name = self._generate_layer_name('BatchNorm', prefix='Bottleneck')
        x = BatchNormalization(momentum=0.995,
                               epsilon=0.001,
                               scale=False,
                               name=bn_name)(x)
        x = Lambda(K.l2_normalize, arguments={'axis': 1}, name='normalize')(x)
    
        # Create model
        model = Model(inputs, x, name='inception_resnet_v1')
        if weights_path is not None:
            model.load_weights(weights_path)
    
        return model
    
class CloudableModel(Model):
    """
    wrapper of keras model. this class override some functions to be available for Google Cloud Strorage.
    """
    
    def load_weights(self, filepath, by_name=False):
        print('begin loading weights file. target file: {}'.format(filepath))

        super().load_weights(filepath=filepath, by_name=by_name)
        
        print('end loading weights file. target file: {}'.format(filepath))
    
    def save_weights(self, filepath, overwrite=True):
        print('begin saving weights file. target file: {}'.format(filepath))
        

        Model.save(filepath, overwrite=overwrite)
            
        print('end saving weights file. target file: {}'.format(filepath))
        
        
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool.map(TPGAN, range(0,500))
    pool.close()     
