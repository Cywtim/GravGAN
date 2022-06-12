"""
Class
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import sys
import glob
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow.keras as keras
import tensorflow as tf
from scipy import ndimage
from IPython import display
from astropy.io import fits
from tensorflow.keras.layers import Activation, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Reshape, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, UpSampling2D, Embedding, multiply
from tensorflow.keras.layers import ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import tensorflow.keras.utils as utils


class Source2Lens():
    # this code is aimed at get the reverse source image from a lens image
    def __init__(self, trainset_path, testset_path,
                 img_rows=64, img_cols=64, img_channels=1,
                 lbl_rows=64, lbl_cols=64, lbl_channels=1,
                 generator_first=16,discriminator_first=8,
                 learning_rate=0.001, beta_1=0.9,
                 discriminator_losses='mse', discriminator_metrics=['accuracy'],
                 generator_losses=['mse', 'mae'], loss_weights=[1, 100]
                 ):

        # data pass
        self.trainset_path = trainset_path

        # 32x32x1 shape_input; input the lensed image
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.lbl_rows = lbl_rows
        self.lbl_cols = lbl_cols
        self.lbl_channels = lbl_channels
        self.lbl_shape = (self.lbl_rows, self.lbl_cols, self.lbl_channels)

        self.patch_length = int(self.img_rows / 2 ** 3)
        self.d_patch = (self.patch_length, self.patch_length, 1)

        # number of s in the first layer
        self.g_f = generator_first
        self.d_f = discriminator_first

        # adam optimizer and loss
        optimizer = keras.optimizers.Adam(learning_rate, beta_1)
        d_losses = discriminator_losses
        g_losses = generator_losses

        # discriminator
        self.discriminator = self.build_discriminator()  # build D
        self.discriminator.compile(loss=d_losses, optimizer=optimizer,
                                   metrics=discriminator_metrics)
        # generator
        self.generator = self.build_generator()  # build G

        ######################
        # def combined network#
        ######################
        # set D trainable as false，
        # build the G
        images = Input(shape=self.img_shape)
        labels = Input(shape=self.lbl_shape)
        fakes = self.generator(labels)

        # fix discriminator when training generator
        self.discriminator.trainable = False

        # discriminator
        valid = self.discriminator([fakes, labels])

        self.combined = Model(inputs=[images, labels], outputs=[valid, fakes])
        self.combined.compile(loss=g_losses,
                              loss_weights=loss_weights
                              , optimizer=optimizer)

    def build_generator(self):
        # --------------------------------- #
        #   generator，64,64,1 --> 32,32,1
        # --------------------------------- #

        #############
        # build model#
        #############
        model = Sequential()

        # 64,64,1 --> 32,32,8
        model.add(Conv2D(8, kernel_size=3, strides=2,
                         padding="same", input_shape=(self.lbl_rows, self.lbl_cols, self.lbl_channels)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # 32,32,8 --> 16,16,64
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # 16,16,64 --> 8,8,64
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # 8,8,64 --> 4,4,128
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # 4,4,128 --> 2,2,256
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # upsampling
        # 2,2,256 -> 4,4,128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # 4, 4, 128 -> 8, 8, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # upsampling
        # 8, 8, 128 -> 16,16 , 64
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # upsampling
        # 16, 16, 64 -> 32,32 , 16
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # 32, 32, 16 -> 32, 32, 1
        model.add(Conv2D(self.img_channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        # model.summary()

        #################
        # Input and noise#
        #################

        # label input layer
        model_input = Input(shape=(self.lbl_rows, self.lbl_cols, self.lbl_channels))

        # output layer
        img = model(model_input)

        return Model(model_input, img)

    def build_discriminator(self):
        # -------------------------------------------------------- #
        #   discriminator receive (64，,64 ,1) pictures, output True or False
        # -------------------------------------------------------- #
        model = Sequential()
        # 64, 64, 1 -> 16, 16, 8
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same",
                         input_shape=(self.img_rows, self.img_cols, self.img_channels * 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # 16, 16, 8 -> 8, 8, 64
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # 8, 8, 64 -> 4, 4, 128
        # model.add(ZeroPadding2D(((0,1),(0,1))))  if needed
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # 4,4,128 -> 2,2,128
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # 2,2,128 --> 128,
        model.add(GlobalAveragePooling2D())

        # bp

        # model.summary()

        ##################
        # Input and output#
        ##################
        # recieve a image as input layer
        img = Input(shape=self.img_shape)  # 32x32
        lbl = Input(shape=self.lbl_shape)  # 64x64
        lbl_img = Conv2D(self.img_channels, kernel_size=3, strides=2, padding='same')(lbl)
        input_layer = Concatenate()([img, lbl_img])

        # grab the feature of the input images
        features = model(input_layer)

        # judge whether the features is right or not
        valid = Dense(1, activation='sigmoid')(features)

        return Model([img, lbl], valid)

    def train(self, epochs, batch_size=128, sample_interval=20,
              progress=False,plot=False,save=0,savegfile=None,savedfile=None):
        """

        :param epochs:
        :param batch_size:
        :param sample_interval:
        :param progress:
        :param plot:
        :param save:
        :param savegfile:
        :param savedfile:
        :return:
        """


        # gain datasets

        im_train = np.load(self.trainset_path[0])
        lb_train = np.load(self.trainset_path[1])

        """set the gaussian as train data"""
        im_train = im_train.astype(np.float32) / 127.5 - 1
        im_train = np.expand_dims(im_train, axis=3)
        lb_train = lb_train.astype(np.float32) / 127.5 - 1
        lb_train = np.expand_dims(lb_train, axis=3)



        # build valid and fake arrays
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # choose random batch images to train discriminator
            # get random labels
            idx = np.random.randint(0, im_train.shape[0], batch_size)
            idx_noise = np.random.randint(0, im_train.shape[0], batch_size)
            imgs, labels = im_train[idx], lb_train[idx]  # 32 # 64
            fake_label = lb_train[idx_noise]  # batch_size of 64x64 matrix

            # generate fake images
            gen_imgs = self.generator.predict(fake_label)

            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            ##################
            # train generator#
            ##################

            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch([imgs, labels], [valid, imgs])



            if progress == True:
                if epoch % sample_interval == 0:
                    print("Generation: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))


            if plot == True:

                im_test = np.load(self.testset_path[0])
                lb_test = np.load(self.testset_path[1])

                test_rand = np.random.randint(0, len(im_train), batch_size)
                test_imgs, test_labels = im_test[test_rand], lb_test[test_rand]  # 32 # 64
                test_gen_imgs = self.generator.predict(test_labels)

                show_num = np.random.randint(128)
                plt.figure(1, figsize=(20, 6))

                plt.subplot(131)
                plt.imshow(test_imgs[show_num].reshape(32, 32))
                plt.title("orign source")

                plt.subplot(132)
                plt.imshow(test_labels[show_num].reshape(64, 64))
                plt.title("blurred image (input)")

                plt.subplot(133)
                plt.imshow(test_gen_imgs[show_num].reshape(32, 32))
                plt.title("predicted source (output)")

                plt.show()

                # save_model(self.generator,"C:\\Users\\CHENG\\Documents\\wam\\generator.h5")
                # save_model(self.discriminator,"C:\\Users\\CHENG\\Documents\\wam\\discriminator.h5")

            if save and ((epoch + 1) / save == 0 or epoch == (epochs - 1) ):

                try:

                    self.generator.save(savegfile)
                    self.discriminator.save(savedfile)

                except:

                    pass
