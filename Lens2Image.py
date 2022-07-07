"""
Class
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:1'
import sys
import glob
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from scipy import ndimage
from astropy.io import fits
from tensorflow.keras.layers import Activation, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Reshape, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, UpSampling2D, Embedding, multiply
from tensorflow.keras.layers import ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Input
import tensorflow.keras.utils as utils


class LensReconstruction():
    # this code is aimed at get the reverse source image from a lens image
    def __init__(self, trainset_path, testset_path,
                 img_size=(256, 256, 1),
                 lbl_size=(256, 256, 1),
                 generator_first=16, discriminator_first=8,
                 kerasoptimizer=keras.optimizers.Adam, learning_rate=(0.001, 0.9),
                 discriminator_losses='mse', discriminator_metrics=['accuracy'],
                 generator_losses=['mse', 'mae'], loss_weights=[1, 100]
                 ):
        """
        :param trainset_path: two-element string array,
                              whose 0-element is the training data pictures
                              the 1-element is the training label pictures
        :param testset_path: two-element array,
                             whose 0-element is the training label pictures
                             the 1-element is the testing label data
        :param img_size: 3-element 1-D int array, default is (256, 256, 1)
                         (row, col, channel)
        :param lbl_size: 3-element 1-D int array, default is (256, 256, 1)
                         (row, col, channel)
        :param generator_first:  int, default is 16
                                the number of cells in first layer of the generator
        :param discriminator_first: int, default is 8
                                    the number of cells in first layer of discriminator
        :param kerasoptimizer: callable, default is keras.optimizers.Adam
        :param learning_rate: float array, default is (0.001, 0.9)
        :param discriminator_losses: string, default is "mse"
        :param discriminator_metrics: list, default is ['accuracy']
        :param generator_losses: list, default is ['mse', 'mae']
        :param loss_weights: list, default is [1, 100]
        """
        # data pass
        self.trainset_path = trainset_path
        self.testset_path = testset_path

        # 32x32x1 shape_input; input the lensed image
        self.img_rows = img_size[0]
        self.img_cols = img_size[1]
        self.img_channels = img_size[2]
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.lbl_rows = lbl_size[0]
        self.lbl_cols = lbl_size[1]
        self.lbl_channels = lbl_size[2]
        self.lbl_shape = (self.lbl_rows, self.lbl_cols, self.lbl_channels)

        self.patch_length = int(self.img_rows / 2 ** 3)
        self.d_patch = (self.patch_length, self.patch_length, 1)

        # number of s in the first layer
        self.g_f = generator_first
        self.d_f = discriminator_first

        # adam optimizer and loss
        optimizer = kerasoptimizer(*learning_rate)
        d_losses = discriminator_losses
        g_losses = generator_losses

        # discriminator
        self.discriminator = self.lens2source_discriminator()  # build D
        self.discriminator.compile(loss=d_losses, optimizer=optimizer,
                                   metrics=discriminator_metrics)
        # generator
        self.generator = self.lens2source_generator()  # build G

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

    def load_image(self, fold_path, str_num, img_type="png", as_gray=True,
                   readmethod=cv2.IMREAD_GRAYSCALE):

        str_num = str(str_num)
        str_num = str_num.zfill(6)
        file_path = fold_path + r"/image_" + str_num + r"." + img_type
        img = cv2.imread(file_path, readmethod)

        return img

    def batch_load_image(self, batch_list, fold_path,
                         img_type="png",
                         readmethod=cv2.IMREAD_GRAYSCALE):

        batch_img = []
        for i in range(len(batch_list)):

            img = self.load_image(fold_path, batch_list[i],
                                  img_type=img_type,
                                  readmethod=readmethod)
            batch_img.append(img)

        return np.array(batch_img)

    def lens2source_generator(self):
        # ----------------------------------#
        #   generator，n, n ,1 --> n, n, 1  #
        # ----------------------------------#

        ##############
        # build model#
        ##############
        model = Sequential()

        # n, n, 1 --> n/2, n/2, 8
        model.add(Conv2D(8, kernel_size=3, strides=2,
                         padding="same", input_shape=(self.lbl_rows, self.lbl_cols, self.lbl_channels)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # n/2, n/2, 8 --> n/4, n/4, 64
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # n/4, n/4, 64 --> n/8, n/8, 64
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # n/8, n/8, 64--> n/16, n/16, 128
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # n/16, n/16, 128 --> n/32, n/32, 256
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # upsampling
        # n/32, n/32, 256 -> n/16, n/16, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # upsampling
        # n/16, n/16, 128 -> n/8, n/8, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # upsampling
        # n/8, n/8, 128 -> n/4, n/4, 64
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # upsampling
        # n/4, n/4, 64 -> n/2, n/2, 16
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # upsampling
        # n/2, n/2, 16 -> n, n, 16
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # n, n, 16 -> n, n, 1
        model.add(Conv2D(self.img_channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        ##################
        # Input and noise#
        ##################

        # label input layer
        model_input = Input(shape=(self.lbl_rows, self.lbl_cols, self.lbl_channels))

        # output layer
        img = model(model_input)

        return Model(model_input, img)

    def lens2source_discriminator(self):
        # -------------------------------------------------------- #
        #   discriminator receive (n，,n ,1) pictures, output True or False
        # -------------------------------------------------------- #
        model = Sequential()
        # n, n, 2 -> n/2, n/2, 16
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same",
                         input_shape=(self.img_rows, self.img_cols, self.img_channels * 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # n/2, n/2, 16 -> n/4, n/4, 32
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # n/4, n/4, 32 -> n/8, n/8, 128
        # model.add(ZeroPadding2D(((0,1),(0,1))))  if needed
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # n/8, n/8, 128 -> n/16, n/16, 128
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        # n/16, n/16, 128 --> (n/16)**2 *128,
        model.add(GlobalAveragePooling2D())

        # bp
        model.add(Dense(128,activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.2))

        ###################
        # Input and output#
        ###################
        # receive a image as input layer
        img = Input(shape=self.img_shape)  # n x n x 1
        lbl = Input(shape=self.lbl_shape)  # n x n x 1
        lbl_img = Conv2D(self.img_channels, kernel_size=3, strides=2, padding='same')(lbl)
        input_layer = Concatenate()([img, lbl])

        # grab the feature of images and labels
        features = model(input_layer)

        # judge whether the features are right or not
        valid = Dense(1, activation='sigmoid')(features)

        return Model([img, lbl], valid)

    def lens2source_train(self, epochs, batch_size=128, train_im_path="gray",
              train_lb_path="lensed",
              img_type="png", as_gray=True,
              readmethod=cv2.IMREAD_GRAYSCALE,
              progress=False, progress_interval=20, progress_save=True, progress_file="result/progress.log",
              plot_image=False, show_plots=False,
              save_plots=True, plot_save_iter=50, save_plots_path="result", save_plots_type="pdf",
              save_iter=100, savegfile=None, savedfile=None,
              g_summary=False, g_summary_path=None,
              d_summary=False, d_summary_path=None):

        """
        :param epochs: int, the training epochs
        :param batch_size: int, default is 128
                            the training batch
        :param train_im_path: string, default is "gray"
                              the training image path
        :param train_lb_path: string, default is "lensed"
                              the training label path
        :param img_type: string, default is "png"
                         the image type of inputs
        :param as_gray: bool, default is True
                        whether read input as gray
        :param readmethod: int, default is cv2.IMREAD_GRAYSCALE
                           the load method of the images and labels
        :param progress_interval: string, default is 20
                                  the interval output the training result
        :param progress: bool, default is False
                        whether output the progress
        :param progress_save: bool, default is True
                              whether save the progress
        :param progress_file: string, default is "result/progress.log"
                              the save file of progress
        :param plot_image: bool, default is False
                           whether show up the results in images
        :param save_plots_path: string, default is "result"
        :param save_plots_type: string, default is "pdf"
        :param plot_save_iter: int, default is 50
        :param save_iter: int, default is 100
        :param savegfile: string, save generator
        :param savedfile: string, sve discriminator
        :param g_summary:
        :param d_summary:
        :return:
        """

        # build valid and fake arrays
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # gain datasets
            #select batch numbers
            batch_list = np.random.randint([400 for i in range(128)])
            # the im_train/lb_train is in form of (number, row, col, channel)

            im_train = self.batch_load_image(batch_list, train_im_path,
                                             img_type=img_type,
                                             readmethod=readmethod)
            lb_train = self.batch_load_image(batch_list, train_lb_path,
                                             img_type=img_type,
                                             readmethod=readmethod)

            """set the gaussian as train data"""
            im_train = im_train.astype(np.float32) / 127.5 - 1
            if len(im_train.shape) == 3:
                im_train = np.expand_dims(im_train, axis=3)
            lb_train = lb_train.astype(np.float32) / 127.5 - 1
            if len(lb_train.shape) == 3:
                lb_train = np.expand_dims(lb_train, axis=3)


            # choose random batch images to train discriminator
            # get random labels
            idx = np.random.randint(0, im_train.shape[0], batch_size)
            idx_noise = np.random.randint(0, im_train.shape[0], batch_size)
            imgs, labels = im_train[idx], lb_train[idx]  # n x n # n x n
            fake_label = lb_train[idx_noise]  # batch_size of n x n matrix

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

            ##########
            # output #
            ##########

            #output progress
            if ( progress == True ) and ( epoch % progress_interval == 0 ):
                print("Generation: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
                if progress_save == True:
                    with open(progress_file, "a+", encoding="utf-8") as prog_file:
                        prog_file.write("Generation: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]\n" % (
                                        epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))


            #output images
            if plot_image == True:

                im_test = np.load(self.testset_path[0])
                lb_test = np.load(self.testset_path[1])

                test_rand = np.random.randint(0, len(im_train), batch_size)
                test_imgs, test_labels = im_test[test_rand], lb_test[test_rand]  # n # n
                test_gen_imgs = self.generator.predict(test_labels)

                show_num = np.random.randint(128)
                plt.figure(1, figsize=(20, 6))

                plt.subplot(131)
                plt.imshow(test_imgs[show_num].reshape(32, 32))
                plt.title("origin source")

                plt.subplot(132)
                plt.imshow(test_labels[show_num].reshape(64, 64))
                plt.title("blurred image (input)")

                plt.subplot(133)
                plt.imshow(test_gen_imgs[show_num].reshape(32, 32))
                plt.title("predicted source (output)")

                if show_plots == True:
                    plt.show()

                if (save_plots == True) and ((epoch+1)%plot_save_iter == 0):
                    plt.savefig(
                                save_plots_path + "/epoch_" +
                                str(epoch).zfill(int(1+np.ceil(np.log10(epochs))))
                                + "." + save_plots_type
                                )

            #save g/d summary
            if g_summary:
                try:
                    self.g.summary()
                except:
                    pass
            if d_summary:
                try:
                    self.d.summary()
                except:
                    pass

            #save g/d models
            if save_iter and ((epoch + 1) / save_iter == 0 or epoch == (epochs - 1)):

                try:

                    self.generator.save(savegfile)
                    self.discriminator.save(savedfile)

                except:

                    pass
