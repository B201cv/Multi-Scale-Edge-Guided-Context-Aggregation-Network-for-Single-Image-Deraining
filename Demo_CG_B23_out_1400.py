from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
#from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Reshape, dot, Multiply, BatchNormalization, \
    Subtract, concatenate, add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import numpy as np
# from PIL import Image
from scipy.signal import convolve2d
# import argparse
import random
import os
import cv2
from keras import backend as K
import tensorflow as tf
from keras import initializers
import math
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast

# dimensions of our images.
nb_rainy_train_samples = 12600
nb_clean_train_samples = 12600
nb_rainy_test_samples = 1400
nb_clean_test_samples = 1400
IMG_CHANNELS = 3
IMG_ROWS = 256
IMG_COLS = 256
BATCH_SIZE = 1
NB_EPOCH = 300
NB_SPLIT = 12600
# VERBOSE=1
OPTIM_main = Adam(lr=0.0001, beta_1=0.5)
OPTIM_adv = Adam(lr=0.00005, beta_1=0.5)
initializers = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)


def calc_psnr(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return peak_signal_noise_ratio(img1,img2)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calc_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def load_rainy_train_data_1400():
    Rainy_train_data = np.zeros([nb_rainy_train_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'G:\pzy\CG/rainy_train_data_1400/rain/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Rainy_train_data[count] = img
                count = count + 1
    return Rainy_train_data


def load_clean_train_data_1400():
    Clean_train_data = np.zeros([nb_clean_train_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'G:\pzy\CG/rainy_train_data_1400/clean/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Clean_train_data[count] = img
                count = count + 1
    return Clean_train_data


def load_mask_train_data_1400():
    Clean_train_data = np.zeros([nb_clean_train_samples, IMG_ROWS, IMG_COLS, 1], dtype='float32')
    class_path = 'G:\pzy\CG/rainy_train_data_1400/mask/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                # img = (img - 127.5) / 127.5
                img = img[:, :, 0]
                img = img_to_array(img)
                Clean_train_data[count] = img
                count = count + 1
    return Clean_train_data


def load_rainy_test_data_1400():
    Rainy_test_data = np.zeros([nb_rainy_test_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'G:\pzy\CG/rainy_test_data_1400/rain/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Rainy_test_data[count] = img
                count = count + 1
    return Rainy_test_data


def load_clean_test_data_1400():
    Clean_test_data = np.zeros([nb_clean_test_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'G:\pzy\CG/rainy_test_data_1400/clean/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Clean_test_data[count] = img
                count = count + 1
    return Clean_test_data


def MSC(x_in, nb_filter):
    x = x_in

    x_1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x_1 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x_1)
    x_1 = Activation("relu")(x_1)
    x_1 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x_1)
    x_1 = Activation("relu")(x_1)
    x_1_in = UpSampling2D(size=(2, 2))(x_1)
    x_1_out = UpSampling2D(size=(4, 4))(x_1)

    x_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x_2 = concatenate([x_1_in, x_2], axis=3)
    x_2 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x_2)
    x_2 = Activation("relu")(x_2)
    x_2 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x_2)
    x_2 = Activation("relu")(x_2)
    x_2_in_out = UpSampling2D(size=(2, 2))(x_2)

    x_3 = concatenate([x_2_in_out, x], axis=3)
    x_3 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x_3)
    x_3 = Activation("relu")(x_3)
    x_3 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x_3)
    x_3 = Activation("relu")(x_3)

    x_out = concatenate([x_1_out, x_2_in_out, x_3], axis=3)
    x_out = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x_out)
    return x_out


def MSARDB(x_in, nb_filter):
    x = x_in
    x = MSC(x, nb_filter)
    x = Activation("relu")(x)
    x = MSC(x, nb_filter)
    x = Activation("relu")(x)
    x_a_out = GlobalAveragePooling2D()(x)
    x_fc_1 = Dense(nb_filter)(x_a_out)
    x_fc_1 = Activation("relu")(x_fc_1)
    x_fc_2 = Dense(nb_filter)(x_fc_1)
    x_c = Activation("sigmoid")(x_fc_2)
    x_c = Reshape([1, 1, nb_filter])(x_c)
    x_c_out = Multiply()([x_c, x])
    x_out = add([x_c_out, x_in])
    return x_out


def MSARDN(x, nb_layers, nb_filter):
    for ii in range(nb_layers):
        conv = MSARDB(x, nb_filter)
        x = conv
    return x


def Res2Block(x_in, in_num):
    x = x_in
    x = Conv2D(in_num, (1, 1), strides=(1, 1), padding='same')(x)
    x = Activation("relu")(x)
    x_1 = Lambda(lambda x: x[:, :, :, 0:int(in_num / 4)])(x)
    x_2 = Lambda(lambda x: x[:, :, :, int(in_num / 4):int(in_num / 2)])(x)
    x_3 = Lambda(lambda x: x[:, :, :, int(in_num / 2):int(in_num / 4 * 3)])(x)
    x_4 = Lambda(lambda x: x[:, :, :, int(in_num / 4 * 3):int(in_num)])(x)
    y_1 = x_1
    y_2 = Conv2D(int(in_num / 4), (3, 3), strides=(1, 1), padding='same')(x_2)
    y_2 = Activation("relu")(y_2)
    x_3 = add([y_2, x_3])
    y_3 = Conv2D(int(in_num / 4), (3, 3), strides=(1, 1), padding='same')(x_3)
    y_3 = Activation("relu")(y_3)
    x_4 = add([y_3, x_4])
    y_4 = Conv2D(int(in_num / 4), (3, 3), strides=(1, 1), padding='same')(x_4)
    y_4 = Activation("relu")(y_4)
    y = concatenate([y_1, y_2, y_3, y_4], axis=3)
    y = Conv2D(in_num, (1, 1), strides=(1, 1), padding='same')(y)
    y = Activation("relu")(y)

    x_a_out = GlobalAveragePooling2D()(y)
    x_fc_1 = Dense(in_num)(x_a_out)
    x_fc_1 = Activation("relu")(x_fc_1)
    x_fc_2 = Dense(in_num)(x_fc_1)
    x_c = Activation("sigmoid")(x_fc_2)
    x_c = Reshape([1, 1, in_num])(x_c)
    x_c_out = Multiply()([x_c, y])
    y_out = add([x_c_out, x_in])
    return y_out


def DCCL(x, nb_filter):
    x_1 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1))(x)
    x_1 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1))(x_1)
    x_3 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', dilation_rate=(3, 3))(x)
    x_3 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', dilation_rate=(3, 3))(x_3)
    x_5 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', dilation_rate=(5, 5))(x)
    x_5 = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', dilation_rate=(5, 5))(x_5)
    x_cat = concatenate([x_1, x_3, x_5], axis=3)
    x_out = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x_cat)
    return x_out


def SDCAB(x_in, nb_filter):
    x = x_in
    x = DCCL(x, nb_filter)
    x = Activation("relu")(x)
    x = DCCL(x, nb_filter)
    x_out = add([x, x_in])
    return x_out


def SDCABS(x, nb_layers, nb_filter):
    for ii in range(nb_layers):
        conv = SDCAB(x, nb_filter)
        x = conv
    return x


def BGA(M1, M2, B1):
    IMG_ROWS = M1.shape[1]
    B1 = Reshape((-1, nb_filter))(B1)

    M1 = Reshape((-1, nb_filter))(M1)

    MB = dot([B1, M1], axes=2)
    MB_S = Activation("softmax")(MB)

    M2 = Reshape((-1, nb_filter))(M2)

    MB_out = dot([MB_S, M2], axes=[2, 1])
    MB_out = Reshape([IMG_ROWS, IMG_ROWS, nb_filter])(MB_out)

    return MB_out


nb_filter = 124  # 卷积层核数

inputA = Input(batch_shape=(BATCH_SIZE, IMG_ROWS, IMG_COLS, IMG_CHANNELS), name='inputA')

Main_1 = Conv2D(nb_filter, (7, 7), padding="same")(inputA)
Main_1 = Activation("relu")(Main_1)
Main_1 = Res2Block(Main_1, nb_filter)
Main_1_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Main_1)#128

Main_2 = Conv2D(nb_filter, (7, 7), padding="same")(Main_1_down)
Main_2 = Activation("relu")(Main_2)
Main_2 = Res2Block(Main_2, nb_filter)
Main_2_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Main_2)#64

Main_3 = Conv2D(nb_filter, (7, 7), padding="same")(Main_2_down)
Main_3 = Activation("relu")(Main_3)
Main_3 = Res2Block(Main_3, nb_filter)
Main_3_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Main_3)#32

Main_4 = Conv2D(nb_filter, (7, 7), padding="same")(Main_3_down)
Main_4 = Activation("relu")(Main_4)
Main_4 = Res2Block(Main_4, nb_filter)
Main_4_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Main_4)#16

Main_5 = Res2Block(Main_4_down, nb_filter)
Main_5 = Res2Block(Main_5, nb_filter)
Main_5 = Res2Block(Main_5, nb_filter)
Main_5 = Res2Block(Main_5, nb_filter)
Main_5 = Res2Block(Main_5, nb_filter)
Main_5_down = Res2Block(Main_5, nb_filter)

Main_4_up = UpSampling2D(size=(2, 2))(Main_5_down)#32
Main_4_up = concatenate([Main_4, Main_4_up], axis=3)
Main_4_up = Conv2D(nb_filter, (1, 1), padding="same")(Main_4_up)
Main_4_up = Activation("relu")(Main_4_up)
Main_4_up = Res2Block(Main_4_up, nb_filter)

Main_4_up_1 = Conv2D(nb_filter, (1, 1), padding="same")(Main_4_up)
Main_4_up_1 = Activation("relu")(Main_4_up_1)
Main_4_up_2 = Conv2D(nb_filter, (1, 1), padding="same")(Main_4_up)
Main_4_up_2 = Activation("relu")(Main_4_up_2)

# edge detection 4
B_4_up = UpSampling2D(size=(2, 2))(Main_5_down)
B_4_up = concatenate([Main_4, B_4_up], axis=3)
B_4_up = Conv2D(nb_filter, (1, 1), padding="same")(B_4_up)
B_4_up = Activation("relu")(B_4_up)
B_4_up = Res2Block(B_4_up, nb_filter)

MB_4 = BGA(Main_4_up_1, Main_4_up_2, B_4_up)
MB_4_up = add([Main_4_up, MB_4])
MB_4_up = Main_4_up

B_4_out = UpSampling2D(size=(8, 8))(B_4_up)
B4 = Conv2D(2, (1, 1), padding="same")(B_4_out)
B4_out = Activation("sigmoid", name="B4_out")(B4)


Main_3_up = UpSampling2D(size=(2, 2))(MB_4_up)#64
Main_3_up = concatenate([Main_3, Main_3_up], axis=3)
Main_3_up = Conv2D(nb_filter, (1, 1), padding="same")(Main_3_up)
Main_3_up = Activation("relu")(Main_3_up)
Main_3_up = Res2Block(Main_3_up, nb_filter)

Main_3_up_1 = Conv2D(nb_filter, (1, 1), padding="same")(Main_3_up)
Main_3_up_1 = Activation("relu")(Main_3_up_1)
Main_3_up_2 = Conv2D(nb_filter, (1, 1), padding="same")(Main_3_up)
Main_3_up_2 = Activation("relu")(Main_3_up_2)
########################################

# edge detection 3
MB_4_up = Main_1_down
B_3_up = UpSampling2D(size=(2, 2))(Main_4_up)
B_3_up = concatenate([Main_3, B_3_up], axis=3)
B_3_up = Conv2D(nb_filter, (1, 1), padding="same")(B_3_up)
B_3_up = Activation("relu")(B_3_up)
B_3_up = Res2Block(B_3_up, nb_filter)

MB_3 = BGA(Main_3_up_1, Main_3_up_2, B_3_up)
MB_3_up = add([Main_3_up, MB_3])
B_3_out = UpSampling2D(size=(4, 4))(B_3_up)
B3 = Conv2D(2, (1, 1), padding="same")(B_3_out)
B3_out = Activation("sigmoid", name="B3_out")(B3)
Main_2_up = UpSampling2D(size=(2, 2))(MB_3_up)#128
Main_2_up = concatenate([Main_2, Main_2_up], axis=3)
Main_2_up = Conv2D(nb_filter, (1, 1), padding="same")(Main_2_up)
Main_2_up = Activation("relu")(Main_2_up)
Main_2_up = Res2Block(Main_2_up, nb_filter)

Main_2_up_1 = Conv2D(nb_filter, (1, 1), padding="same")(Main_2_up)
Main_2_up_1 = Activation("relu")(Main_2_up_1)
Main_2_up_2 = Conv2D(nb_filter, (1, 1), padding="same")(Main_2_up)
Main_2_up_2 = Activation("relu")(Main_2_up_2)
######################################

# edge detection 2
B_3_up = Main_2_down
B_2_up = UpSampling2D(size=(2, 2))(B_3_up)
B_2_up = concatenate([Main_2, B_2_up], axis=3)
B_2_up = Conv2D(nb_filter, (1, 1), padding="same")(B_2_up)
B_2_up = Activation("relu")(B_2_up)
B_2_up = Res2Block(B_2_up, nb_filter)

MB_2 = BGA(Main_2_up_1, Main_2_up_2, B_2_up)
MB_2_up = add([Main_2_up, MB_2])

B_2_out = UpSampling2D(size=(2, 2))(B_2_up)
B2 = Conv2D(2, (1, 1), padding="same")(B_2_out)
B2_out = Activation("sigmoid", name="B2_out")(B2)
#############################

Main_1_up = UpSampling2D(size=(2, 2))(MB_2_up)#256
Main_1_up = concatenate([Main_1, Main_1_up], axis=3)
Main_1_up = Conv2D(nb_filter, (1, 1), padding="same")(Main_1_up)
Main_1_up = Activation("relu")(Main_1_up)
Main_1_up = Res2Block(Main_1_up, nb_filter)

Main = Conv2D(3, (3, 3), padding="same")(Main_1_up)
Main_out = Activation("tanh", name="Main_out")(Main)
# model = Model(inputs=inputA, outputs=[Main_out, B2_out, B3_out, B4_out])
model = Model(inputs=inputA, outputs=[Main_out, B2_out])
model.summary()
# model.compile(
#     loss={'Main_out': "mean_absolute_error", 'B2_out': "binary_crossentropy",
#           'B3_out': "binary_crossentropy", 'B4_out': "binary_crossentropy"},
#     loss_weights={'Main_out': 1, 'B2_out': 0.5, 'B3_out': 0.5, 'B4_out': 0.5}, optimizer=OPTIM_main)
model.compile(
    loss={'Main_out': "mean_absolute_error", 'B2_out': "binary_crossentropy"},
    loss_weights={'Main_out': 1, 'B2_out': 0.5, 'B3_out': 0.5}, optimizer=OPTIM_main)

Rainy_train_data_1400 = load_rainy_train_data_1400()
Clean_train_data_1400 = load_clean_train_data_1400()
Mask_train_data_1400 = load_mask_train_data_1400()
Rainy_test_data_1400 = load_rainy_test_data_1400()
Clean_test_data_1400 = load_clean_test_data_1400()

for i in range(NB_EPOCH):
    length_1400 = Rainy_train_data_1400.shape[0]
    idx_1400 = np.arange(0, length_1400)
    np.random.shuffle(idx_1400)
    Rainy_train_data_1400 = Rainy_train_data_1400[idx_1400]
    Clean_train_data_1400 = Clean_train_data_1400[idx_1400]
    Mask_train_data_1400 = Mask_train_data_1400[idx_1400]
    Rain_train_image_1400 = np.array_split(Rainy_train_data_1400, NB_SPLIT)
    Clean_train_image_1400 = np.array_split(Clean_train_data_1400, NB_SPLIT)
    Mask_train_image_1400 = np.array_split(Mask_train_data_11400, NB_SPLIT)

    index = list(range(NB_SPLIT))
    random.shuffle(index)
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))
    '''for j in index:
        history = model.train_on_batch(x={'inputA': Rain_train_image_1200[j]},
                                       y={'Main_out': Clean_train_image_1200[j], 'B2_out': Mask_train_image_1200[j],
                                          'B3_out': Mask_train_image_1200[j], 'B4_out': Mask_train_image_1200[j]})
        print('epoch:' + str(i) + '...' +
              'Loss:' + str(history[0]) + '...' +
              'Main_out:' + str(history[1]) + '...' +
              'B2_out:' + str(history[2]) + '...' +
              'B3_out:' + str(history[3]) + '...' +
              'B4_out:' + str(history[4]))'''
    for j in index:
        history = model.train_on_batch(x={'inputA': Rain_train_image_1400[j]},
                                       y={'Main_out': Clean_train_image_1400[j], 'B2_out': Mask_train_image_1400[j], 'B3_out': Mask_train_image_1400[j]})
        print('epoch:' + str(i) + '...' +
              'Loss:' + str(history[0]) + '...' +
              'Main_out:' + str(history[1]) + '...' +
              'B2_out:' + str(history[2]) + '...' +
              'B3_out:' + str(history[3]) + '...' )

    [predicted_clean_data_1400, bb, cc] = model.predict(Rainy_test_data_1400, batch_size=BATCH_SIZE)

    num = Rainy_test_data_1400.shape[0]
    Score_psnr_1400 = []
    Score_ssim_1400 = []

    for k in range(num):

        predicted_clean_image_1400 = predicted_clean_data_1400[k].reshape(IMG_ROWS, IMG_COLS, 3)
        predicted_clean_image_1400 = np.uint8((predicted_clean_image_1400 + 1) * 127.5)

        Clean_test_image_1400 = Clean_test_data_1400[k]
        Clean_test_image_1400 = Clean_test_image_1400.reshape(IMG_ROWS, IMG_COLS, 3)
        Clean_test_image_1400 = np.uint8((Clean_test_image_1400 + 1) * 127.5)

        img_PSNR_1400 = calc_psnr(Clean_test_image_1400, predicted_clean_image_1400)
        img_SSIM_1400 = calc_ssim(Clean_test_image_1400, predicted_clean_image_1400)
        Score_psnr_1400.append(img_PSNR_1400)
        Score_ssim_1400.append(img_SSIM_1400)

        if k % 50 == 0:
            Image_1400= np.concatenate((Rainy_test_image_1400, predicted_clean_image_1400, Clean_test_image_1400),
                                        axis=1)
            cv2.imwrite('G:\pzy\CG\output_CG_V2\output1/generated_' + str(i) + '_' + str(k) + '_' + '.png', Image_1200)

    Score_psnr_mean_1400 = np.mean(Score_psnr_1400)
    Score_ssim_mean_1400 = np.mean(Score_ssim_1400)
    line_PSNR_1400 = "%.4f \n" % (Score_psnr_mean_1400)
    with open('G:\pzy\CG\output_CG_V2\PSNR_1200.txt', 'a') as f:
        f.write(line_PSNR_1400)
    line_SSIM_1400 = "%.4f \n" % (Score_ssim_mean_1400)
    with open('G:\pzy\CG\output_CG_V2\SSIM_1200.txt', 'a') as f:
        f.write(line_SSIM_1400)
    if i%50==0:
        model.save_weights('G:\pzy\CG\output_CG_V2\CG_V0.h5')