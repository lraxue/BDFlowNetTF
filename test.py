# -*- coding: utf-8 -*-
# @Time    : 17-10-4 下午4:10
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : test.py
# @Software: PyCharm Community Edition

import os
import tensorflow as tf
import argparse
import numpy as np
from bilinear_sampler import *
from eval.FlowLoader import *
from collections import namedtuple
from skimage.io import imread, imsave

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

weightedflow_parameters = namedtuple('parameters',
                                     'encoder,'
                                     'height, width,'
                                     'dataset_dir,'
                                     'grid_params,'
                                     'bs_params,'
                                     'd_shape_img,'
                                     'd_shape_flow,'
                                     'img_net_shape,'
                                     'flow_net_shape')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('grid_params', {'sigma_luma': 4, 'sigma_chroma': 4, 'sigma_spatial': 2}, 'grid_params')
flags.DEFINE_string('bs_params', {'lam': 80, 'A_diag_min': 1e-5, 'cg_tol': 1e-5, 'cg_maxiter': 25}, 'bs_params')
flags.DEFINE_boolean('write_flows', False, 'Write confidence, .flo and img files')
flags.DEFINE_integer('batchsize', 20, 'Batch size for eval loop.')
flags.DEFINE_integer('testsize', 640, 'Number of test samples')
flags.DEFINE_integer('d_shape_img', [384, 512, 3], 'Data shape: width, height, channels')
flags.DEFINE_integer('d_shape_flow', [384, 512, 2], 'Data shape: width, height, channels')
flags.DEFINE_integer('img_net_shape', [384, 512, 3], 'Image shape: width, height, channels')
flags.DEFINE_integer('flow_net_shape', [384, 512, 2], 'Image shape: width, height, 2')
flags.DEFINE_integer('record_bytes', 1572876, 'Flow record bytes reader')


def aee_tf(gt, calc_flows):
    "average end point error"
    square = tf.square(gt - calc_flows)
    x = square[:, :, 0]
    y = square[:, :, 1]
    sqr = tf.sqrt(tf.add(x, y))
    aee = tf.reduce_mean(sqr)
    return aee


RGB_TO_YUV = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0, 0.0, 1.402],
    [1.0, -0.34414, -0.71414],
    [1.0, 1.772, 0.0]])

YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)
MAX_VAL = 255.0


def test_transform():
    # I = tf.constant(1, tf.float32, [1, 4, 4, 1])
    I = tf.random_uniform([1, 3, 4, 1], 0, 1, tf.float32)
    flow = tf.random_normal([1, 3, 4, 2], 0, 0, tf.float32)

    I_bil = tf.image.resize_bilinear(I, [3, 4])

    I_est_h = bilinear_sampler_1d_h(I, flow[:, :, :, 0])
    I_est = bilinear_sampler(I, flow)

    sess = tf.Session()

    print(sess.run(I))
    print('\n')

    print(sess.run(flow))
    print('\n')

    print(sess.run(I_est))
    print('\n')

    print(sess.run(I_bil))
    print('\n')

    print(sess.run(I_est_h))
    print('\n')

    # print(sess.run(x_t))
    # print(sess.run(y_t))
    # print('\n')


def test_flow_reader():
    filepath = './flyingchairs_test_test.txt'
    dataset_dir = '/home/fei/Data/fei/flow/FlyingChairs_release/data'
    with open(filepath, 'r') as f:
        filelists = f.readlines()
    filenames = []

    for i in range(len(filelists)):
        filenames.append(os.path.join(dataset_dir, filelists[i].strip().split()[2]))
        flow = load_flow(filenames, 'flyingchairs', flags.FLAGS)
        print(flow)


def test_rgb2yuv():
    # rgb = np.arange(60.).reshape(4, 5, 3)
    rgb = imread('/home/fei/Data/fei/flow/FlyingChairs_release/data_png/00001_img1.png')
    rgb = np.array(rgb, np.float32)
    print(rgb)
    print(rgb.shape)
    yuv = np.tensordot(rgb, RGB_TO_YUV, ([2], [1]))
    print(yuv)


def test_aee():
    I0 = tf.random_uniform([2, 3, 4, 2], 0, 1, tf.float32)
    I1 = I0
    I2 = tf.ones([2, 3, 4, 2], tf.float32)
    I3 = tf.zeros([2, 3, 4, 2], tf.float32)

    aee01 = aee_tf(I0, I1)
    aee02 = aee_tf(I0, I2)
    aee23 = aee_tf(I2, I3)

    sess = tf.Session()
    print(sess.run(aee01))
    print(sess.run(aee02))
    print(sess.run(aee23))

def test_filter(kernel_size, mode='mean'):
    values = np.ones([1, kernel_size, kernel_size, 3], np.float32)

    img = tf.ones([1, 5, 5, 3], tf.float32)
    ftr = tf.Variable(values)
    out = tf.nn.conv2d(img, ftr, strides=[1, 1, 1, 1], padding='VALID')

    sess = tf.Session()
    sess.run(img)
    sess.run(ftr)
    sess.run(out)

def main():
    # test_flow_reader()
    # test_transform()
    # test_rgb2yuv()

    # test_aee()

    test_filter(3)


if __name__ == '__main__':
    main()
