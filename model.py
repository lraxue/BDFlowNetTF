# -*- coding: utf-8 -*-
# @Time    : 17-10-2 下午3:04
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : model.py
# @Software: PyCharm Community Edition

from collections import namedtuple
import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
import cv2
import os
import math
import tensorflow.contrib.slim as slim
import tools.visualize as viz
import tools.bilateral_solver as bills

from bilinear_sampler import *

weightedflow_parameters = namedtuple('parameters',
                                     'encoder, '
                                     'height, width, '
                                     'batch_size, '
                                     'batch_norm, '
                                     'record_bytes, '
                                     'd_shape_flow, '
                                     'd_shape_img, '
                                     'num_threads, '
                                     'num_epochs, '
                                     'wrap_mode, '
                                     'use_deconv, '
                                     'alpha_image_loss, '
                                     'flow_gradient_loss_weight, '
                                     'lr_loss_weight, '
                                     'full_summary, '
                                     'scale')


class WeightedFlow(object):
    """weighted optical flow"""

    def __init__(self, params, mode, left, right, flow_gt=None, reuse_variables=None, model_index=0, optim=None,
                 batch_norm=True):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.flow_gt = flow_gt
        self.model_collection = ['model_' + str(model_index)]
        self.reuse_variables = reuse_variables
        self.batch_norm = batch_norm

        self.optim = optim
        self.scales = [params.scale, params.scale / 2., params.scale / 4., params.scale / 8., params.scale / 16.]
        self.s = 20.
        self.patch_kernel_size = 63

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def scale_pyramid(self, img, num_scales=5):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def flow_pyramid(self, flow, num_scales=5):
        scaled_flows = [flow]
        s = tf.shape(flow)
        h = s[1]
        w = s[2]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            flow_i = tf.image.resize_area(flow, [nh, nw]) * (1. / ratio)
            scaled_flows.append(flow_i)
        return scaled_flows

    def generate_image_left(self, img, flow):
        return bilinear_sampler(img, -flow)

    def generate_image_right(self, img, flow):
        return bilinear_sampler(img, flow)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_flow_smoothness(self, flow, pyramid):
        flow_gradients_x = [self.gradient_x(d) for d in flow]
        flow_gradients_y = [self.gradient_y(d) for d in flow]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [flow_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [flow_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def batch_normalization(self, inputs, is_training=True, decay=0.999, epsilon=1e-3):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    def lrelu(self, x, leak=0.1, name="lrelu"):
        if "_" in name:
            name = name + "_Relu"
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def msra_initializer(self, kl, dl):
        """
        kl for kernel size, dl for filter number
        """
        stddev = math.sqrt(2. / (kl ** 2 * dl))
        return tf.truncated_normal_initializer(stddev=stddev)

    def epe(self, input_flow, target_flow):
        square = tf.square(input_flow - target_flow)
        x = square[:, :, :, 0]
        y = square[:, :, :, 1]
        epe = tf.sqrt(tf.add(x, y))
        return epe

    def visualize_flow(self, flow, s):
        square = tf.square(flow)
        x = square[:, :, :, 0]
        y = square[:, :, :, 1]
        sqr = tf.sqrt(tf.add(x, y))
        return sqr
        # return tf.sqrt(tf.multiply(flow[:, :, :, 0], flow[:, :, :, 0]) + tf.multiply(flow[:, :, :, 1], flow[:, :, :, 1]))

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def conv(self, x, outchannels, kernel_size, stride, scope):
        output = slim.conv2d(x, outchannels, kernel_size=kernel_size, stride=stride, activation_fn=None,
                             scope=scope, weights_initializer=self.msra_initializer(kernel_size, outchannels))
        output = self.lrelu(output, name=scope)

        return output

    def upconv(self, x, num_out_layers, kernel_size, scale, scope):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1, scope=scope)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME', activation_fn=None,
                                     weights_initializer=self.msra_initializer(kernel_size, num_out_layers))
        conv = self.lrelu(conv)
        return conv[:, 3:-1, 3:-1, :]

    def conv_block(self, x, outchannels, kernel_size, scope):
        output_1 = self.conv(x, outchannels, kernel_size, 1, scope=scope)
        output = self.conv(output_1, outchannels, kernel_size, 2, scope=scope + '_1')
        return output

    def upsample_flow(self, x, ratio=2):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        output = x * ratio
        return tf.image.resize_images(output, [h * ratio, w * ratio])

    def predict_flow(self, x, scope):
        output = slim.conv2d(x, 4, kernel_size=3, stride=1, scope=scope, biases_initializer=None, activation_fn=None,
                             weights_initializer=self.msra_initializer(3, 4))
        return output

    def patch_l1_loss(self, img1, img2, kernel_size=63, stride=1, padding='SAME'):
        fil


    def NCC_vector(self, vec1, vec2):
        return None

    def NCC_image(self, img1, img2, kernel_size=63, stride=1, padding='SAME'):
        patches1 = tf.extract_image_patches(img1, ksizes=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding=padding)
        patches2 = tf.extract_image_patches(img2, ksizes=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding=padding)

        b, h, w, _ = patches1.shape().aslist()

    def build_bdflownet(self):
        conv = self.conv
        upconv = self.upconv
        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input, 32, 7, scope='conv1')  # (64, H/2)
            conv2 = self.conv_block(conv1, 64, 5, scope='conv2')  # (128, H/4)
            conv3 = self.conv_block(conv2, 128, 3, scope='conv3')  # (256, H/8)
            conv4 = self.conv_block(conv3, 256, 3, scope='conv4')  # (512, H/16)
            conv5 = self.conv_block(conv4, 512, 3, scope='conv5')  # (512, H/32)
            conv6 = self.conv_block(conv5, 512, 3, scope='conv6')  # (1024, H/64)
            conv7 = self.conv_block(conv6, 512, 3, scope='conv7')  # (1024, H/128)

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, kernel_size=3, scale=2, scope='deconv_7')  # (H/64)
            concat7 = tf.concat([upconv7, conv6], 3, name='concat_7')  # (512+512=1024)
            iconv7 = self.conv(concat7, 512, kernel_size=3, stride=1, scope='iconv_7')  # (1024->512)

            upconv6 = upconv(iconv7, 512, kernel_size=3, scale=2, scope='deconv_6')  # (H/32)
            concat6 = tf.concat([upconv6, conv5], 3, name='concat_6')  # (512+512=1024)
            iconv6 = self.conv(concat6, 512, kernel_size=3, stride=1, scope='iconv_6')  # (1024->512)

            deconv5 = upconv(iconv6, 256, kernel_size=3, scale=2, scope='deconv_5')  # (H/16 )
            concat5 = tf.concat([deconv5, conv4], 3, name='concat_5')  # (256+256=512)
            iconv5 = self.conv(concat5, 256, kernel_size=3, stride=1, scope='iconv_5')  # (512->256)
            self.flow5 = self.predict_flow(iconv5, scope='predict_flow_5')  # (512->4)
            flow5_up = self.upsample_flow(self.flow5, 2)  # (4, )

            deconv4 = upconv(iconv5, 128, kernel_size=3, scale=2, scope='deconv_4')  # (H/8 )
            concat4 = tf.concat([deconv4, conv3, flow5_up], 3, name='concat_4')  # (128+128+4=260)
            iconv4 = self.conv(concat4, 128, kernel_size=3, stride=1, scope='iconv_4')  # (260->128)
            self.flow4 = self.predict_flow(iconv4, scope='predict_flow_4')  # (1028->4)
            flow4_up = self.upsample_flow(self.flow4, 2)  # (4, )

            deconv3 = upconv(iconv4, 64, kernel_size=3, scale=2, scope='deconv_3')  # (H/4)
            concat3 = tf.concat([deconv3, conv2, flow4_up], 3, name='concat_3')  # (64+64+4=132)
            iconv3 = self.conv(concat3, 64, kernel_size=3, stride=1, scope='iconv_3')  # (132->64)
            self.flow3 = self.predict_flow(iconv3, scope='predict_flow_3')  # (132->4)
            flow3_up = self.upsample_flow(self.flow3, 2)  # (4, )

            deconv2 = upconv(iconv3, 32, kernel_size=3, scale=2, scope='deconv_2')  # (H/2)
            concat2 = tf.concat([deconv2, conv1, flow3_up], 3, name='concat_2')  # (32+32+4=68)
            iconv2 = self.conv(concat2, 32, kernel_size=3, stride=1, scope='iconv_2')  # (68->32)
            self.flow2 = self.predict_flow(iconv2, scope='predict_flow_2')  # (4, )
            flow2_up = self.upsample_flow(self.flow2, 2)  # (4, )

            deconv1 = upconv(iconv2, 16, kernel_size=3, scale=2, scope='deconv_1')  # H
            concat1 = tf.concat([deconv1, flow2_up], 3, name='concat_1')  # (32+4=36)
            iconv1 = self.conv(concat1, 16, kernel_size=3, stride=1, scope='iconv_1')  # (36->16)
            self.flow1 = self.predict_flow(iconv1, scope='predict_flow_1')  # (16->4)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose]):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                self.left_pyramid = self.scale_pyramid(self.left, self.params.scale)
                self.right_pyramid = self.scale_pyramid(self.right, self.params.scale)

                self.model_input = tf.concat([self.left, self.right], 3)

                if self.mode == 'train':
                    self.flow_gt_pyramid = self.flow_pyramid(self.flow_gt, 5)

                # build model
                if self.params.encoder == 'bdflownet':
                    self.build_bdflownet()
                else:
                    return None

    def build_outputs(self):
        # Store Flow
        with tf.variable_scope('flows'):
            self.flow_est = [self.flow1, self.flow2, self.flow3, self.flow4, self.flow5]
            self.flow_left_est = [d[:, :, :, 0:2] for d in self.flow_est]
            self.flow_right_est = [d[:, :, :, 2:4] for d in self.flow_est]

            print(self.flow1.shape)
            print(self.flow2.shape)
            print(self.flow3.shape)
            print(self.flow4.shape)
            print(self.flow5.shape)

            self.flow_images = [self.visualize_flow(self.flow_left_est[i] * (self.s / (2 ** i)), i) for i in
                                range(self.params.scale)]

            if self.mode == 'test':
                self.flow_left_est[0] *= self.s
                # self.flow_left_est[0][: ,:, :, 1] *= self.params.height

                # self.flow_left_est[0] = tf.image.resize_bicubic(self.flow_left_est[0], [384, 512])
                # self.flow_right_est[0] = tf.image.resize_bicubic(self.flow_left_est[0], [384, 512])
                return

            self.flow_final = (self.flow_left_est[0] + self.flow_right_est[0]) * self.s / 2.
            self.flow_diff = self.epe(self.flow_final, self.flow_gt)
            self.flow_error = tf.reduce_mean(self.flow_diff)
            self.flow_gt_img = self.visualize_flow(self.flow_gt, 0)

        # Generate images
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.flow_left_est[i] * self.s / (2. ** i))
                             for i in range(self.params.scale)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.flow_right_est[i] * self.s / (2. ** i))
                              for i in range(self.params.scale)]

        # LR consistency
        with tf.variable_scope('left-right'):
            self.right_to_left_flow = [self.generate_image_left(self.flow_right_est[i], self.flow_left_est[i]) for i in
                                       range(self.params.scale)]
            self.left_to_right_flow = [self.generate_image_right(self.flow_left_est[i], self.flow_right_est[i]) for i in
                                       range(self.params.scale)]

        # Flow smoothness
        with tf.variable_scope('smoothness'):
            self.flow_left_smoothness = self.get_flow_smoothness(self.flow_left_est, self.left_pyramid)
            self.flow_right_smoothness = self.get_flow_smoothness(self.flow_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # Image reconstruction
            # L1
            # self.l1_left = [self.patch_l1_loss(self.left_est[i], self.left_pyramid[i], kernel_size= self.patch_kernel_size//(2**i)) for i in range(self.params.scale)]
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(self.params.scale)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            # self.l1_right = [self.patch_l1_loss(self.right_est[i], self.right_pyramid[i], kernel_size= self.patch_kernel_size//(2**i)) for i in range(self.params.scale)]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(self.params.scale)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(self.params.scale)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(self.params.scale)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # Weight Sum
            self.image_loss_right = [
                self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_right[i] for i in range(self.params.scale)]
            self.image_loss_left = [
                self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left[i] for i in range(self.params.scale)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # Flow smoothness
            self.flow_left_loss = [tf.reduce_mean(tf.abs(self.flow_left_smoothness[i])) / (2 ** i) for i in
                                   range(self.params.scale)]
            self.flow_right_loss = [tf.reduce_mean(tf.abs(self.flow_right_smoothness[i])) / (2 ** i) for i in
                                    range(self.params.scale)]
            self.flow_gradient_loss = tf.add_n(self.flow_left_loss + self.flow_right_loss)

            # LR consistency
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_flow[i] - self.flow_left_est[i])) for i in
                                 range(self.params.scale)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_flow[i] - self.flow_right_est[i])) for i in
                                  range(self.params.scale)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # Total loss
            self.total_loss = self.image_loss + self.params.flow_gradient_loss_weight * self.flow_gradient_loss + self.params.lr_loss_weight * self.lr_loss

    def build_summaries(self):
        # SUMMARIES
        max_output = 3
        with tf.device('/cpu:0'):
            for i in range(self.params.scale):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i),
                                  self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('flow_gradient_loss_' + str(i), self.flow_left_loss[i] + self.flow_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.image('flow_left_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_left_est[i][:, :, :, 0]) * self.s / (2 ** i), -1),
                                 max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_left_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_left_est[i][:, :, :, 1]) * self.s / (2 ** i), -1),
                                 max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_right_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_right_est[i][:, :, :, 0]) * self.s / (2 ** i), -1),
                                 max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_right_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_right_est[i][:, :, :, 1]) * self.s / (2 ** i), -1),
                                 max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_result_' + str(i), tf.expand_dims(self.flow_images[i], -1),
                                 max_outputs=max_output, collections=self.model_collection)

                tf.summary.histogram('flow_u_' + str(i), self.flow_left_est[i][:, :, :, 0] * self.s / (2 ** i),
                                     collections=self.model_collection)
                tf.summary.histogram('flow_v_' + str(i), self.flow_left_est[i][:, :, :, 1] * self.s / (2 ** i),
                                     collections=self.model_collection)
                # tf.summary.image('_result_' + str(i), tf.expand_dims(self.flow_images[i], -1), max_outputs=max_output, collections=self.model_collection)
                # tf.summary.image('flow_result_' + str(i), tf.expand_dims(self.flow_images[i], -1), max_outputs=max_output, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_left_' + str(i), self.ssim_left[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('l1_left_' + str(i), self.l1_left[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('left_' + str(i), self.left_pyramid[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('right_' + str(i), self.right_pyramid[i], max_outputs=max_output,
                                     collections=self.model_collection)
            tf.summary.image('flow_error_with_gt', tf.expand_dims(self.flow_diff, -1), max_outputs=max_output,
                             collections=self.model_collection)
            tf.summary.image('flow_groundtruth', tf.expand_dims(self.flow_gt_img, -1), max_outputs=max_output,
                             collections=self.model_collection)
            tf.summary.histogram('flow_diff', self.flow_diff, collections=self.model_collection)
            # if self.optim is not None:
            #     train_vars = [var for var in tf.trainable_variables()]
            #     self.grads_and_vars = self.optim.compute_gradients(self.total_loss,
            #                                                   var_list=train_vars)
            #     for var in tf.trainable_variables():
            #           tf.summary.histogram(var.op.name + "/values", var)
            #     for grad, var in self.grads_and_vars:
            #         tf.summary.histogram(var.op.name + "/gradients", grad)
