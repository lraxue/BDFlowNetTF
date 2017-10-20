# -*- coding: utf-8 -*-
# @Time    : 17-10-3 下午4:23
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : WeightedFlowDataloader.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class WeightedFlowDataloader(object):
    """weightedflow dataloader"""

    def __init__(self, datapath, file_names_file, params, dataset, mode):
        self.data_path = datapath
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch = None
        self.right_image_batch = None
        self.flow_batch = None

        input_queue = tf.train.string_input_producer([file_names_file], shuffle=False)
        line_reader = tf.TextLineReader()
        key, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # Load two images for test/train
        left_image_path = tf.string_join([self.data_path, split_line[0]])
        right_image_path = tf.string_join([self.data_path, split_line[1]])
        flow_path = tf.string_join([self.data_path, split_line[2]])

        left_image_o = self.read_image(left_image_path)
        right_image_o = self.read_image(right_image_path)
        flow_o = self.load_flow(flow_path)

        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            left_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o), lambda: right_image_o)
            flow = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(flow_o), lambda: flow_o)

            # randomly augment images
            do_augment = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5,
                                              lambda: self.augment_image_pair(left_image, right_image),
                                              lambda: (left_image, right_image))

            left_image.set_shape([None, None, 3])
            right_image.set_shape([None, None, 3])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.left_image_batch, self.right_image_batch, self.flow_batch = tf.train.shuffle_batch(
                [left_image, right_image, flow],
                params.batch_size, capacity,
                min_after_dequeue,
                params.num_threads)

        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o, tf.image.flip_left_right(left_image_o)], 0)
            self.left_image_batch.set_shape([2, None, None, 3])

            self.right_image_batch = tf.stack([right_image_o, tf.image.flip_left_right(right_image_o)], 0)
            self.right_image_batch.set_shape([2, None, None, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image

    def load_flow(self, flow_path):
        # flow_path = [flow_path]
        filename_queue = tf.train.string_input_producer([flow_path], shuffle=False)
        record_bytes = self.params.record_bytes  # 1572876
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.float32)

        magic = tf.slice(record_bytes, [0], [1])  # .flo number 202021.25
        size = tf.slice(record_bytes, [1], [2])  # size of flow / image
        flows = tf.slice(record_bytes, [3], [np.prod(self.params.d_shape_flow)])
        flows = tf.reshape(flows, self.params.d_shape_flow)
        flows.set_shape(self.params.d_shape_flow)
        return flows
        # with open(flow_path, 'rb') as f:
        #     magic = np.fromfile(f, np.float32, count=1)
        #     assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        #     h = np.fromfile(f, np.int32, count=1)[0]
        #     w = np.fromfile(f, np.int32, count=1)[0]
        #     data = np.fromfile(f, np.float32, count=2 * w * h)
        # # Reshape data into 3D array (columns, rows, bands)
        # data2D = np.resize(data, (w, h, 2))
        # data_tensor = tf.convert_to_tensor(data2D)
        # return tf.image.resize_images(data_tensor, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)


class FlowDataloader(object):
    def __init__(self, datapath, list0, list1, flow_list, params, dataset, mode, img_type):
        self.data_path = datapath
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch = None
        self.right_image_batch = None
        self.flow_batch = None

        print(len(list0), len(list1), len(flow_list))
        assert len(list0) == len(list1) == len(flow_list) != 0, ('Input lengths not correct')

        input_queue = tf.train.slice_input_producer([list0, list1], shuffle=False)
        # image reader
        content_0 = tf.read_file(input_queue[0])
        content_1 = tf.read_file(input_queue[1])
        if img_type == "jpeg":
            left_image_o = tf.image.decode_jpeg(content_0, channels=3)
            right_image_o = tf.image.decode_jpeg(content_1, channels=3)
        elif img_type == "png":
            left_image_o = tf.image.decode_png(content_0, channels=3)
            right_image_o = tf.image.decode_png(content_1, channels=3)

        left_image_o = tf.image.convert_image_dtype(left_image_o, tf.float32)
        right_image_o = tf.image.convert_image_dtype(right_image_o, tf.float32)

        # flow reader
        filename_queue = tf.train.string_input_producer(flow_list, shuffle=False)
        record_bytes = self.params.record_bytes  # 1572876
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.float32)

        magic = tf.slice(record_bytes, [0], [1])  # .flo number 202021.25
        size = tf.slice(record_bytes, [1], [2])  # size of flow / image
        flows = tf.slice(record_bytes, [3], [np.prod(self.params.d_shape_flow)])
        flow_o = tf.reshape(flows, self.params.d_shape_flow)

        self.negative_ones = -tf.ones([self.params.height, self.params.width, 1])

        if mode == 'train':
            # randomly flip images
            left_image = left_image_o
            right_image = right_image_o
            flow = flow_o
            # do_h_flip = tf.random_uniform([], 0, 1)
            # left_image = tf.cond(do_h_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o), lambda: left_image_o)
            # right_image = tf.cond(do_h_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: right_image_o)
            # flow = tf.cond(do_h_flip > 0.5, lambda: tf.image.flip_left_right(flow_o), lambda: flow_o)
            # flow = tf.cond(do_h_flip > 0.5, lambda: self.opposite(flow, 1), lambda: flow)

            # TODO: Not good in test
            # do_v_flip = tf.random_uniform([], 0, 1)
            # left_image = tf.cond(do_v_flip > 0.5, lambda: tf.image.flip_up_down(left_image), lambda: left_image)
            # right_image = tf.cond(do_v_flip > 0.5, lambda: tf.image.flip_up_down(right_image), lambda: right_image)
            # flow = tf.cond(do_v_flip > 0.5, lambda: tf.image.flip_up_down(flow), lambda: flow)
            # flow = tf.cond(do_v_flip > 0.5, lambda: self.opposite(flow, 2), lambda: flow)

            # do_ex_left_right = tf.random_uniform([], 0, 1)
            # left_image = tf.cond(do_ex_left_right > 0.5, lambda: right_image, lambda: left_image)
            # right_image = tf.cond(do_ex_left_right > 0.5, lambda: left_image, lambda: right_image)
            # flow = tf.cond(do_ex_left_right > 0.5, lambda: flow, lambda: flow)
            # flow = tf.cond(do_ex_left_right > 0.5, lambda: self.opposite(flow, 0), lambda: flow)


            # randomly augment images
            do_augment = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5,
                                              lambda: self.augment_image_pair(left_image, right_image),
                                              lambda: (left_image, right_image))

            left_image.set_shape(self.params.d_shape_img)
            right_image.set_shape(self.params.d_shape_img)
            flow.set_shape(self.params.d_shape_flow)

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.left_image_batch, self.right_image_batch, self.flow_batch = tf.train.shuffle_batch(
                [left_image, right_image, flow],
                params.batch_size, capacity,
                min_after_dequeue,
                params.num_threads)

        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o, tf.image.flip_left_right(left_image_o)], 0)
            self.left_image_batch.set_shape([2, None, None, 3])

            self.right_image_batch = tf.stack([right_image_o, tf.image.flip_left_right(right_image_o)], 0)
            self.right_image_batch.set_shape([2, None, None, 3])

    def opposite(self, x, tag=0):
        if tag == 0:
            return -x
        elif tag == 1:
            x_0 = tf.expand_dims(x[:, :, 0], -1)
            x_1 = tf.expand_dims(x[:, :, 1], -1)
            out = tf.concat([-x_0, x_1], -1)
            return out
        elif tag == 2:
            x_0 = tf.expand_dims(x[:, :, 0], -1)
            x_1 = tf.expand_dims(x[:, :, 1], -1)
            out = tf.concat([x_0, -x_1], -1)
            return out
        return x

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
