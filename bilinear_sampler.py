# -*- coding: utf-8 -*-
# @Time    : 17-10-3 上午10:52
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : bilinear_sampler.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[1]
        _width        = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output


def bicubic_interp_2d(input_, new_size, endpoint=False):
    """
    Args :
      input_ : Input tensor. Its shape should be
          [batch_size, height, width, channel].
          In this implementation, the shape should be fixed for speed.
      new_size : The output size [new_height, new_width]
    ref : http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
    """

    with tf.variable_scope('bicubic'):
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channel = shape[3]

        def _hermite(A, B, C, D, t):
            a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
            b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
            c = A * (-0.5) + C * 0.5
            d = B

            return a * t * t * t + b * t * t + c * t + d

        def _get_grid_array(n_i, y_i, x_i, c_i):
            n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
            n = np.expand_dims(n, axis=4)
            y = np.expand_dims(y, axis=4)
            x = np.expand_dims(x, axis=4)
            c = np.expand_dims(c, axis=4)

            return np.concatenate([n, y, x, c], axis=4)

        def _get_frac_array(y_d, x_d, n, c):
            y = y_d.shape[0]
            x = x_d.shape[0]
            y_t = y_d.reshape([1, -1, 1, 1])
            x_t = x_d.reshape([1, 1, -1, 1])
            y_t = tf.constant(np.tile(y_t, (n, 1, x, c)), dtype=tf.float32)
            x_t = tf.constant(np.tile(x_t, (n, y, 1, c)), dtype=tf.float32)
            return y_t, x_t

        def _get_index_tensor(grid, x, y):
            new_grid = np.array(grid)

            grid_y = grid[:, :, :, :, 1] + y
            grid_x = grid[:, :, :, :, 2] + x

            grid_y = np.clip(grid_y, 0, height - 1)
            grid_x = np.clip(grid_x, 0, width - 1)

            new_grid[:, :, :, :, 1] = grid_y
            new_grid[:, :, :, :, 2] = grid_x

            return tf.constant(new_grid, dtype=tf.int32)

        new_height = new_size[0]
        new_width = new_size[1]

        n_i = np.arange(batch_size)
        c_i = np.arange(channel)

        if endpoint:
            y_f = np.linspace(0., height - 1, new_height)
        else:
            y_f = np.linspace(0., height, new_height, endpoint=False)
        y_i = y_f.astype(np.int32)
        y_d = y_f - np.floor(y_f)

        if endpoint:
            x_f = np.linspace(0., width - 1, new_width)
        else:
            x_f = np.linspace(0., width, new_width, endpoint=False)
        x_i = x_f.astype(np.int32)
        x_d = x_f - np.floor(x_f)

        grid = _get_grid_array(n_i, y_i, x_i, c_i)
        y_t, x_t = _get_frac_array(y_d, x_d, batch_size, channel)

        i_00 = _get_index_tensor(grid, -1, -1)
        i_10 = _get_index_tensor(grid, +0, -1)
        i_20 = _get_index_tensor(grid, +1, -1)
        i_30 = _get_index_tensor(grid, +2, -1)

        i_01 = _get_index_tensor(grid, -1, +0)
        i_11 = _get_index_tensor(grid, +0, +0)
        i_21 = _get_index_tensor(grid, +1, +0)
        i_31 = _get_index_tensor(grid, +2, +0)

        i_02 = _get_index_tensor(grid, -1, +1)
        i_12 = _get_index_tensor(grid, +0, +1)
        i_22 = _get_index_tensor(grid, +1, +1)
        i_32 = _get_index_tensor(grid, +2, +1)

        i_03 = _get_index_tensor(grid, -1, +2)
        i_13 = _get_index_tensor(grid, +0, +2)
        i_23 = _get_index_tensor(grid, +1, +2)
        i_33 = _get_index_tensor(grid, +2, +2)

        p_00 = tf.gather_nd(input_, i_00)
        p_10 = tf.gather_nd(input_, i_10)
        p_20 = tf.gather_nd(input_, i_20)
        p_30 = tf.gather_nd(input_, i_30)

        p_01 = tf.gather_nd(input_, i_01)
        p_11 = tf.gather_nd(input_, i_11)
        p_21 = tf.gather_nd(input_, i_21)
        p_31 = tf.gather_nd(input_, i_31)

        p_02 = tf.gather_nd(input_, i_02)
        p_12 = tf.gather_nd(input_, i_12)
        p_22 = tf.gather_nd(input_, i_22)
        p_32 = tf.gather_nd(input_, i_32)

        p_03 = tf.gather_nd(input_, i_03)
        p_13 = tf.gather_nd(input_, i_13)
        p_23 = tf.gather_nd(input_, i_23)
        p_33 = tf.gather_nd(input_, i_33)

        col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
        col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
        col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
        col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
        value = _hermite(col0, col1, col2, col3, y_t)

        return value


def bilinear_sampler(input_images, flows, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W, )
        - y: flattened tensor of shape (B*H*W, )
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(img)

        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    def _generate_grid():

        # create normalized 2D grid
        x = tf.linspace(-1.0, 1.0, _width)
        y = tf.linspace(-1.0, 1.0, _height)
        x_t, y_t = tf.meshgrid(x, y)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to (x_t, y_t , 1)
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid num_batch times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        batch_grids = tf.tile(sampling_grid, tf.stack([_num_batch, 1, 1]))

        # cast to float32 (required for matmul)
        # theta = tf.cast(theta, 'float32')
        # sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid - batch multiply
        # batch_grids = tf.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape(batch_grids, [_num_batch, 2, _height, _width])
        # batch_grids = tf.transpose(batch_grids, [0, 2, 1, 3])

        return batch_grids

    def _interpolate(img, x, y):
        with tf.variable_scope('_interpolate'):
            """
            Performs bilinear sampling of the input images according to the 
            normalized coordinates provided by the sampling grid. Note that 
            the sampling is done identically for each channel of the input.
            To test if the function works properly, output image should be
            identical to input image when theta is initialized to identity
            transform.
            Input
            -----
            - img: batch of images in (B, H, W, C) layout.
            - grid: x, y which is the output of affine_grid_generator.
            Returns
            -------
            - interpolated images according to grids. Same size as grid.
            """

            """
            # prepare useful params
            B = tf.shape(img)[0]
            H = tf.shape(img)[1]
            W = tf.shape(img)[2]
            C = tf.shape(img)[3]

            max_y = tf.cast(H - 1, 'int32')
            max_x = tf.cast(W - 1, 'int32')
            zero = tf.zeros([], dtype='int32')

            # cast indices as float32 (for rescaling)
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')

            # rescale x and y to [0, W/H]
            x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
            y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

            # grab 4 nearest corner points for each (x_i, y_i)
            # i.e. we need a rectangle around the point of interest
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            # clip to range [0, H/W] to not violate img boundaries
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            # get pixel value at corner coords
            Ia = _get_pixel_value(img, x0, y0)
            Ib = _get_pixel_value(img, x0, y1)
            Ic = _get_pixel_value(img, x1, y0)
            Id = _get_pixel_value(img, x1, y1)

            # recast as float for delta calculation
            x0 = tf.cast(x0, 'float32')
            x1 = tf.cast(x1, 'float32')
            y0 = tf.cast(y0, 'float32')
            y1 = tf.cast(y1, 'float32')

            # calculate deltas
            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            # add dimension for addition
            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)

            # compute output
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            """

            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])

            # # constants
            # num_batch = tf.shape(img)[0]
            # _, height, width, channels = img.get_shape().as_list()

            x = tf.to_float(x)
            y = tf.to_float(y)
            # height_f = tf.cast(height, tf.float32)
            # width_f = tf.cast(width, tf.float32)
            zero = tf.constant(0, dtype=tf.int32)
            max_y = tf.cast(tf.shape(img)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(img)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width-1/height-1]
            # x = (x + 1.0) * (_width_f - 1.0) / 2.0
            # y = (y + 1.0) * (_height_f - 1.0) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = _width
            dim1 = _width * _height

            # Create base index
            base = tf.range(_num_batch) * dim1
            base = tf.reshape(base, [-1, 1])
            base = tf.tile(base, [1, _height * _width])
            base = tf.reshape(base, [-1])

            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore channels dim
            im_flat = tf.reshape(img, tf.stack([-1, _num_channels]))
            im_flat = tf.to_float(im_flat)
            pixel_a = tf.gather(im_flat, idx_a)
            pixel_b = tf.gather(im_flat, idx_b)
            pixel_c = tf.gather(im_flat, idx_c)
            pixel_d = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x1_f = tf.to_float(x1)
            y1_f = tf.to_float(y1)

            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
            wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
            wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

            output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
            output = tf.reshape(output,
                                shape=tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    def _transform(input_images, flows):
        with tf.variable_scope('transform'):
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(flows[:,:,:,0], [-1]) * _width_f
            y_t_flat = y_t_flat + tf.reshape(flows[:,:,:,1], [-1]) * _height_f

            x_t_flat = tf.reshape(x_t_flat, shape=(_num_batch, _height, _width))
            y_t_flat = tf.reshape(y_t_flat, shape=(_num_batch, _height, _width))

            # sample input with grid to get output
            out_fmap = _interpolate(input_images, x_t_flat, y_t_flat)

            return out_fmap

    with tf.variable_scope(name):
        # print(input_images)
        _num_batch     = tf.shape(input_images)[0]
        _height        = tf.shape(input_images)[1]
        _width         = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        # print(_num_batch, _height, _width, _num_channels)
        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        out_put = _transform(input_images, flows)
        return out_put