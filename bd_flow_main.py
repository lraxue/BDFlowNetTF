# -*- coding: utf-8 -*-
# @Time    : 17-10-3 下午3:45
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : bd_flow_main.py
# @Software: PyCharm Community Edition

from __future__ import division
import os
import argparse
import numpy as np
import time
import tensorflow as tf
from model import *
from compute_average_gradients import *
from WeightedFlowDataloader import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='WeightedFlow TensorFlow implementation.')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--model_name', type=str, help='model name', default='weighted_flow')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or resnet50', default='bdflownet')
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=384)
parser.add_argument('--input_width', type=int, help='input width', default=512)
parser.add_argument('--record_bytes', type=int, help='flow record bytes reader', default=1572876)
parser.add_argument('--d_shape_flow', type=int, help='flow record bytes reader', default=[384, 512, 2])
parser.add_argument('--d_shape_image', type=int, help='flow record bytes reader', default=[384, 512, 3])

parser.add_argument('--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--batch_norm', type=bool, help='batch normalizatio', default=False)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight', type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--scale', type=int, help='pyramid of image', default=5)
parser.add_argument('--flow_gradient_loss_weight', type=float, help='flow smoothness weigth', default=0.1)
parser.add_argument('--wrap_mode', type=str, help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv', help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus', type=int, help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=12)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                    action='store_true')
parser.add_argument('--full_summary',
                    help='if set, will keep more data for each summary. Warning: the file can become very large',
                    action='store_true', default=True)

args = parser.parse_args()

def slice_vector(vec, size):
    x = tf.slice(vec, [0, 0, 0, 0], [size] + [256, 512, 2][:2] + [1])
    y = tf.slice(vec, [0, 0, 0, 1], [size] + [256, 512, 2][:2] + [1])
    return tf.squeeze(x), tf.squeeze(y)


def aee_f(gt, calc_flows, size):
    "average end point error"
    square = tf.square(gt - calc_flows)
    x, y = slice_vector(square, size)
    sqr = tf.sqrt(tf.add(x, y))
    aee = tf.metrics.mean(sqr)
    return aee


def var_mean(flow_to_mean):
    """Pyfunc wrapper for the confidence / mean calculation"""

    def _var_mean(flow_to_mean):
        """ confidence / mean calculation"""
        flow_to_mean = np.array(flow_to_mean)
        x = flow_to_mean[:, :, :, 0]
        y = flow_to_mean[:, :, :, 1]
        var_x = np.var(x, 0)
        var_y = np.var(y, 0)
        # var_mea = np.mean(np.array([var_x, var_y]), 0)
        var_mea = (var_x + var_y) / 2
        # TODO check /2, /8 here ...
        var_mea = np.exp(-1 * np.array(var_mea, np.float32) / 8)
        # normalize
        flow_x_m = np.mean(x, 0)
        flow_y_m = np.mean(y, 0)
        flow_to_mean = np.zeros(list([256, 512, 2]), np.float32)
        flow_to_mean[:, :, 0] = flow_x_m
        flow_to_mean[:, :, 1] = flow_y_m
        var_img = np.zeros(list([256, 512, 2]), np.float32)
        var_img[:, :, 0] = var_mea
        var_img[:, :, 1] = var_mea
        var_img[:, :, 2] = var_mea
        return [flow_to_mean, var_mea, var_img]

    solved_data = tf.py_func(_var_mean, [flow_to_mean], [tf.float32, tf.float32, tf.float32], name='flow_mean')
    mean, var, var_img = solved_data[:]
    mean = tf.squeeze(tf.stack(mean))
    var = tf.squeeze(tf.stack(var))
    var_img = tf.squeeze(tf.stack(var_img))
    mean.set_shape(list([256, 512, 2]))
    var.set_shape(list([256, 512, 2][:2]) + [1])
    var_img.set_shape(list([256, 512, 2]))
    return mean, var, var_img


def post_process_flow(flow):
    _, h, w, c = flow.shape
    l_flow = flow[0, :, :, :]
    r_flow = np.fliplr(flow[1, :, :, :])
    m_flow = 0.5 * (l_flow + r_flow)

    return m_flow
    # l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    # r_mask = np.fliplr(l_mask)
    # flow_u = r_mask * l_flow[:, :, :, 0] + l_mask * r_flow + (1.0 - l_mask - r_mask) * m_flow
    # return r_mask * l_flow + l_mask * r_flow + (1.0 - l_mask - r_mask) * m_flow


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def train(params):
    """Training loop"""
    with open(args.filenames_file) as f:
        file_list = f.readlines()

    img_list0 = []
    img_list1 = []
    flow_list = []

    for i in range(len(file_list)):
        files = file_list[i].strip().split()
        img_list0.append(os.path.join(args.data_path, files[0]))
        img_list1.append(os.path.join(args.data_path, files[1]))
        flow_list.append(os.path.join(args.data_path, files[2]))

    with tf.Graph().as_default():  # , tf.device('/cpu:0')
        global_step = tf.Variable(0, trainable=False)

        # Optimizer
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        # boundaries = [np.int32((3 / 6) * num_total_steps), np.int32((4 / 6) * num_total_steps)]
        # values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        boundaries = [np.int32((2 / 5) * num_total_steps), np.int32((3 / 5) * num_total_steps), np.int32((4 / 5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4, args.learning_rate / 8]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = FlowDataloader(args.data_path, img_list0, img_list1, flow_list, params, args.dataset, args.mode, img_type='png')
        left = dataloader.left_image_batch
        right = dataloader.right_image_batch
        flow = dataloader.flow_batch

        # Split for each gpu
        left_splits = tf.split(left, args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)
        flow_splits = tf.split(flow, args.num_gpus, 0)

        tower_grads = []
        tower_losses = []
        tower_errors = []
        reuse_variables = None

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                # with tf.device('/gpu:%d' % i):
                model = WeightedFlow(params, args.mode, left_splits[i], right_splits[i], flow_splits[i], reuse_variables, i, opt_step,
                                     batch_norm=False)

                loss = model.total_loss
                tower_losses.append(loss)

                error = model.flow_error
                tower_errors.append(error)

                # reuse_variables = True
                grads = opt_step.compute_gradients(loss)
                tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)
        total_loss = tf.reduce_mean(tower_losses)
        total_error = tf.reduce_mean(tower_errors)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        tf.summary.scalar('total_error', total_error, ['model_0'])
        train_vars = [var for var in tf.trainable_variables()]
        grads_and_vars = opt_step.compute_gradients(total_loss, var_list=train_vars)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var, ['model_0'])
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad, ['model_0'])


        summary_op = tf.summary.merge_all('model_0')

        # Session
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # Save
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # Count params
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # Init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # Load checkpoint
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path)

            if args.retrain:
                sess.run(global_step.assign(0))

        # Go
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value, error_value = sess.run([apply_gradient_op, total_loss, total_error])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | error: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, error_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 5000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)


def test(params):
    dataloader = WeightedFlowDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = WeightedFlow(params, args.mode, left, right, batch_norm=False)

    # Session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # Saver
    train_saver = tf.train.Saver()

    # Init
    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    flows = np.zeros((num_test_samples * 2, 384, 512, 2), dtype=np.float32)
    flows_pp = np.zeros((num_test_samples, 384, 512, 2), dtype=np.float32)
    for step in range(num_test_samples):
        flow = sess.run(model.flow_left_est[0])
        flows[step * 2 + 0] = flow[0].squeeze()
        flows[step * 2 + 1] = flow[1].squeeze()

        print('test example: ', step)
        print(np.max(flows[step * 2 + 0]))
        print(np.min(flows[step * 2 + 0]))
        print(np.mean(flows[step * 2 + 0]))
        print('\n')
        # flows_pp[step] = post_process_flow(flow.squeeze())

    print('done.')

    print('writing flow.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/fei_20171019_bi_flows.npy', flows)
    # np.save(output_directory + '/disparities_pp.npy', disparities_pp)

    print('done.')


def main(_):
    params = weightedflow_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        scale=args.scale,
        batch_size=args.batch_size,
        batch_norm=args.batch_norm,
        record_bytes=args.record_bytes,
        d_shape_flow=args.d_shape_flow,
        d_shape_img=args.d_shape_image,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        flow_gradient_loss_weight=args.flow_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)


if __name__ == '__main__':
    tf.app.run()
