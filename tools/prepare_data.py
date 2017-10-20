# -*- coding: utf-8 -*-
# @Time    : 17-10-3 下午6:26
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : prepare_data.py
# @Software: PyCharm Community Edition

import os
import numpy as np
import cv2
from glob import glob
from joblib import Parallel, delayed


def split_data_with_tag(tag_list):
    train_files = []
    test_files = []

    f = open(tag_list, 'r')
    tags = f.readlines()
    for i in range(1, len(tags)):
        tag = tags[i - 1]
        text = ('%.05d_img1.png %.05d_img2.png %.05d_flow.flo' % (i, i, i))

        if int(tag) == 1:
            train_files.append(text)
        else:
            test_files.append(text)

    f.close()

    return train_files, test_files 


def write_file(filename, file_list):
    file = open(filename, 'w')

    for i in range(len(file_list)):
        f = file_list[i] + '\n'
        file.write(f)

    file.close()

def convert_ppm_2_png(id):

    raw_data_dir = '/home/fei/Data/fei/flow/FlyingChairs_release/data'
    save_data_dir = '/home/fei/Data/fei/flow/FlyingChairs_release/data_png'

    raw_filename1 = '%.5d_img1.ppm' % id
    raw_filename2 = '%.5d_img2.ppm' % id

    if os.path.isfile(os.path.join(raw_data_dir, raw_filename1)) is False or os.path.isfile(os.path.join(raw_data_dir, raw_filename2)) is False:
        return

    print(raw_filename1 + ' exists')

    img1 = cv2.imread(os.path.join(raw_data_dir, raw_filename1))
    img2 = cv2.imread(os.path.join(raw_data_dir, raw_filename2))

    save_filename1 = '%.5d_img1.png' % id
    save_filename2 = '%.5d_img2.png' % id

    cv2.imwrite(os.path.join(save_data_dir, save_filename1), img1)
    cv2.imwrite(os.path.join(save_data_dir, save_filename2), img2)








# test
train_files, test_files = split_data_with_tag('../flyting_train_valid.txt')
print(len(train_files), len(test_files))
write_file('flyingchairs_train.txt', train_files)
write_file('flyingchairs_validation.txt', test_files)


# convert_ppm_2_jpg(raw_data_dir, 2, save_data_dir)

# Parallel(n_jobs=12)(delayed(convert_ppm_2_png)(n) for n in range(1, 22873))



