from tqdm import tqdm
from time import sleep
import numpy as np
import os
import csv
import tensorflow as tf

channel = 1
default_height = 48
default_width = 48

csv_file_name = 'fer2013.csv'

# 直接使用绝对路径来指定 fer2013.csv 的位置
csv_path = r"A:\projectcode\emotiondect\data\fer2013\fer2013.csv"

# 如果需要生成 TFRecord 的路径，可以保持不变；若需要保存在别处可自行修改
data_folder_name = r'A:\projectcode\emotiondect\emotion_classifier-master\emotion_classifier_tensorflow_version\temp'
data_path_name = 'cv'
record_name_train = 'fer2013_train.tfrecord'
record_name_test = 'fer2013_test.tfrecord'
record_name_eval = 'fer2013_eval.tfrecord'
record_path_train = os.path.join(data_folder_name, data_path_name, record_name_train)
record_path_test = os.path.join(data_folder_name, data_path_name, record_name_test)
record_path_eval = os.path.join(data_folder_name, data_path_name, record_name_eval)

# 读取 CSV 文件
with open(csv_path, 'r') as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']

def write_binary(record_name_, labels_images_, height_=default_height, width_=default_width):
    writer_ = tf.io.TFRecordWriter(record_name_)
    for label_image_ in tqdm(labels_images_):
        label_ = int(label_image_[0])
        image_ = np.asarray([int(p) for p in label_image_[-1].split()])

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_])),
                    "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height_])),
                    "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width_])),
                    "image/raw": tf.train.Feature(int64_list=tf.train.Int64List(value=image_))
                }
            )
        )
        writer_.write(example.SerializeToString())
    writer_.close()

# 写入 TFRecord 文件
write_binary(record_path_train, trn)
write_binary(record_path_test, tst)
write_binary(record_path_eval, val)
