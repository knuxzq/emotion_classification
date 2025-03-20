import os
import tensorflow as tf
import cv2
import numpy as np
import csv
import tqdm
from tensorflow.keras import layers, regularizers, initializers

#====================#
#  路径参数（已修改）  #
#====================#
# CSV 文件名
csv_file_name = 'fer2013.csv'
# CSV 文件绝对路径
csv_path = r"A:\projectcode\emotiondect\data\fer2013\fer2013.csv"

# TFRecord 等文件所属文件夹
data_folder_name = r"A:\projectcode\emotiondect\emotion_classifier-master\emotion_classifier_tensorflow_version\temp"
data_path_name = 'cv'  # 子文件夹名
pic_path_name = 'pic'
cv_path_name = 'fer2013'

# 模型文件及其位置
model_path = r"A:\projectcode\emotiondect\emotion_classifier-master\pic\model\keras_model\model_weight.h5"
casc_name = 'haarcascade_frontalface_alt.xml'

# 中间路径组合
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)

#====================#
#     其他参数设置    #
#====================#
channel = 1
default_height = 48
default_width = 48
confusion_matrix = False
use_advanced_method = True
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)

# 定义 LocalResponseNormalization 层
class LocalResponseNormalization(layers.Layer):
    def __init__(self, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        return tf.compat.v1.nn.lrn(
            inputs,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta
        )

# 定义 CNNModel 类
class CNNModel(tf.keras.Model):
    def __init__(self, num_tags=7, hidden_dim=1024, optimizer='Adam', learning_rate=0.001):
        super(CNNModel, self).__init__()
        self.num_tags = num_tags
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotUniform()
        self.l2_reg = regularizers.l2(0.001)

        self.conv_feature = [32, 32, 32, 64]
        self.conv_size = [1, 5, 3, 5]
        self.maxpool_size = [None, 3, 3, 3]
        self.maxpool_stride = [None, 2, 2, 2]

        self.conv1 = layers.Conv2D(filters=self.conv_feature[0], kernel_size=self.conv_size[0], padding='same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)
        self.lrn1 = LocalResponseNormalization()
        self.conv2 = layers.Conv2D(filters=self.conv_feature[1], kernel_size=self.conv_size[1], padding='same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)
        self.maxpool2 = layers.MaxPooling2D(pool_size=self.maxpool_size[1], strides=self.maxpool_stride[1], padding='same')
        self.lrn2 = LocalResponseNormalization()
        self.conv3 = layers.Conv2D(filters=self.conv_feature[2], kernel_size=self.conv_size[2], padding='same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)
        self.maxpool3 = layers.MaxPooling2D(pool_size=self.maxpool_size[2], strides=self.maxpool_stride[2], padding='same')
        self.lrn3 = LocalResponseNormalization()
        self.conv4 = layers.Conv2D(filters=self.conv_feature[3], kernel_size=self.conv_size[3], padding='same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)
        self.maxpool4 = layers.MaxPooling2D(pool_size=self.maxpool_size[3], strides=self.maxpool_stride[3], padding='same')
        self.lrn4 = LocalResponseNormalization()

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(hidden_dim * 2, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(hidden_dim, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)
        self.dropout2 = layers.Dropout(0.5)
        self.out = layers.Dense(num_tags, activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)

        if optimizer == 'Momentum':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.lrn1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.lrn2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.lrn3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.lrn4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        logits = self.out(x)
        return logits

# 创建模型并加载权重
model = CNNModel(num_tags=num_class, hidden_dim=1024, optimizer='Adam', learning_rate=0.001)
model.build(input_shape=(None, default_height, default_width, channel))  # 构建模型以初始化权重
model.load_weights(model_path)  # 加载 .h5 文件中的权重

# 定义预测函数
@tf.function
def predict_logits(inputs, training=False):
    return model(inputs, training=training)

def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        cropped = image_[2:45, :]
        rsz_img.append(np.reshape(cv2.resize(cropped, (default_height, default_width)), [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        flipped = cv2.flip(image_, 1)
        rsz_img.append(np.reshape(cv2.resize(flipped, (default_height, default_width)), [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs

def produce_result(images_):
    images_ = np.multiply(np.array(images_), 1. / 255)  # 归一化
    if use_advanced_method:
        rsz_imgs = advance_image(images_)
    else:
        rsz_imgs = [images_]
    pred_logits_ = []
    for rsz_img in rsz_imgs:
        logits = predict_logits(rsz_img, training=False)
        pred_logits_.append(tf.nn.softmax(logits).numpy())
    return np.sum(pred_logits_, axis=0)

def produce_results(images_):
    results = []
    pred_logits_ = produce_result(images_)
    pred_logits_list_ = np.argmax(pred_logits_, axis=1).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)
    return results, result_decimals

def produce_confusion_matrix(images_list_, total_num_):
    total = []
    total_decimals = []
    for ii, images_ in enumerate(images_list_):
        results, result_decimals = produce_results(images_)
        total.append(results)
        total_decimals.append(result_decimals)
        print(results, ii, ":", result_decimals[ii])
        print(result_decimals)
    sum_ = 0
    for i_ in range(num_class):
        sum_ += total[i_][i_]
    print('acc: {:.3f} %'.format(sum_ * 100. / total_num_))
    print('Using ', os.path.basename(model_path))

def predict_emotion(image_):
    image_ = cv2.resize(image_, (default_height, default_width))
    image_ = np.reshape(image_, [-1, default_height, default_width, channel])
    return produce_result(image_)[0]

def face_detect(image_path, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_cascade_ = cv2.CascadeClassifier(casc_path_)
        img_ = cv2.imread(image_path)
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))

if __name__ == '__main__':
    if not confusion_matrix:
        images_path = []
        files = os.listdir(pic_path)
        for file in files:
            if file.lower().endswith('jpg') or file.lower().endswith('png'):
                images_path.append(os.path.join(pic_path, file))

        for image in images_path:
            faces, img_gray, img = face_detect(image)
            spb = img.shape
            sp = img_gray.shape
            height = sp[0]
            width = sp[1]
            size = 600
            emotion_pre_dict = {}
            face_exists = 0

            for (x, y, w, h) in faces:
                face_exists = 1
                face_img_gray = img_gray[y:y + h, x:x + w]
                results_sum = predict_emotion(face_img_gray)
                for i, emotion_pre in enumerate(results_sum):
                    emotion_pre_dict[emotion_labels[i]] = emotion_pre
                print(emotion_pre_dict)
                label = np.argmax(results_sum)
                emo = emotion_labels[int(label)]
                print('Emotion : ', emo)

                t_size = 2
                ww = int(spb[0] * t_size / 300)
                www = int((w + 10) * t_size / 100)
                www_s = int((w + 20) * t_size / 100) * 2 / 5
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            www_s, (255, 0, 255), thickness=www, lineType=1)

            if face_exists:
                cv2.namedWindow('Emotion_classifier', 0)
                cent = int((height * 1.0 / width) * size)
                cv2.resizeWindow('Emotion_classifier', size, cent)
                cv2.imshow('Emotion_classifier', img)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()

    if confusion_matrix:
        with open(csv_path, 'r') as f:
            csvr = csv.reader(f)
            header = next(csvr)
            rows = [row for row in csvr]
            val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
            tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']

        confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        test_set = tst
        total_num = len(test_set)

        for label_image_ in test_set:
            label_ = int(label_image_[0])
            image_ = np.reshape(
                np.asarray([int(p) for p in label_image_[-1].split()]),
                [default_height, default_width, 1]
            )
            confusion_images[label_].append(image_)

        produce_confusion_matrix(confusion_images.values(), total_num)