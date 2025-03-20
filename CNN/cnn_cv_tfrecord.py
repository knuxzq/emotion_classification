import os
import numpy as np
import tensorflow as tf

# ------------------------
# 路径相关定义（新）
# ------------------------
csv_path = r"A:\projectcode\emotiondect\data\fer2013\fer2013.csv"  # fer2013.csv 文件路径

data_folder_name = r"A:\projectcode\emotiondect\emotion_classifier-master\emotion_classifier_tensorflow_version\temp"
data_path_name = "cv"
pic_path_name = "pic"

# TFRecord 文件名称
record_name_train = "fer2013_train.tfrecord"
record_name_test = "fer2013_test.tfrecord"
record_name_eval = "fer2013_eval.tfrecord"

# TFRecord 文件完整路径
record_path_train = os.path.join(data_folder_name, data_path_name, record_name_train)
record_path_test = os.path.join(data_folder_name, data_path_name, record_name_test)
record_path_eval = os.path.join(data_folder_name, data_path_name, record_name_eval)

# 其他文件名保持不变
save_ckpt_name = "cnn_emotion_classifier_tf2.h5"
model_log_name = "model_log.txt"
tensorboard_name = "tensorboard"

tensorboard_path = os.path.join(data_folder_name, data_path_name, tensorboard_name)
model_log_path = os.path.join(data_folder_name, data_path_name, model_log_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)

# ------------------------
# 其他参数
# ------------------------
channel = 1  # 图像通道数
default_height = 48
default_width = 48
batch_size = 256  # 批尺寸
test_batch_size = 256  # 测试时的批尺寸
shuffle_pool_size = 4000
generations = 5000  # 总迭代数
save_flag = True  # 是否保存模型
retrain = False  # 是否要继续之前的训练


# ------------------------
# 数据增强函数
# ------------------------
def pre_process_img(image):
    """对图像进行随机的数据增强, 并缩放回 48×48 大小。"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # 随机裁剪后再 resize 回 48x48
    rand_h = tf.random.uniform([], 0, 4, dtype=tf.int32)
    rand_w = tf.random.uniform([], 0, 4, dtype=tf.int32)
    new_height = default_height - rand_h
    new_width = default_width - rand_w
    image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
    image = tf.image.resize(image, [default_height, default_width])
    return image


# ------------------------
# TFRecord 的数据解析函数 (TF2.x)
# ------------------------
def parse_function_csv(serial_exmp_):
    """从 TFRecord 中读取并解析单条样本。"""
    features_ = tf.io.parse_single_example(
        serial_exmp_,
        features={
            "image/label": tf.io.FixedLenFeature([], tf.int64),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/raw": tf.io.FixedLenFeature([default_width * default_height * channel], tf.int64)
        }
    )
    label_ = tf.cast(features_["image/label"], tf.int32)
    height_ = tf.cast(features_["image/height"], tf.int32)
    width_ = tf.cast(features_["image/width"], tf.int32)
    image_ = tf.cast(features_["image/raw"], tf.float32)
    image_ = tf.reshape(image_, [height_, width_, channel])
    image_ = image_ / 255.0  # 缩放到 [0,1] 范围

    # 数据增强
    image_ = pre_process_img(image_)
    return image_, label_


def get_dataset(record_path_, batch_size_, shuffle_size=4000, repeat=True):
    """基于 TFRecord 文件构建 tf.data.Dataset。"""
    dataset_ = tf.data.TFRecordDataset(record_path_)
    dataset_ = dataset_.map(parse_function_csv, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle_size > 0:
        dataset_ = dataset_.shuffle(shuffle_size)
    if repeat:
        dataset_ = dataset_.repeat()

    dataset_ = dataset_.batch(batch_size_).prefetch(tf.data.AUTOTUNE)
    return dataset_


# ------------------------
# 评估准确度
# ------------------------
def evaluate(y_pred, y_true):
    """评估准确度: y_pred, y_true 都是 NumPy array 或 Tensor皆可。"""
    y_pred_label = tf.argmax(y_pred, axis=1)
    correct = tf.equal(tf.cast(y_pred_label, tf.int32), tf.cast(y_true, tf.int32))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# ------------------------
# 简化的 CNN 模型示例 (可用你自己的 cnn.py 进行替换)
# ------------------------
class CNN_Model(tf.keras.Model):
    """简单示例：使用 Keras 搭建的情感识别 CNN."""

    def __init__(self, num_classes=7):
        super(CNN_Model, self).__init__()
        # 这里可根据需要自行替换层结构
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.output_layer(x)  # logits
        return x


# ------------------------
# 自定义的训练主函数 (TF2.x)
# ------------------------
def main():
    # 1. 准备数据集
    train_ds = get_dataset(record_path_train, batch_size, shuffle_pool_size, repeat=True)
    test_ds = get_dataset(record_path_test, test_batch_size, shuffle_pool_size, repeat=True)

    # 2. 构建模型与优化器、损失函数
    cnn_model = CNN_Model(num_classes=7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 3. 如果需要从已有检查点恢复
    if retrain:
        if os.path.exists(os.path.join(data_folder_name, data_path_name, save_ckpt_name)):
            print("Retraining from existing checkpoint...")
            cnn_model.load_weights(os.path.join(data_folder_name, data_path_name, save_ckpt_name))

    # 4. TensorBoard 日志
    summary_writer = tf.summary.create_file_writer(tensorboard_path)

    # 5. 自定义训练循环
    print("Start training...")
    max_accuracy = 0.0
    temp_train_loss, temp_test_loss = [], []
    temp_train_acc, temp_test_acc = [], []

    train_iter = iter(train_ds)
    test_iter = iter(test_ds)

    # 这里用一个简单的 for 循环模拟训练 generations 次，每次获取 batch
    for step in range(generations):
        # 从 Dataset 中读取一个批次
        x_batch, y_batch = next(train_iter)

        # 前向 + 反向传播
        with tf.GradientTape() as tape:
            logits = cnn_model(x_batch, training=True)  # training=True 表示启用 dropout
            train_loss = loss_fn(y_batch, logits)
        grads = tape.gradient(train_loss, cnn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))

        # 每隔 100 iteration 打印一次训练集上的损失和准确率
        if (step + 1) % 100 == 0:
            acc_value = evaluate(logits, y_batch)
            acc_value = acc_value.numpy()
            print(f"Generation # {step + 1}. Train Loss: {train_loss.numpy():.3f}, Train Acc: {acc_value:.3f}")
            temp_train_loss.append(train_loss.numpy())
            temp_train_acc.append(acc_value)

            # 写入 TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("Training Loss", train_loss, step=step)
                tf.summary.scalar("Training Accuracy", acc_value, step=step)

        # 每隔 400 iteration 在测试集上进行验证
        if (step + 1) % 400 == 0:
            test_x_batch, test_y_batch = next(test_iter)
            test_logits = cnn_model(test_x_batch, training=False)
            test_loss_value = loss_fn(test_y_batch, test_logits)
            test_accuracy_value = evaluate(test_logits, test_y_batch).numpy()
            print(
                f"Generation # {step + 1}. Test Loss: {test_loss_value.numpy():.3f}, Test Acc: {test_accuracy_value:.3f}")
            temp_test_loss.append(test_loss_value.numpy())
            temp_test_acc.append(test_accuracy_value)

            # 若准确率较高，保存模型（可根据需要修改条件）
            if test_accuracy_value >= max_accuracy and save_flag and step > generations // 2:
                max_accuracy = test_accuracy_value
                cnn_model.save_weights(os.path.join(data_folder_name, data_path_name, save_ckpt_name))
                print(f"Generation # {step + 1}. --model saved--")

    print("Training finished. Last max test accuracy:", max_accuracy)
    # 6. 将训练和测试的损失及准确率写入日志文件
    with open(model_log_path, 'w') as f:
        f.write('train_loss: ' + str(temp_train_loss))
        f.write('\n\ntest_loss: ' + str(temp_test_loss))
        f.write('\n\ntrain_acc: ' + str(temp_train_acc))
        f.write('\n\ntest_acc: ' + str(temp_test_acc))
    print('Log saved to:', model_log_path)


if __name__ == '__main__':
    main()
