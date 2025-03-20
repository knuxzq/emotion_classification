import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers


# 如果需要用到兼容版的 LRN，可加下行，启用部分 v1 功能：
# tf.compat.v1.disable_eager_execution()  # 仅在某些场景下需要

class LocalResponseNormalization(layers.Layer):
    """
    在 TensorFlow 2.x 中已经不再提供 tf.nn.lrn。
    这里通过 tf.compat.v1.nn.lrn 或自定义实现作模拟。
    仅供演示，如无特殊需要，可省略 LRN。
    """

    def __init__(self, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        # 调用 TF 1.x 的 LRN
        return tf.compat.v1.nn.lrn(
            inputs,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta
        )


class CNNModel(tf.keras.Model):
    def __init__(self,
                 num_tags=7,
                 hidden_dim=1024,
                 optimizer='Adam',
                 learning_rate=0.001):
        """
        :param num_tags: 分类类别数（例如 7 种表情）
        :param hidden_dim: 全连接层的大小
        :param optimizer: 优化器名称，可选 'Adam'、'Momentum'、'SGD' 等
        :param learning_rate: 学习率
        """
        super(CNNModel, self).__init__()

        # 超参数
        self.num_tags = num_tags
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim

        # 替代之前 tf.contrib.layers.xavier_initializer
        self.initializer = initializers.GlorotUniform()

        # 替代之前 tf.contrib.layers.l2_regularizer
        self.l2_reg = regularizers.l2(0.001)

        # 定义网络层参数
        self.conv_feature = [32, 32, 32, 64]
        self.conv_size = [1, 5, 3, 5]
        self.maxpool_size = [None, 3, 3, 3]
        self.maxpool_stride = [None, 2, 2, 2]

        # (1) Conv1 + LRN
        self.conv1 = layers.Conv2D(
            filters=self.conv_feature[0],
            kernel_size=self.conv_size[0],
            padding='same',
            activation='relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )
        self.lrn1 = LocalResponseNormalization()

        # (2) Conv2 + MaxPool + LRN
        self.conv2 = layers.Conv2D(
            filters=self.conv_feature[1],
            kernel_size=self.conv_size[1],
            padding='same',
            activation='relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )
        self.maxpool2 = layers.MaxPooling2D(
            pool_size=self.maxpool_size[1],
            strides=self.maxpool_stride[1],
            padding='same'
        )
        self.lrn2 = LocalResponseNormalization()

        # (3) Conv3 + MaxPool + LRN
        self.conv3 = layers.Conv2D(
            filters=self.conv_feature[2],
            kernel_size=self.conv_size[2],
            padding='same',
            activation='relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )
        self.maxpool3 = layers.MaxPooling2D(
            pool_size=self.maxpool_size[2],
            strides=self.maxpool_stride[2],
            padding='same'
        )
        self.lrn3 = LocalResponseNormalization()

        # (4) Conv4 + MaxPool + LRN
        self.conv4 = layers.Conv2D(
            filters=self.conv_feature[3],
            kernel_size=self.conv_size[3],
            padding='same',
            activation='relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )
        self.maxpool4 = layers.MaxPooling2D(
            pool_size=self.maxpool_size[3],
            strides=self.maxpool_stride[3],
            padding='same'
        )
        self.lrn4 = LocalResponseNormalization()

        # 全连接层部分
        # 先不指定输入维度，等执行时自动推断
        self.flatten = layers.Flatten()

        # 第一层全连接
        self.dense1 = layers.Dense(
            hidden_dim * 2,
            activation='relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )
        self.dropout1 = layers.Dropout(0.5)

        # 第二层全连接
        self.dense2 = layers.Dense(
            hidden_dim,
            activation='relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )
        self.dropout2 = layers.Dropout(0.5)

        # 输出层
        self.out = layers.Dense(
            num_tags,
            activation=None,  # logits
            kernel_initializer=self.initializer,
            kernel_regularizer=self.l2_reg
        )

        # 根据不同优化器名称设置优化器
        if optimizer == 'Momentum':
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=0.9
            )
        elif optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate
            )
        else:
            # 默认使用 Adam
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )

    def call(self, inputs, training=False):
        """
        前向传播逻辑。
        :param inputs: 形状 [batch_size, H, W, channels]
        :param training: 是否处于训练模式（控制 dropout 等）
        :return: logits，形状 [batch_size, num_tags]
        """
        # 1. conv1 + LRN
        x = self.conv1(inputs)
        x = self.lrn1(x)

        # 2. conv2 + pool + LRN
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.lrn2(x)

        # 3. conv3 + pool + LRN
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.lrn3(x)

        # 4. conv4 + pool + LRN
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.lrn4(x)

        # 抻平
        x = self.flatten(x)

        # Dense1 + Dropout
        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        # Dense2 + Dropout
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        # 输出层（logits）
        logits = self.out(x)
        return logits


# 如果需要通过 model.compile + model.fit 方式进行训练，可如下示例：
def create_compiled_model(num_tags=7, hidden_dim=1024, optimizer='Adam', lr=0.001):
    """
    创建并编译一个 CNNModel 的 Keras 模型。
    """
    model = CNNModel(
        num_tags=num_tags,
        hidden_dim=hidden_dim,
        optimizer=optimizer,
        learning_rate=lr
    )
    # sparse_categorical_crossentropy 可直接处理整型标签
    model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # 假设有 (batch_size, 48, 48, 1) 的灰度图输入
    import numpy as np

    # 构造一个简单的示例数据：
    x_dummy = np.random.randn(8, 48, 48, 1).astype(np.float32)  # batch_size=8
    y_dummy = np.random.randint(0, 7, size=(8,))  # 7 类

    # 创建并编译模型
    model = create_compiled_model(num_tags=7, hidden_dim=1024, optimizer='Adam', lr=0.001)

    # 打印模型结构
    model.build(input_shape=(None, 48, 48, 1))
    model.summary()

    # 训练 1 步（仅演示）
    model.fit(x_dummy, y_dummy, epochs=1, batch_size=4)
