import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers


# 一维卷积后，再全连接获得x,y,r
class FcCvModelFCPos(Model):
    def __init__(self):
        super(FcCvModelFCPos, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        # 1.减少过拟合的情况
        # 2.模拟道数据的丢失
        self.dropout_fc = layers.Dropout(0.3)

        # 1维卷积综合提取特征
        self.conv1 = Sequential([
            layers.Conv1D(filters=64, kernel_size=9, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c1 = layers.Dropout(0.3)
        self.conv2 = Sequential([
            layers.Conv1D(filters=64, kernel_size=9, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c2 = layers.Dropout(0.3)
        # 1维卷积综合提取特征+降维
        self.conv3 = Sequential([
            layers.Conv1D(filters=128, kernel_size=4, strides=4, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c3 = layers.Dropout(0.3)
        # 1维卷积综合提取特征
        self.conv4 = Sequential([
            layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c4 = layers.Dropout(0.3)
        self.conv5 = Sequential([
            layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c5 = layers.Dropout(0.3)
        # 1维卷积综合提取特征+降维
        self.conv6 = Sequential([
            layers.Conv1D(filters=256, kernel_size=4, strides=4, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        # 最终作出异常体坐标与半径的预测
        self.fcThree = Sequential([
            layers.Dense(3)
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x = self.fcOne(x)
        x = tf.reshape(x, (batchSize, dao_num, -1))
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.dropout_fc(x, training=training)
        x = self.conv1(x)
        x = self.dropout_c1(x, training=training)
        x = self.conv2(x)
        x = self.dropout_c2(x, training=training)
        x = self.conv3(x)
        x = self.dropout_c3(x, training=training)
        x = self.conv4(x)
        x = self.dropout_c4(x, training=training)
        x = self.conv5(x)
        x = self.dropout_c5(x, training=training)
        x = self.conv6(x)
        x = tf.reshape(x, (batchSize, -1))
        x = self.fcThree(x)

        return x
