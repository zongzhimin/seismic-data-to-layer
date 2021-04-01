import numpy as np
from tensorflow.keras import Sequential, layers, Model
import tensorflow as tf

# 一维卷积后，再全连接获得三分界面深度预测
class FcCvModelReFCDepth(Model):
    def __init__(self):
        super(FcCvModelReFCDepth, self).__init__()
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
#         self.dropout_c6 = layers.Dropout(0.3)
        # 最终作出的预测
        # 深度预测
        self.fcThree_d1 = Sequential([
            layers.Dense(256)
        ])

        self.fcThree_d2 = Sequential([
            layers.Dense(256)
        ])

        self.fcThree_d3 = Sequential([
            layers.Dense(256)
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
        x_d1 = self.fcThree_d1(x)
        x_d2 = self.fcThree_d2(x)
        x_d3 = self.fcThree_d3(x)
        x_d = tf.concat([tf.expand_dims(x_d1,axis=1),tf.expand_dims(x_d2,axis=1),tf.expand_dims(x_d3,axis=1)],axis=1)
        return x_d


# 一维卷积后，再全连接获得四层速度预测
class FcCvModelReFCSpeed(Model):
    def __init__(self):
        super(FcCvModelReFCSpeed, self).__init__()
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

        # 最终作出的预测
        # 速度预测
        self.fcThree_s1 = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('softmax')
        ])
        self.fcThree_s2 = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('softmax')
        ])
        self.fcThree_s3 = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('softmax')
        ])
        self.fcThree_s4 = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('softmax')
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
        x_s1 = self.fcThree_s1(x)
        x_s2 = self.fcThree_s2(x)
        x_s3 = self.fcThree_s3(x)
        x_s4 = self.fcThree_s4(x)
        x_s = tf.concat([tf.expand_dims(x_s1, axis=1), tf.expand_dims(x_s2, axis=1), tf.expand_dims(x_s3, axis=1),
                         tf.expand_dims(x_s4, axis=1)], axis=1)
        return x_s
