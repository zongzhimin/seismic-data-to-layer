from model_em1 import FcCvModelV
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np

optimizer = optimizers.Adam(lr=4e-5)
epochs = 3000
model = FcCvModelV()

loss_log_path = './tf_dir/loss_all_Fcov_xiangxiebeixie_1362_2'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)
train_loss_m = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
dev_loss_m = tf.keras.metrics.Mean('dev_loss', dtype=tf.float32)

# 数据集整理
train_x_1 = np.concatenate([np.load('/openbayes/input/input1/x_train_512_' + str(1) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(2) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(3) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(4) + '.npy')], axis=0)

train_x_2 = np.concatenate([np.load('/openbayes/input/input1/x_train_512_' + str(5) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(6) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(7) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(8) + '.npy')], axis=0)
train_x_3 = np.concatenate([np.load('/openbayes/input/input1/x_train_512_' + str(9) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(10) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(11) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(12) + '.npy')], axis=0)
train_x_4 = np.concatenate([np.load('/openbayes/input/input1/x_train_512_' + str(13) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(14) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(15) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(16) + '.npy')], axis=0)
train_x_5 = np.concatenate([np.load('/openbayes/input/input1/x_train_512_' + str(17) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(18) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(19) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(20) + '.npy')], axis=0)
train_x_6 = np.concatenate([np.load('/openbayes/input/input1/x_train_512_' + str(21) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(22) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(23) + '.npy'),
                            np.load('/openbayes/input/input1/x_train_512_' + str(24) + '.npy')], axis=0)

train_x = [train_x_1, train_x_2, train_x_3, train_x_4, train_x_5, train_x_6]

train_y_1 = 1000 * np.concatenate([np.load('/openbayes/input/input0/x_train_s_' + str(1) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(2) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(3) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(4) + '.npy')], axis=0)

train_y_2 = 1000 * np.concatenate([np.load('/openbayes/input/input0/x_train_s_' + str(5) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(6) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(7) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(8) + '.npy')], axis=0)

train_y_3 = 1000 * np.concatenate([np.load('/openbayes/input/input0/x_train_s_' + str(9) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(10) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(11) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(12) + '.npy')], axis=0)

train_y_4 = 1000 * np.concatenate([np.load('/openbayes/input/input0/x_train_s_' + str(13) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(14) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(15) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(16) + '.npy')], axis=0)

train_y_5 = 1000 * np.concatenate([np.load('/openbayes/input/input0/x_train_s_' + str(17) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(18) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(19) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(20) + '.npy')], axis=0)

train_y_6 = 1000 * np.concatenate([np.load('/openbayes/input/input0/x_train_s_' + str(21) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(22) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(23) + '.npy'),
                                   np.load('/openbayes/input/input0/x_train_s_' + str(24) + '.npy')], axis=0)
train_y = [train_y_1, train_y_2, train_y_3, train_y_4, train_y_5, train_y_6]

dev_x = np.concatenate([np.load('/openbayes/input/input1/x_dev_512_' + str(1) + '.npy'),
                        np.load('/openbayes/input/input1/x_dev_512_' + str(2) + '.npy'),
                        np.load('/openbayes/input/input1/x_dev_512_' + str(3) + '.npy'),
                        np.load('/openbayes/input/input1/x_dev_512_' + str(4) + '.npy')]
                       , axis=0)
dev_y = 1000.0 * np.concatenate([np.load('/openbayes/input/input0/x_dev_s_' + str(1) + '.npy'),
                                 np.load('/openbayes/input/input0/x_dev_s_' + str(2) + '.npy'),
                                 np.load('/openbayes/input/input0/x_dev_s_' + str(3) + '.npy'),
                                 np.load('/openbayes/input/input0/x_dev_s_' + str(4) + '.npy')]
                                , axis=0)


# 训练
for e in range(epochs):
    for i in range(6):
        with tf.GradientTape() as tape:
            y_pred = model(train_x[i])
            loss = tf.reduce_mean(tf.losses.MSE(y_pred, train_y[i]))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_m(loss)
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss_m.result(), step=e)
    train_loss_m.reset_states()

    for i in range(1):
        y_dev_pred = model(dev_x)
        dev_loss = tf.reduce_mean(tf.losses.MSE(y_dev_pred, dev_y))
        dev_loss_m(dev_loss)
    with loss_summary_writer.as_default():
        tf.summary.scalar('dev_loss', dev_loss_m.result(), step=e)
    dev_loss_m.reset_states()
    model.save_weights('weights_em1/weights_'+str(e+1))
