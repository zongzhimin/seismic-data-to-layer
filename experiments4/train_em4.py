import tensorflow as tf
from tensorflow.keras import optimizers
from model_em4 import FcCvModelFCPos
import numpy as np

# 1024*33*10240
data = np.load("/openbayes/input/input0/x.npy")
labels = np.load('/openbayes/input/input1/labels.npy')[0:10240]

# # 打乱数据
# def data_shuffle(x, y):
#     index = np.random.permutation(x.shape[0])
#     return x[index, :, :], y[index, :]
# data,labels = data_shuffle(data,labels)

train_data = data[0:8192]
train_y = labels[0:8192]
dev_data = data[8192:9216]
dev_y = labels[8192:9216]

lr = 4e-6
epochs = 10000
optimizer = optimizers.Adam(learning_rate=lr)

loss_log_path = './tf_dir/loss_all_dropout_marmousi_end3+'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

model = FcCvModelFCPos()

for epoch in range(epochs):
    for step in range(2):
        with tf.GradientTape() as tape:
            y_pred = model(train_data[step * 4096:(step + 1) * 4096], training=True)
            loss = tf.reduce_mean(tf.losses.MSE(train_y[step * 4096:(step + 1) * 4096], y_pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
    train_loss.reset_states()

    if epoch % 50 == 0:
        y_dev_pred = model(dev_data, training=False)
        dev_loss = tf.reduce_mean(tf.losses.MSE(dev_y, y_dev_pred))
        with loss_summary_writer.as_default():
            tf.summary.scalar('dev_loss', dev_loss, step=epoch)
    if (epoch + 1) % 100 == 0:
        model.save_weights('weights_dropout_marmousi_ksize_end2_4600_2652/weights_' + str(epoch + 1))
