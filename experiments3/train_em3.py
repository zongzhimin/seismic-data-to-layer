from model_em3 import FcCvModelFCPos
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np

lr = 5e-4
epochs = 13000
optimizer = optimizers.Adam(learning_rate=lr)

# 1024*33*10240
data = np.load("/openbayes/input/input0/x.npy")
labels = np.load('/openbayes/input/input1/labels.npy')

# test1：1024的数据量
train_data = data[0:768]
train_y = labels[0:768]
dev_data = data[768:896]
dev_y = labels[768:896]
test_data = data[896:1024]
test_y = labels[896:1024]

loss_log_path = './tf_dir/loss_all_dropout_water_2'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)

model = FcCvModelFCPos()

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(train_data, training=True)
        loss = tf.reduce_mean(tf.losses.MSE(train_y, y_pred))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss', loss, step=epoch)

    y_dev_pred = model(dev_data, training=False)
    dev_loss = tf.reduce_mean(tf.losses.MSE(dev_y, y_dev_pred))
    with loss_summary_writer.as_default():
        tf.summary.scalar('dev_loss', dev_loss, step=epoch)
    if (epoch + 1) % 200 == 0:
        model.save_weights('weights_dropout_water_2/weights_' + str(epoch + 1))
