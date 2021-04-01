import numpy as np
from tensorflow.keras import optimizers
import tensorflow as tf
from model_em2 import FcCvModelReFCDepth

# loss记录
loss_log_path = './tf_dir/loss_all_layer3_fc_depth_4'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)
train_loss_depth_m = tf.keras.metrics.Mean('train_loss_depth', dtype=tf.float32)
dev_loss_depth_m = tf.keras.metrics.Mean('dev_loss_depth', dtype=tf.float32)

# 数据加载
x = np.load('/openbayes/input/input0/data_all.npy')
y_depth = np.load('/openbayes/input/input2/label_depth.npy')*4

optimizer = optimizers.Adam(lr=1e-4)
epochs = 20000
model = FcCvModelReFCDepth()

# 训练
for e in range(epochs):
    for i in range(12):
        with tf.GradientTape() as tape:
            y_depth_pred = model(x[i*512:(i+1)*512],training=True)
            train_loss_depth = tf.reduce_mean(tf.losses.MSE(y_depth_pred, y_depth[i*512:(i+1)*512]))
        grads = tape.gradient(train_loss_depth, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_depth_m(train_loss_depth)
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss_depth', train_loss_depth_m.result(), step=e)
    train_loss_depth_m.reset_states()
    if e%100 == 0:
        for i in range(4):
            y_dev_depth_pred = model(x[(i+12)*512:(i+13)*512],training=False)
            dev_loss_depth = tf.reduce_mean(tf.losses.MSE(y_dev_depth_pred, y_depth[(i+12)*512:(i+13)*512]))
            dev_loss_depth_m(dev_loss_depth)
        with loss_summary_writer.as_default():
            tf.summary.scalar('dev_loss_depth', dev_loss_depth_m.result(), step=e)
        dev_loss_depth_m.reset_states()
        model.save_weights('weights_fc_depth_4/weights_'+str(e+1))
