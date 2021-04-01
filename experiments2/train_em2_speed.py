import numpy as np
from tensorflow.keras import optimizers
import tensorflow as tf
from model_em2 import FcCvModelReFCSpeed

loss_log_path = './tf_dir/loss_all_layer3_fc_speed_2'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)

train_loss_speed_m = tf.keras.metrics.Mean('train_loss_speed', dtype=tf.float32)
train_acc_m = tf.keras.metrics.Mean('train_acc', dtype=tf.float32)
dev_loss_speed_m = tf.keras.metrics.Mean('dev_loss_speed', dtype=tf.float32)
dev_acc_m = tf.keras.metrics.Mean('dev_acc', dtype=tf.float32)

x = np.load('/openbayes/input/input0/data_all.npy')
y_class = np.load('/openbayes/input/input2/label_class.npy')
y_class_oh = tf.one_hot(y_class.astype(np.int32),depth=8,axis=-1)

optimizer = optimizers.Adam(lr=4e-5)
epochs = 20000
model = FcCvModelReFCSpeed()

for e in range(epochs):
    for i in range(12):
        with tf.GradientTape() as tape:
            y_speed_pred = model(x[i*512:(i+1)*512],training=True)
            train_loss_speed = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_speed_pred, y_class_oh[i*512:(i+1)*512]))
        grads = tape.gradient(train_loss_speed, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_speed_m(train_loss_speed)
        y_speed_pred_ = tf.argmax(y_speed_pred,axis=-1)
        train_acc = tf.reduce_mean(tf.cast(y_speed_pred_ == tf.constant(y_class[i*512:(i+1)*512],dtype=tf.int64),dtype=tf.float32))
        train_acc_m(train_acc)
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss_speed', train_loss_speed_m.result(), step=e)
        tf.summary.scalar('train_acc', train_acc_m.result(), step=e)
    train_loss_speed_m.reset_states()
    train_acc_m.reset_states()
    if e%100 == 0:
        for i in range(4):
            y_dev_speed_pred = model(x[(i+12)*512:(i+13)*512],training=False)
            dev_loss_speed = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_dev_speed_pred, y_class_oh[(i+12)*512:(i+13)*512]))
            dev_loss_speed_m(dev_loss_speed)
            y_dev_speed_pred_ = tf.argmax(y_dev_speed_pred,axis=-1)
            dev_acc = tf.reduce_mean(tf.cast(y_dev_speed_pred_ == tf.constant(y_class[(i+12)*512:(i+13)*512],dtype=tf.int64),dtype=tf.float32))
            dev_acc_m(dev_acc)
        with loss_summary_writer.as_default():
            tf.summary.scalar('dev_loss_speed', dev_loss_speed_m.result(), step=e)
            tf.summary.scalar('dev_acc', dev_acc_m.result(), step=e)
        dev_loss_speed_m.reset_states()
        dev_acc_m.reset_states()
        model.save_weights('weights_em2/weights_'+str(e+1))