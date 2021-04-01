import numpy as np
from model_em2 import FcCvModelReFCDepth,FcCvModelReFCSpeed
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.load('/openbayes/input/input0/data_all.npy')
y_depth = np.load('/openbayes/input/input2/label_depth.npy')*4
y_class = np.load('/openbayes/input/input2/label_class.npy')
y_class_oh = tf.one_hot(y_class.astype(np.int32),depth=8,axis=-1)

model_d = FcCvModelReFCDepth()
model_d.load_weights('weights_em2/depth/weights_'+str(8901))
model_s = FcCvModelReFCSpeed()
model_s.load_weights('weights_em2/speed/weights_'+str(19901))

test_loss_depth_m = tf.keras.metrics.Mean('test_loss_depth', dtype=tf.float32)

test_loss_speed_m = tf.keras.metrics.Mean('test_loss_speed', dtype=tf.float32)
test_acc_m = tf.keras.metrics.Mean('test_acc', dtype=tf.float32)
dev_loss_speed_m = tf.keras.metrics.Mean('dev_loss_speed', dtype=tf.float32)
dev_acc_m = tf.keras.metrics.Mean('dev_acc', dtype=tf.float32)

for i in range(4):
    y_dev_depth_pred = model_d(x[(i+16)*512:(i+17)*512],training=False)
    test_loss_depth = tf.reduce_mean(tf.losses.MSE(y_dev_depth_pred, y_depth[(i+16)*512:(i+17)*512]))
    test_loss_depth_m(test_loss_depth)

for i in range(4):
    y_test_speed_pred = model_s(x[(i+16)*512:(i+17)*512],training=False)
    test_loss_speed = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test_speed_pred, y_class_oh[(i+17)*512:(i+17)*512]))
    test_loss_speed_m(test_loss_speed)
    y_test_speed_pred_ = tf.argmax(y_test_speed_pred,axis=-1)
    test_acc = tf.reduce_mean(tf.cast(y_test_speed_pred_ == tf.constant(y_class[(i+16)*512:(i+17)*512],dtype=tf.int64),dtype=tf.float32))
    test_acc_m(test_acc)

# 测试集loss
test_loss_depth_m.result()
test_loss_speed_m.result(),test_acc_m.result()


# 还原第8192预测地层
y_test_depth_pred = model_d(x[8192:8192+1],training=False)
y_test_speed_pred = model_s(x[8192:8192+1],training=False)

speed_list = [1500,1725,2000,2250,2500,2750,3000,3250]

layer = np.zeros((256,256))
num = 123
for i in range(256):
    layer[0:int(y_test_depth_pred[0,0,i]),i] = speed_list[int(y_test_speed_pred[num,0])]
    layer[int(y_test_depth_pred[0,0,i]):int(y_test_depth_pred[num,1,i]),i] = speed_list[int(y_test_speed_pred[num,1])]
    layer[int(y_test_depth_pred[0,1,i]):int(y_test_depth_pred[num,2,i]),i] = speed_list[int(y_test_speed_pred[num,2])]
    layer[int(y_test_depth_pred[0,2,i]):,i] = speed_list[int(y_test_speed_pred[num,3])]


plt.title('还原后地层',y=1.16)
plt.xlabel("距离（m）")
plt.ylabel('深度（m）')
plt.xticks([0,50,100,150,200,250],[0,200,400,600,800,1000])
plt.yticks([0,50,100,150,200,250],[0,200,400,600,800,1000])
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.imshow(layer)
cax = plt.axes([0.265,0.03,0.5,0.05])
cbar = plt.colorbar(cax=cax,orientation='horizontal')
plt.show()