from model_em3 import FcCvModelFCPos
import tensorflow as tf
import numpy as np

# 1024*33*10240
data = np.load("/openbayes/input/input0/x.npy")
labels = np.load('/openbayes/input/input1/labels.npy')

test_data = data[896:1024]
test_y = labels[896:1024]

model = FcCvModelFCPos()
model.load_weights('weights_em3/weights_'+str(13000))

# 预测
test_y_pred = model(test_data)
