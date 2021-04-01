import tensorflow as tf
from tensorflow.keras import optimizers
from model_em4 import FcCvModelFCPos
import numpy as np

data = np.load("/openbayes/input/input0/x.npy")
labels = np.load('/openbayes/input/input1/labels.npy')

test_data = data[9216:10240]
test_y = labels[9216:10240]

model = FcCvModelFCPos()
model.load_weights('weights_em4/weights_'+str(1100))

# 预测坐标x,y半径r
y_test_pred = model(test_data)
