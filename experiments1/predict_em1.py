from model_em1 import FcCvModelV
import tensorflow as tf
import numpy as np

model = FcCvModelV()
model.load_weights('weights_em1/weights_' + str(1798))

test_x = np.concatenate([np.load('/openbayes/input/input1/x_test_512_' + str(1) + '.npy'),
                         np.load('/openbayes/input/input1/x_test_512_' + str(2) + '.npy'),
                         np.load('/openbayes/input/input1/x_test_512_' + str(3) + '.npy'),
                         np.load('/openbayes/input/input1/x_test_512_' + str(4) + '.npy')]
                        , axis=0)
test_y = 1000.0 * np.concatenate([np.load('/openbayes/input/input0/x_test_s_' + str(1) + '.npy'),
                                  np.load('/openbayes/input/input0/x_test_s_' + str(2) + '.npy'),
                                  np.load('/openbayes/input/input0/x_test_s_' + str(3) + '.npy'),
                                  np.load('/openbayes/input/input0/x_test_s_' + str(4) + '.npy')]
                                 , axis=0)


y_test_pred = model(test_x)

test_loss = tf.reduce_mean(tf.losses.MSE(y_test_pred, test_y))