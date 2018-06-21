import tensorflow as tf

from deepvo.networks.flownet.flownet import FlownetS
from deepvo.networks.layers.se3 import SE3CompositeLayer


class DeepVo:
    LSTM_NUM = 2
    LSTM_HIDDEN_SIZE = 1024

    def __init__(self):
        self.feature = FlownetS()

    def build(self, unstacked_input_ab):
        batch_size = unstacked_input_ab[0].shape[0]

        rnn_layers = [tf.nn.rnn_cell.LSTMCell(DeepVo.LSTM_HIDDEN_SIZE) for _ in range(DeepVo.LSTM_NUM)]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        rnn_inputs = []
        for input_ab in unstacked_input_ab:
            features = self.feature.build(input_ab)
            feature = features['conv6']

            rnn_inputs.append(feature)

        rnn_inputs = [tf.contrib.layers.flatten(rnn_inputs[i], [-1, ]) for i in range(len(rnn_inputs))]

        lstm_outputs, lstm_state = tf.nn.static_rnn(cell=multi_rnn_cell,
                                                    inputs=rnn_inputs,
                                                    dtype=tf.float32)

        # motion
        xs = [tf.layers.dense(o, 128, activation=tf.nn.relu) for o in lstm_outputs]
        du = [tf.layers.dense(x, 6, activation=tf.nn.relu) for x in xs]

        # pose : SE(3)
        init_s = tf.placeholder_with_default(tf.eye(4, batch_shape=[batch_size]), shape=(batch_size, 4, 4))
        l_se3comp = SE3CompositeLayer()
        se3_outputs, se3_state = tf.nn.static_rnn(cell=l_se3comp,
                                                  inputs=du,
                                                  initial_state=init_s,
                                                  dtype=tf.float32)

        r = {
            'state': {
                'lstm': lstm_state,
                'se3': se3_state
            },
            'output': {
                'motion': du,
                'pose': se3_outputs
            }
        }
        return r


if __name__ == '__main__':
    input_data = tf.placeholder(tf.float32, [1, 10, 512, 640, 6])   # [batch, time, height, width, channel]
    input_ = tf.unstack(input_data, 10, 1)

    net = DeepVo()
    net.build(input_)

    pass