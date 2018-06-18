import tensorflow as tf

from deepvo.networks.flownet.flownet import FlownetS


class DeepVo:
    LSTM_NUM = 2
    LSTM_HIDDEN_SIZE = 1024

    def __init__(self):
        self.feature = FlownetS()

    def build(self, unstacked_input_ab):
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(DeepVo.LSTM_HIDDEN_SIZE) for _ in range(DeepVo.LSTM_NUM)]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        rnn_inputs = []
        for input_ab in unstacked_input_ab:
            features = self.feature.build(input_ab)
            feature = features['conv6']

            rnn_inputs.append(feature)

        rnn_inputs = [tf.contrib.layers.flatten(rnn_inputs[i], [-1, ]) for i in range(len(rnn_inputs))]

        outputs, state = tf.nn.static_rnn(cell=multi_rnn_cell,
                                          inputs=rnn_inputs,
                                          dtype=tf.float32)

        # motion
        xs = [tf.layers.dense(o, 128, activation=tf.nn.relu) for o in outputs]
        du = [tf.layers.dense(x, 12, activation=tf.nn.relu) for x in xs]

        # pose : SE(3)

        return du


if __name__ == '__main__':
    input_data = tf.placeholder(tf.float32, [1, 10, 512, 640, 6])
    input_ = tf.unstack(input_data, 10, 1)

    net = DeepVo()
    net.build(input_)

    pass