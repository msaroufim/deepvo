import os

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.framework import ops

gvnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build/libgvnn.so')
gvnn = tf.load_op_library(gvnn_path)
print(dir(gvnn))


@ops.RegisterGradient("SE3toMatrixRt")
def se3toMatrixRt_grad_cc(op, grad):
    return gvnn.se3toMatrixRt_grad(grad, op.inputs)


def layer_matrix_rt(xyzw):
    m = gvnn.se3to_matrix_rt(xyzw)
    m = tf.map_fn(lambda x: tf.concat((x, [[0, 0, 0, 1]]), axis=0), m)
    return m


def layer_xyzq(matrix_rt, scope='pose', name='xyzq'):
    # Rotation Matrix to quaternion + xyz
    with tf.variable_scope(scope):
        qw = tf.sqrt(tf.reduce_sum(tf.matrix_diag_part(matrix_rt), axis=-1)) / 2.0
        qx = (matrix_rt[:, 2, 1] - matrix_rt[:, 1, 2]) / (4 * qw)
        qy = (matrix_rt[:, 0, 2] - matrix_rt[:, 2, 0]) / (4 * qw)
        qz = (matrix_rt[:, 1, 0] - matrix_rt[:, 0, 1]) / (4 * qw)

        '''
        # See : http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = tf.reduce_sum(tf.matrix_diag_part(matrix_rt), axis=-1) - 1.0

        if trace > 0:
            S = tf.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (matrix_rt[:, 2, 1] - matrix_rt[:, 1, 2]) / S
            qy = (matrix_rt[:, 0, 2] - matrix_rt[:, 2, 0]) / S
            qz = (matrix_rt[:, 1, 0] - matrix_rt[:, 0, 1]) / S
        elif matrix_rt[:, 0, 0] > matrix_rt[:, 1, 1] and matrix_rt[:, 0, 0] > matrix_rt[:, 2, 2]:
            S = tf.sqrt(1 + matrix_rt[:, 0, 0] - matrix_rt[:, 1, 1] - matrix_rt[:, 2, 2]) * 2
            qw = (matrix_rt[:, 2, 1] - matrix_rt[:, 1, 2]) / S
            qx = 0.25 * S
            qy = (matrix_rt[:, 0, 1] + matrix_rt[:, 1, 0]) / S
            qz = (matrix_rt[:, 0, 2] + matrix_rt[:, 2, 0]) / S
        elif matrix_rt[:, 1, 1] > matrix_rt[:, 2, 2]:
            S = tf.sqrt(1 + matrix_rt[:, 1, 1] - matrix_rt[:, 0, 0] - matrix_rt[:, 2, 2]) * 2
            qw = (matrix_rt[:, 0, 2] - matrix_rt[:, 2, 0]) / S
            qx = (matrix_rt[:, 0, 1] + matrix_rt[:, 1, 0]) / S
            qy = 0.25 * S
            qz = (matrix_rt[:, 1, 2] + matrix_rt[:, 2, 1]) / S
        else:
            S = tf.sqrt(1 + matrix_rt[:, 2, 2] - matrix_rt[:, 0, 0] - matrix_rt[:, 1, 1]) * 2
            qw = (matrix_rt[:, 1, 0] - matrix_rt[:, 0, 1]) / S
            qx = (matrix_rt[:, 0, 2] + matrix_rt[:, 2, 0]) / S
            qy = (matrix_rt[:, 1, 2] + matrix_rt[:, 2, 1]) / S
            qz = 0.25 * S
        '''

        x = matrix_rt[:, 0, 3]
        y = matrix_rt[:, 1, 3]
        z = matrix_rt[:, 2, 3]

        xyzq = tf.stack([qw, qx, qy, qz, x, y, z], axis=-1, name=name)
    return xyzq


class SE3CompositeLayer(LayerRNNCell):
    def __init__(self, reuse=None, name=None):
        super(LayerRNNCell, self).__init__(_reuse=reuse, name=name)

        self._state_size = tensor_shape.TensorShape([None, 4, 4])       # Accumulated SE3 Matrix
        self._output_size = tensor_shape.TensorShape([None, 7])   # xyz + q

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        print(type(inputs_shape), inputs_shape)
        self.built = True

    def __call__(self, inputs, state, scope=None, *args, **kwargs):
        r_matrix = layer_matrix_rt(inputs)
        accu_pose = tf.matmul(state, r_matrix)
        xyzq = layer_xyzq(accu_pose)

        return xyzq, accu_pose
