
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class mygru( RNNCell ):
    def __init__( self, num_units ):
        self.num_units = num_units
        pass
 
    @property
    def state_size(self):
        return self.num_units
 
    @property
    def output_size(self):
        return self.num_units
 
    def __call__( self, inputs, state, scope=None ):
        init = tf.contrib.layers.variance_scaling_initializer()
        xt = inputs
        h_minusone = state
        Wz = tf.get_variable("W_z", [self.state_size(), inputs.get_shape()[1]], initializer = init)
        Wh = tf.get_variable("W_h", [self.state_size(), inputs.get_shape()[1]], initializer = init)
        Wr = tf.get_variable("W_r", [self.state_size(), inputs.get_shape()[1]], initializer = init)
        Uz = tf.get_variable("U_z", [self.state_size(), self.state_size()], initializer = init)
        Ur = tf.get_variable("U_r", [self.state_size(), self.state_size()], initializer = init)
        bz = tf.get_variable("b_z", [self.state_size()], initializer = init)
        br = tf.get_variable("b_r", [self.state_size()], initializer = init)
        zt = tf.sigmoid(sigmag, tf.nn.bias_add((tf.matmul(Wz, xt) + tf.matmul(Uz, h_minusone) , bz)))
        rt = tf.sigmoid(sigmag, tf.nn.bias_add((tf.matmul(Wr, xt) + tf.matmul(Ur, h_minusone) + br)))
        h_twid = tf.tanh(tf.nn.bias_add((tf.matmul(Wh, xt) + tf.matmul(Uh, tf.tensordot(rt, h_minusone))), bh))
        ht = tf.tensordot(zt,h_minusone) + tf.tensordot((1-zt),  h_twid)
        pass
