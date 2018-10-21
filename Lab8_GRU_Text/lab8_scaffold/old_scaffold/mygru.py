from tensorflow.python.ops.rnn_cell import RNNCell
 
class mygru( RNNCell ):
 
    def __init__( self, num_unit ):
	h = 0
	U = [128,128]
	W = [128,vocab_size]
    	pass
 
    @property
    def state_size(self):
    	pass
 
    @property
    def output_size(self):
    	pass
 
    def __call__( self, inputs, state, scope=None ):
    	pass
