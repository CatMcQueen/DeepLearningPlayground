import tensorflow as tf
import numpy as np
from matplotlib import image
import os
from scipy.misc import imread, imresize
from skimage import io, transform
import matplotlib.pyplot as plt
from tqdm import tqdm

#collect file names to import (they are the same for input and output)
in_files = os.listdir('final_proj/input_images')
len_files = len(in_files)
in_data, out_data = [], []
input_y = 512

# get all the files
for inp in in_files:
    in_img = np.load('final_proj/input_images/{}'.format(inp))
    in_data.append(imresize(in_img, (input_y, input_y)))
    name = inp.split('.')[0]
    out_data.append(transform.resize(io.imread('final_proj/output_images/{}.png'.format(name)), (512,512,1), mode='constant'))
    

out_data = np.array(out_data)
in_data = np.array(in_data)
in_x = in_data.shape[1]
in_y = in_data.shape[2]
in_data = np.reshape(in_data, (in_data.shape[0], in_x, in_y, 1))

out_x = out_data.shape[1]
out_y = out_data.shape[2]

#set batch size
batch = 2
c = .001 # learning rate


#here is the computation graph
with tf.Graph().as_default():
    input_data = tf.placeholder( tf.float32, [batch, 512, 512, 1] )
    input_label = tf.placeholder(tf.int64, [batch, 512, 512, 1])
    #keep_prob = tf.placeholder(tf.float32)
    
    
    #basic convolution... 
    #def comp_gr1(x_func, name, num_filt = 2):
    with tf.name_scope("comp_graph") as scope:
        kernel_init = kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        c0 = tf.layers.conv2d(input_data, 64, 3, 1,  padding = "same", activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c0")
        c1 = tf.layers.conv2d(c0, 32, 3, 1, padding = "same",activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c1")
        c2 = tf.layers.conv2d(c1, 32, 3, 1 , padding = "same",activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c2")
        c3 = tf.layers.conv2d(c2, 32, 3, 1 , padding = "same",activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c3")
        c4 = tf.layers.conv2d(c3, 2, 3, 1 , padding = "same",kernel_initializer = kernel_init,  name = "c4")
        

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(input_label,(batch,512,512)), logits=c4)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=c).minimize(loss)
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    writer = tf.summary.FileWriter("./tf_lab_final", sess.graph)
    sess.run(init)
   
    epochs =1000 # 12*len_files
    #epochs = 1
    for ep in xrange(epochs):
        
        #set up batching
        input_coll = []
        output_coll = []
        for m in xrange(batch):
            ind = np.random.randint(len_files)
            input_coll.append(in_data[ind])
            output_coll.append(out_data[ind])
        input_images = np.reshape(input_coll,(batch,512,512,1))
        output_images = np.reshape(output_coll,(batch,512,512,1))
        
        #run the test
        loss_in = sess.run(loss, feed_dict = {input_data:input_images,input_label:output_images})
        _ ,test_out  = sess.run( [train_step, c4], feed_dict = {input_data:input_images,input_label:output_images})
        if ep % 200 == 0:
            print  ep, ": ", loss_in
            #print test_out.shape
            testOut = test_out[0,:,:,0]
            plt.imshow(testOut)
            image.imsave('test_img_{}.png'.format(ep), testOut)






