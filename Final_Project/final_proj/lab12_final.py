import tensorflow as tf
import numpy as np
from matplotlib import image
import os
from scipy.misc import imread, imresize
from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from math import sqrt

#collect file names to import (they are the same for input and output)
in_files = os.listdir('final_proj/input_images')
len_files = len(in_files)
in_data, out_data = [], []
input_y = 512

# get all the files
for inp in in_files:
    in_img = np.load('final_proj/input_images/{}'.format(inp))
    #in_data.append(imresize(in_img, (input_y, input_y))) # What?? I think you want to resize...but not with imresize
    # maybe consider
    in_data.append(in_img[:,:512])
    name = inp.split('.')[0]
    out_data.append(transform.resize(io.imread('final_proj/output_images/{}.png'.format(name)), (512,512,1), mode='constant'))
    

out_data = np.array(out_data)
in_data = np.array(in_data)
in_x = in_data.shape[1]
in_y = in_data.shape[2]
in_data = np.reshape(in_data, (in_data.shape[0], in_x, in_y, 1))

out_x = out_data.shape[1]
out_y = out_data.shape[2]

#basic convolution... 
def comp_gr1(input_data):
    with tf.name_scope("comp_graph") as scope:
        kernel_init = kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        c0 = tf.layers.conv2d(input_data, 64, 3, 2,  padding = "same", activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c0")
        c1 = tf.layers.conv2d(c0, 32, 3, 2, padding = "same",activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c1")
        c2 = tf.layers.conv2d_transpose(c1, 32, 3, 2 , padding = "same",activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c2")
        c3 = tf.layers.conv2d_transpose(c2, 32, 3, 2 , padding = "same",activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c3")
        c4 = tf.layers.conv2d(c3, 1, 3, 1 , padding = "same",kernel_initializer = kernel_init,  name = "c4")
    return c4

def dropout_comp_gr(input_data, dropout_rate):
    with tf.name_scope("comp_graph_do") as scope:
        kernel_init =  tf.contrib.layers.variance_scaling_initializer()
        c0 = tf.layers.conv2d(input_data, 64, 3, 2,  padding = "same", activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c0_dr")
        c0_1 = tf.nn.dropout(c0, dropout_rate) 
        c1 = tf.layers.conv2d(c0_1, 32, 3, 2, padding = "same", activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c1_dr")
        c1_1 = tf.nn.dropout(c1, dropout_rate) 
        c2 = tf.layers.conv2d_transpose(c1_1, 32, 3, 2 , padding = "same", activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c2_dr")
        c2_1 = tf.nn.dropout(c2, dropout_rate)
        c3 = tf.layers.conv2d_transpose(c2_1, 32, 3, 2 , padding = "same", activation = tf.nn.relu,kernel_initializer = kernel_init, name = "c3_dr")
        c4 = tf.layers.conv2d(c3, 1, 3, 1 , padding = "same", kernel_initializer = kernel_init,  name = "c4_dr")
    return c4

def conv2d( in_var, output_dim, ksize = 3, stride = 2, is_output=False, name="conv2d" ):
    # filter width/height
    # x,y strides

    with tf.variable_scope( name):
        W = tf.get_variable( "W", [ksize, ksize, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        conv = tf.nn.conv2d( in_var, W, strides=[1, stride, stride, 1], padding='SAME' )
        return conv  

def maxpool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')  

def bn(in_var, output_dim, ksize = 3, name = "bn"):

    mean, variance = tf.nn.moments(in_var, axes=[0,1,2])
    
    with tf.variable_scope(name):
        beta = tf.get_variable( "beta", initializer=tf.zeros([output_dim]) )

        gamma = tf.get_variable( "gamma", [output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        in_var, mean, variance, beta, gamma, 0.001,
        scale_after_normalization=True)
    return batch_norm

def block(in_var, output_dim, stride):

    with tf.variable_scope( "block_conv_1" ):
        conv_1 = conv2d(in_var, output_dim, 3, stride)
        r_bn_1 = bn(conv_1, output_dim)
        r_activation_1 = tf.nn.relu(r_bn_1) 

    with tf.variable_scope( "block_conv_2" ):
        conv_2 = conv2d(r_activation_1, output_dim, 3, stride)
        r_bn_2 = bn(conv_2, output_dim)

    input_dim = int(in_var.get_shape()[-1])
    #print "input_dim", input_dim
    #print "output_dim", output_dim
    if input_dim != output_dim:
        in_var = tf.pad(in_var, [[0,0], [0,0], [0,0], [0, output_dim - input_dim]])

    return tf.nn.relu(in_var + r_bn_2)

#credit to https://github.com/mirisr

def resNet( imgs , batch_size):
    #reshaping to make compatible with convolution
    imgs = tf.reshape( imgs, [ batch_size, 512, 512, 1 ] )
    kernel_init = tf.contrib.layers.variance_scaling_initializer()

    with tf.variable_scope( "r_conv_1" ):
        r_conv_1 = conv2d(imgs, 64, 7, 2)
        r_bn_1 = bn(r_conv_1, 64)
        r_activation_1 = tf.nn.relu(r_bn_1) 

    with tf.variable_scope( "r_maxpool_1" ):
        r_maxpool_1 = maxpool(r_activation_1)

    conv = r_maxpool_1
    for i in xrange(2):#3
        with tf.variable_scope( "r_conv_64s_%d" % (i) ):
            conv = block(conv, 64, 1)
    
    for i in xrange(3):#4
        with tf.variable_scope( "r_conv_128s_%d" % (i) ):
            conv = block(conv, 128, 1)
    
    for i in xrange(3):#5
        with tf.variable_scope( "r_conv_256s_%d" % (i) ):
            conv = block(conv, 256, 1)

    for i in xrange(2):#6
        with tf.variable_scope( "r_conv_512s_%d" % (i) ):
            conv = block(conv, 512, 1)
    conv = tf.layers.conv2d_transpose(conv, 128, 3, 2, kernel_initializer= kernel_init, activation=tf.nn.relu, padding = "SAME")
    conv = tf.layers.conv2d_transpose(conv, 32, 3, 2, kernel_initializer= kernel_init, activation=tf.nn.relu, padding = "SAME")
    conv = tf.layers.conv2d_transpose(conv, 1, 3, 1, kernel_initializer= kernel_init, padding = "SAME")
    #print "conv_out", conv


    #with tf.variable_scope('fc'):
    #    avgpool = tf.reduce_mean(conv, [1, 2])
    #    fc_linear = linear(avgpool, 10)
    #    fc = tf.nn.softmax(fc_linear)

    return conv

from keras.models import Model    
from keras.layers import *



def u_net(x, batch=2):
    
    with tf.name_scope("downsample") as scope:
        kernel_init = tf.contrib.layers.variance_scaling_initializer()
        conv1_t = tf.layers.conv2d(x, 64, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        conv1 = tf.layers.conv2d(conv1_t, 64, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,256,256,128]<-[2,512,512,64]
        pool1 = maxpool(conv1)
        conv2_t = tf.layers.conv2d(pool1, 128, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        conv2 = tf.layers.conv2d(conv2_t, 128, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,128,128,256]<-[2,256,256,128]
        pool2 = maxpool(conv2)
        conv3_t = tf.layers.conv2d(pool2, 256, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        conv3 = tf.layers.conv2d(conv3_t, 256, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,64,64,512]<-[2,128,128,256]
        pool3 = maxpool(conv3)
        conv4_t = tf.layers.conv2d(pool3, 512, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        conv4 = tf.layers.conv2d(conv4_t, 512, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,32,32,1024]<-[2,64,64,512]
        pool4 = maxpool(conv4)
        conv5_t = tf.layers.conv2d(pool4, 1024, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        conv5 = tf.layers.conv2d(conv5_t, 1024, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")

        #print "conv5", conv5
    with tf.name_scope("upsample") as scope:
        ####[2,32,32,1024]->[2,64,64,512]
        up4 = tf.layers.conv2d_transpose(conv5, 512, 3, 2, kernel_initializer= kernel_init, activation=tf.nn.relu, padding = "SAME")
        up4_t = concatenate([up4, conv4], axis = 2)
        uconv4_t = tf.layers.conv2d(up4_t, 512, 3, (1,2), kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        uconv4 = tf.layers.conv2d(uconv4_t, 512, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,64,64,512]->[2,128,128,256]
        up3_t = tf.layers.conv2d_transpose(uconv4, 256, 3, 2,  kernel_initializer= kernel_init, activation=tf.nn.relu, padding = "SAME")
        up3 = concatenate([up3_t, conv3], axis = 2)
        uconv3_t = tf.layers.conv2d(up3, 256, 3, (1,2), kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        uconv3 = tf.layers.conv2d(uconv3_t, 256, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,128,128,256]->[2,256,256,128]
        up2_t = tf.layers.conv2d_transpose(uconv3, 128, 3, 2, kernel_initializer= kernel_init, activation=tf.nn.relu, padding = "SAME")
        up2 = concatenate([up2_t, conv2], axis = 2)
        uconv2_t = tf.layers.conv2d(up2, 128, 3, (1,2), kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        uconv2 = tf.layers.conv2d(uconv2_t, 128, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        
        ####[2,256,256,128]->[2,512,512,64]
        up1_t = tf.layers.conv2d_transpose(uconv2, 64, 3, 2, kernel_initializer= kernel_init, activation=tf.nn.relu, padding = "SAME")
        up1 = concatenate([up1_t, conv1] , axis = 2)
        uconv1_t = tf.layers.conv2d(up1, 128, 3, (1,2), kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        uconv1_te = tf.layers.conv2d(uconv1_t, 64, 3, 1, kernel_initializer= kernel_init, activation=tf.nn.relu, padding="SAME")
        uconv1 = tf.layers.conv2d(uconv1_te, 1, 3, 1, kernel_initializer= kernel_init,  padding="SAME")
    return uconv1

def rmse_finder(predictions, labels):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predictions))))

#set batch size
batch = 2
c = .001 # learning rate
dropout_rt = .5

#here is the computation graph
with tf.Graph().as_default():
    input_data = tf.placeholder( tf.float32, [batch, 512, 512, 1] )
    input_label = tf.placeholder(tf.float32, [batch, 512, 512, 1])
    #keep_prob = tf.placeholder(tf.float32)

    #5 layer computation graph
    c4 = comp_gr1(input_data)
    #pl = tf.cast(input_label, tf.int64)
    # do rmse for loss
    cross_entropy_cnn = rmse_finder(input_label, c4)
    loss_cnn = tf.reduce_mean(cross_entropy_cnn)
    train_step_cnn = tf.train.AdamOptimizer(learning_rate=c).minimize(loss_cnn)
    
    # dropout computation graph
    drop = dropout_comp_gr(input_data, dropout_rt)
    cross_entropy_drout = rmse_finder(input_label, drop)
    loss_drout = tf.reduce_mean(cross_entropy_drout)
    train_step_drout = tf.train.AdamOptimizer(learning_rate=c).minimize(loss_drout)
    
    # ResNet computation graph
    resn = resNet(input_data, batch)
    cross_entropy_resn = rmse_finder(input_label, resn)
    loss_resn = tf.reduce_mean(cross_entropy_resn)
    train_step_resn = tf.train.AdamOptimizer(learning_rate=c).minimize(loss_resn)
    
    # U-Net computation graph
    unet = u_net(input_data, batch)
    #weight = 10
    #loss_unet = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=input_label, logits=unet, pos_weight=weight))
    loss_unet = rmse_finder(input_label, unet)
    train_step_unet = tf.train.AdamOptimizer(learning_rate=c).minimize(loss_unet)
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    writer = tf.summary.FileWriter("./tf_lab_final", sess.graph)
    sess.run(init)
    loss_orig, loss_r, loss_d, loss_u = [], [], [], []
 
    epochs = 1000
    #epochs = 5*len_files
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
        loss_in = sess.run(loss_cnn, feed_dict = {input_data:input_images,input_label:output_images})
        _ ,test_out  = sess.run( [train_step_cnn, c4], feed_dict = {input_data:input_images,input_label:output_images})
        loss_dr_in = sess.run(loss_drout, feed_dict = {input_data:input_images,input_label:output_images})
        _, test_dr_out = sess.run([train_step_drout, drop], feed_dict = {input_data:input_images,input_label:output_images})
        loss_res_in = sess.run(loss_resn, feed_dict = {input_data:input_images,input_label:output_images})
        _, test_res_out = sess.run([train_step_resn, resn], feed_dict = {input_data:input_images,input_label:output_images})
        loss_unet_in = sess.run(loss_unet, feed_dict = {input_data:input_images,input_label:output_images})
        _, test_unet_out = sess.run([train_step_unet, unet], feed_dict = {input_data:input_images,input_label:output_images})
        loss_orig.append(loss_in)
        loss_r.append(loss_res_in)
        loss_u.append(loss_unet_in)
        loss_d.append(loss_dr_in)
        if ep % 100 == 0:
            print  ep, ": ", loss_in
            testOut = test_out[0,:,:,0]
            image.imsave('test_img_{}.png'.format(ep), testOut)
            testOutDr = test_dr_out[0,:,:,0]
            image.imsave('test_img_drout{}.png'.format(ep), testOutDr)
            testOutRN = test_res_out[0,:,:,0]
            image.imsave('test_img_resn{}.png'.format(ep), testOutRN)
            testOutunet = test_unet_out[0,:,:,0]
            image.imsave('test_img_unet{}.png'.format(ep), testOutunet)

    print "orig: ", loss_orig
    print "drout: ", loss_d
    print "resn: ", loss_r
    print "unet: ", loss_u
    plt.plot(loss_orig, label='5 Layer CNN')
    plt.plot(loss_d, label ='Dropout')
    plt.plot(loss_r, label='ResNet')
    plt.plot(loss_u, label='U-Net')
    plt.title("Root Mean Squared Error")
    plt.xlabel("Error")
    plt.ylabel("Epochs")
    plt.legend(loc='upper right')
    plt.savefig('rmse.png')



    sess.close()
