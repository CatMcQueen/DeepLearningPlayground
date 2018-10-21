import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import matplotlib.image as image
from tqdm import tqdm

 
sess = tf.Session()
 
opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )
 
tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )
 
style_img = imread( 'style.png', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )
 
content_img = imread( 'content.png', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )
 
layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]
 
ops = [ getattr( vgg, x ) for x in layers ]
#print "ops", len(ops)
 
content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

#
# --- construct your cost function here
#

content_f = content_acts[-5]
img_content = vgg.conv4_2
print "content", img_content

content_loss = .5*tf.reduce_sum(tf.square(content_f - img_content))
# use standard error back prop using eq (2)

style_layers = [0,2,4,7,-3]
w_l = 1.0/len(style_layers)
c = .1

gram_style, gram_content = [], []
loss_style = []

def gram(x):
    dim = x.get_shape().as_list()
    x = tf.reshape(x, [dim[1]*dim[2], dim[3]])
    if(dim[1]*dim[2] < dim[3]):
        return tf.matmul(x, x, transpose_b=True)  
    else:
        return tf.matmul(x, x, transpose_a=True)

#style_layers = [ 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1' ]
#style_ops = [ getattr( vgg, x ) for x in style_layers ]
#content_ops = getattr(vgg, 'conv4_2')
#content_acts = sess.run( content_ops, feed_dict={vgg.imgs: content_img } )
#content_acts_tf = tf.convert_to_tensor(content_acts, dtype=tf.float32)
#style_acts = sess.run( style_ops, feed_dict={vgg.imgs: style_img} )
#style_acts_tf = [tf.convert_to_tensor(i, dtype=tf.float32) for i in style_acts]

for i in style_layers:
    b, x, y, z = style_acts[i].shape # we don't need batch size
    M = x*y # feature map height by width
    N = z   # the number of filters
    #np.reshape(style_acts[i], (M,N))
    style_f = tf.convert_to_tensor(style_acts[i], dtype=tf.float32)
    style_p = getattr(vgg, layers[i])
    #g_style = np.add.reduce(np.dot(style_f.T, style_f))
    #gram = tf.reduce_mean(tf.matmul(style_p, style_p, transpose_a = True))
    g_style = gram(style_f)
    g_p = gram(style_p)
    el = 1.0/(4*N**2*M**2)* tf.reduce_sum(tf.square(g_p-g_style))
    loss_style.append(el*w_l)

style_loss = tf.add_n(loss_style)
print style_loss

# Var list the noise
alpha = 1
beta = 1000
total_loss = alpha * content_loss + beta * style_loss

train_step = tf.train.AdamOptimizer(learning_rate=c).minimize(total_loss, var_list = [opt_img])


# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.global_variables_initializer() )
vgg.load_weights( 'vgg16_weights.npz', sess )
 
# initialize with the content image
sess.run( opt_img.assign( content_img ))
 
# --- place your optimization loop here
epoch_len = 500
for i in tqdm(xrange(epoch_len)):
    _, t_loss, s_loss, c_loss, im = sess.run([train_step, total_loss, style_loss, content_loss, opt_img]) 
    if i % 50 == 0:
        print "c ", c_loss, " s ", s_loss, " t ", t_loss 
	im = sess.run(opt_img)
        plt.imsave('image1/test_img_{}.png'.format(i), im[0]/255.0)

img_opt = sess.run(opt_img)
#plt.imshow(img_opt.reshape(224,224,3))
#plt.show()
image.imsave('image1/final_img.png', img_opt[0]/255.0)
