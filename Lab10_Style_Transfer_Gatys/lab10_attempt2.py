import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize
 
sess = tf.Session()
 
opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )
 
tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
tmp_img_a = tf.clip_by_value( opt_img, 0.0, 255.0 )
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

content_p = content_acts[-5]
content_x = style_acts[-5]

content_loss = .5*tf.reduce_sum(tf.square(content_p - content_x))
# use standard error back prop using eq (2)

style_layers = [0,2,4,7,-3]
w_l = 1/len(style_layers)
c = .0001

gram_style, gram_content = [], []
loss_style = []

for i in style_layers:
    print "i ", i
    b, x, y, z = style_acts[i].shape # we don't need batchN size
    print "b, x, y, z ", b, x, y, z
    M = x*y # feature map height by width
    N = z   # the number of filters
    content_x = np.reshape(content_acts[i], (M,N))
    style_x = np.reshape(style_acts[i], (M,N))
    g_style = tf.reduce_mean(np.dot(np.transpose(style_x), style_x))
    g_cont = tf.reduce_mean(np.dot(np.transpose(content_x), content_x))
    el = 1.0/(4*N**2*M**2) *  tf.square(tf.reduce_mean(g_style-g_cont))
    loss_style.append(el*w_l)


style_loss = tf.reduce_mean(loss_style)

#style_loss = tf.reduce_mean(w_l*el)
loss_style.append(style_loss)

# Var list the noise
alpha = .1
beta = .00001
total_loss = alpha * content_loss + beta * style_loss

train_step = tf.train.AdamOptimizer(learning_rate=c).minimize(total_loss, var_list= opt_img )


# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.global_variables_initializer() )
vgg.load_weights( 'vgg16_weights.npz', sess )
 
# initialize with the content image
sess.run( opt_img.assign( content_img ))
 
# --- place your optimization loop here
for i in xrange(100):
    tr = sess.run(train_step)

img_opt = sess.run(opt_img)
print img_opt.shape
imshow(img_opt)


