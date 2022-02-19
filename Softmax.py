from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import scipy.misc as sm
import numpy as np

# import warnings
# warnings.filterwarnings('ignore')
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
mnist = input_data.read_data_sets("mnist_data", one_hot = True)


save_dir = 'mnist_data/images/'

if not os.path.exists(save_dir) :
	os.makedirs(save_dir)

# for i in range(50) :
# 	image_array = mnist.train.images[i, :]
# 	one_hot_label = mnist.train.labels[i, :]
# 	label = np.argmax(one_hot_label)
# 	image_array = image_array.reshape(28, 28)
# 	filename = save_dir + 'image_train_%d_%d.jpg' % (i, label)
# 	sm.toimage(image_array).save(filename)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

saver = tf.train.Saver()

with tf.Session() as sess :
	tf.global_variables_initializer().run()
	for _ in range(1000) :
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(optimizer, feed_dict  = {x : batch_xs, y_ : batch_ys})
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print(sess.run(accuracy, feed_dict = {x : mnist.test.images, y_ : mnist.test.labels}))

	saver.save(sess, './saver/mnist.ckpt')