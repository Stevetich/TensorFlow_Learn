import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


# 加载MNIST数据
mnist = input_data.read_data_sets('mnist_data', one_hot = True)

# 占位符
x = tf.placeholder(tf.float32, [None, 784])
# lable 记录实际的标签值
label = tf.placeholder(tf.float32, [None, 10])

# 将图片从784的向量还原为28 * 28的矩阵图片 深度为1
# 第一个参数 -1 表示一维的长度不限定
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 过滤器
def weight_variable(shape) :
	# stddev : standard deviation 标准差
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

# 初始化偏置
def bias_variable(shape) :
	return tf.Variable(tf.constant(0.1, shape = shape))

# 卷积运算
def conv2d(x, W) :
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

# 池化运算
def max_pool_2x2(x) :
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

# TensorFlow 封装了卷积操作，调用tf.nn.conv2d函数即可
# 卷积以后，使用激活函数去线性化，再经过池化层
def conv_layer(input, filter_shape, bias_shape) :
	W = weight_variable(filter_shape)
	b = bias_variable(bias_shape)
	# 使用conv2d函数进行卷积计算，再用ReLU作为激活函数
	h = tf.nn.relu(conv2d(input, W) + b)
	# 卷积以后再池化
	return max_pool_2x2(h)

# 全连接层
def dense(input, weight_shape, bias_shape, reshape) :
	W = weight_variable(weight_shape)
	b = bias_variable(bias_shape)
	# 将输入的数据还原为向量
	h = tf.reshape(input, reshape)
	# 仍然使用ReLU作为激活函数
	return tf.nn.relu(tf.matmul(h, W) + b)

# Dropout层 正则化，防止过拟合
def dropout(input) :
	keep_prob = tf.placeholder(tf.float32)
	return keep_prob, tf.nn.dropout(input, keep_prob)

# Softmax层 输出
def softmax(input, weight_shape, bias_shape) :
	W = weight_variable(weight_shape)
	b = bias_variable(bias_shape)
	return tf.nn.softmax(tf.matmul(input, W) + b)

# 损失函数和优化器
def optimizer(label, y) :
	loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(y)))
	return tf.train.AdamOptimizer(1e-4).minimize(loss), loss

# 计算模型预测准确率
def accuracy(label, y) :
	pred = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
	return tf.reduce_mean(tf.cast(pred, tf.float32))

def net(input, label) :
	c1 = conv_layer(input, [5, 5, 1, 32], [32])
	c2 = conv_layer(c1, [5, 5, 32, 64], [64])

	f1 = dense(c2, [7 * 7 * 64, 1024], [1024], [-1, 7 * 7 * 64])

	keep_prob, h = dropout(f1)

	y = softmax(h, [1024, 10], [10])

	op, loss = optimizer(label, y)
	acc = accuracy(label, y)

	return acc, op, keep_prob, loss

acc, op, keep_prob, loss = net(x_image, label)
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', acc)

# saveFile = mkdir_saver()
saver = tf.train.Saver()

with tf.Session() as sess :
	# 初始化变量
	sess.run(tf.global_variables_initializer())

	summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter("log/", sess.graph)

	for i in range(20000) :
		batch = mnist.train.next_batch(50)

		sess.run(op, feed_dict = {x : batch[0], label : batch[1], keep_prob : 0.5})
		if i % 100 == 0:
			train_accuracy = sess.run(acc, feed_dict={x: batch[0], label: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g" % (i, train_accuracy))
 
			summary_str = sess.run(summary_op, feed_dict={x: batch[0], label: batch[1], keep_prob: 1.0})
			summary_writer.add_summary(summary_str, i)

	print("end train, start testing...")
	print("test accuracy %g" % sess.run(acc, feed_dict={x:mnist.test.images, label:mnist.test.labels, keep_prob:1.0}))
	# saver.save(sess, saveFile)
