#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

from matplotlib import pyplot as plt
from random import randint
import numpy as np
import glob
import matplotlib.image as mpimg

'''
mnist_img = np.ndarray(shape=(550000,784),dtype=np.float32)
mnist_lab = np.ndarray(shape=(550000,10),dtype=np.float32)


def one_hot(i):
  a = np.zeros(10, 'float32')
  a[i] = 1.0
  return a

def load():
  j = 0
  ind = 0
  folder = 'ABCDEFGHIJ'
  for i in folder:
    print 'Folder ', i
    for path in glob.glob(i+"/*.png"):
      try:
        img = mpimg.imread(path)
        img = img.reshape((784))
        mnist_img[j] = img
        j = j + 1
        if j % 1000 == 0:
          print j, 'files'
        mnist_lab[j] = one_hot(ind)
      except :
        print 'Error : ',path
    ind = ind + 1

load()
outfile_i = open("images.npy", "wb")
np.save(outfile_i,mnist_img)
outfile_i.close()
outfile_l = open("labels.npy", "wb")
np.save(outfile_l,mnist_lab)
outfile_l.close()

'''

#infile_i = open("images.npy", "rb")
#infile_l = open("labels.npy", "rb")


#mnist_img = np.load(infile_i)
#mnist_lab = np.load(infile_l)

#smnist = [np.load(infile_i), np.load(infile_l)]


import cPickle as pickle
with open('smnist.p', 'rb') as fp:
  smnist = pickle.load(fp)

print 'Data Loaded...'



sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(5000):
  batch = (smnist[0][i*100:(i+1)*100], smnist[1][i*100:(i+1)*100])
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  if i%100 == 0:
    print(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))

batch_test = (smnist[0][500000:], smnist[1][500000:])
#generate test data
for i in range(5000):
  n = randint(0,500000)
  batch_test[0][i] = smnist[0][n]
  batch_test[1][i] = smnist[1][n]
print(accuracy.eval(feed_dict={x: batch_test[0], y_: batch_test[1]}))

for i in range(25):

  num = randint(0, 500000)
  img = smnist[0][num]


  classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
  #plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
  #plt.show()
  print 'Actual', np.argmax(smnist[1][num])
  print 'NN predicted', classification[0]




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(101):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



for i in range(25):

  num = randint(0, mnist.test.images.shape[0])
  img = mnist.test.images[num]

  prediction = tf.argmax(y_conv,1)
  classification = prediction.eval(feed_dict={x: [img],keep_prob: 1.0}, session=sess)
  #plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
  #plt.show()
  print 'Actual', np.argmax(mnist.test.labels[num])
  print 'NN predicted', classification


