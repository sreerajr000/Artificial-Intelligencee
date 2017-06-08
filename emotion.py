import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from random import randint
import sys

infile_i = open("images.npy", "rb")
infile_l = open("labels.npy", "rb")

emo = [np.load(infile_i), np.load(infile_l)]

infile_i.close()
infile_l.close()

print('Data Loaded...')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 10000])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([10000,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(69):
  batch = (emo[0][i*10:(i+1)*10], emo[1][i*10:(i+1)*10])
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  if i%10 == 0:
    print(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))

batch_test = (emo[0][600:], emo[1][600:])
#generate test data
for i in range(90):
  n = randint(0,690)
  batch_test[0][i] = emo[0][n]
  batch_test[1][i] = emo[1][n]
print(accuracy.eval(feed_dict={x: batch_test[0], y_: batch_test[1]}))
'''
for i in range(25):

  num = randint(0, 689)
  img = emo[0][num]
  classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
  #plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
  #plt.show()
  print ('Actual', np.argmax(emo[1][num]))
  print ('Logistic Regression Predicted ', classification[0])
'''



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

x_image = tf.reshape(x, [-1,100,100,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([40000, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 40000])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

saver = tf.train.Saver()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

try:
  saver.restore(sess, "model.ckpt")
  print("Model restored.")
  print("Continuing Training...")
except:
  print("Model not found\nTraining new Model...')
for epoch in range(120):
  for i in range(69):
    try:
      batch = (emo[0][i*10:(i+1)*10], emo[1][i*10:(i+1)*10])
      if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("epoch %d : step %d, training accuracy %g"%(epoch, i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    except:
      print('Exception caught')
      save_path = saver.save(sess, "model.ckpt")
      print("Model saved in file: %" % save_path)
      print("Test accuracy %g"%accuracy.eval(feed_dict={x: emo[0][570:], y_: emo[1][570:], keep_prob: 1.0}))
      sys.exit()

  


'''
for i in range(25):

  num = randint(0, mnist.test.images.shape[0])
  img = mnist.test.images[num]

  prediction = tf.argmax(y_conv,1)
  classification = prediction.eval(feed_dict={x: [img],keep_prob: 1.0}, session=sess)
  #plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
  #plt.show()
  print 'Actual', np.argmax(mnist.test.labels[num])
  print 'NN predicted', classification


'''
