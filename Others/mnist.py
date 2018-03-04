import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print("number of train data is ", mnist.train.num_examples)
print("number of test data is ", mnist.test.num_examples)
print("shape of train data is ", mnist.train.images.shape)
print("shape of test data is ", mnist.test.images.shape)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabels = mnist.test.labels
# batch
training_epochs = 10
batch_size = 100
display_step = 2
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# init paras
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# 使用交叉熵损失函数
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
cost = -y * tf.log(actv)
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
pred = tf.equal(tf.arg_max(actv ,1), tf.arg_max(y, 1))
accr = tf.reduce_mean(tf.cast(pred, tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys}) / num_batch
        if epoch % display_step == 0:
            feeds_trains = {x:batch_xs, y:batch_ys}
            feeds_test = {x:mnist.test.images, y:mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_trains)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("trian_acc ", train_acc, "test_accr ", test_acc)