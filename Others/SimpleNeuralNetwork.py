import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# network model
stddev = 0.1
weights = {
    "w1": tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    "w2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    "out": tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    "b1": tf.Variable(tf.zeros([n_hidden_1])),
    "b2": tf.Variable(tf.zeros([n_hidden_2])),
    "out": tf.Variable(tf.zeros([n_classes]))
}
# FP
def multilayers_preceptron(_X, _weights, _biases):
    layer_1 = tf.sigmoid(tf.matmul(_X, _weights["w1"]) + _biases["b1"])
    layer_2 = tf.sigmoid(tf.matmul(layer_1, _weights["w2"]) + _biases["b2"])
    return (tf.matmul(layer_2, _weights["out"]) + _biases["out"])
# related functions definations
pred = multilayers_preceptron(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
opti = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
corr = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))
# model paras
training_epoches = 20
batch_size = 100
display_step = 2
# train
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(training_epoches):
        n_batch = int(mnist.train.num_examples / batch_size)
        avg_cost = 0
        for i in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feeds = {x: batch_x, y: batch_y}
            sess.run(opti, feed_dict=feeds)
            avg_cost += sess.run(loss, feed_dict=feeds)
        avg_cost = avg_cost / n_batch
#       display
        if epoch % display_step == 0:
            print("Epoch:", epoch, "/20", "Cost: ", avg_cost)
            feeds = {x: batch_x, y: batch_y}
            train_accr = sess.run(accr, feed_dict=feeds)
            print("Training Accr: {}".format(train_accr))
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_accr = sess.run(accr, feeds)
            print("Test Accr: {}".format(test_accr))