import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("MNIST Ready")

n_input = 28 * 28
n_output = 10
weight = {
    # conv
    "wc1": tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    "wc2": tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    # FC
    "wd1": tf.Variable(tf.random_normal([7*7*128, 1024], stddev=0.1)),
    "wd2": tf.Variable(tf.random_normal([1024, n_output], stddev=0.1)),
}
biases = {
    "bc1": tf.Variable(tf.random_normal([64], stddev=0.1)),
    "bc2": tf.Variable(tf.random_normal([128], stddev=0.1)),
    "bd1": tf.Variable(tf.random_normal([1024], stddev=0.1)),
    "bd2": tf.Variable(tf.random_normal([n_output], stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio):
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    _conv1 = tf.nn.conv2d(input=_input_r, filter=_w["wc1"], strides=[1,1,1,1], padding="SAME")
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b["bc1"]))
    _pool1 = tf.nn.max_pool(value=_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    _conv2 = tf.nn.conv2d(input=_pool_dr1, filter=_w["wc2"], strides=[1,1,1,1], padding="SAME")
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b["bc2"]))
    _pool2 = tf.nn.max_pool(value=_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    _pool2_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # FC Input
    _densel = tf.reshape(_pool2_dr2, [-1, _w["wd1"].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_densel, _w["wd1"]), _b["bd1"]))
    _fc1_dr1 = tf.nn.dropout(_fc1, _keepratio)
    _out = tf.add(tf.matmul(_fc1_dr1, _w["wd2"]), _b["bd2"])
    out = {
        "input_r": _input_r,
        "conv1": _conv1, "pool1": _pool1, "pool1_dr1": _pool_dr1,
        "conv2": _conv2, "pool2": _pool2, "pool2_dr2": _pool2_dr2,
        "fc1": _fc1, "fc1_dr1": _fc1_dr1,
        "out": _out
    }
    return out
print("CNN Ready")


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)
_pred = conv_basic(x, weight, biases, keepratio)["out"]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
optm = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
_corr = tf.equal(tf.arg_max(_pred, 1), tf.arg_max(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))

init = tf.global_variables_initializer()
training_epoches = 15
batch_size = 16
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epoches):
        avg_cost = 0
        # num_batch = int(mnist.train.num_examples / batch_size)
        num_batch = 10
        for i in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_x, y: batch_y, keepratio: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y, keepratio:1.}) / num_batch
        if epoch % display_step == 0:
            print("Epoch: ", epoch, "/", training_epoches, " Cost: ", avg_cost)
            train_accr = sess.run(accr, feed_dict={x: batch_x, y:batch_y, keepratio: 1.})
            print("Accr: ", train_accr)