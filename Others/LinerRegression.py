import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# build dataset
num_points = 1000
vector_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = 0.3*x1 + 0.2 + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])
x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

plt.scatter(x_data, y_data, c="blue")
plt.show()

# train
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data), name="loss")
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name="train")

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print("0 W=", W.eval(), "b=", b.eval(), "loss=", loss.eval())
    for step in range(50):
        sess.run(train)
        print("W=", W.eval(), "b=", b.eval(), "loss=", loss.eval())

    plt.scatter(x_data, y_data, c="blue")
    plt.plot(x_data, W.eval()*x_data+b.eval())
    plt.show()