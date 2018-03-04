import tensorflow as tf

w = tf.Variable([[1, 2]])
x = tf.Variable([[3], [4]])
y = tf.Variable(tf.random_uniform([1],-1.0, 1))

# 默认初始化全部本地变量
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # Tensor.eval()等价于tf.get_default_session().run(t)
    print(y.eval())
