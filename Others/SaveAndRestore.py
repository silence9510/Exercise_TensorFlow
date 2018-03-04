import tensorflow as tf

x = tf.Variable([1, 2], name="x")
y = tf.Variable([3, 4], name="y")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, save_path="C:\WorkSpace\PyCharm\TensorFlow\Others\save\model.ckpt")
    print("save_path: ", save_path)

x1 = tf.placeholder(tf.int32, shape=[1,2])
y1 = tf.placeholder(tf.int32, shape=[1,2])
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_path="C:\WorkSpace\PyCharm\TensorFlow\Others\save\model.ckpt")
    print(x.eval())



