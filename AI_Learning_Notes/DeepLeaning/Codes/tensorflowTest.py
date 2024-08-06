import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

coefficients = np.array([[1.], [-8.], [16.]])

w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(shape=[3, 1], dtype=tf.float32)
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()
# session = tf.Session()
# session.run(init)
# print((session.run(w)))
with tf.Session() as session:
    session.run(init)
    print((session.run(w)))

session.run(train, feed_dict={x: coefficients})
print((session.run(w)))

for i in range(1000):
    session.run(train, feed_dict={x: coefficients})
print((session.run(w)))
