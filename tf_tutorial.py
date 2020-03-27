# import tensorflow as tf

# # a = (b + c) * (c + 2)

# const = tf.constant(2.0, name='const')
# b = tf.Variable(2.0, name='b')
# c = tf.Variable(1.0, name='c')

# d = tf.add(b, c, name='d')
# e = tf.add(c, const, name='e')
# a = tf.multiply(d, e, name='a')

# init_opt = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init_opt)
#     a_out = sess.run(a)
#     print('Variable a is {}'.format(a_out))

# Simple Neural Network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print(mnist)

learning_rate = 0.001
epochs = 100
batch_size = 32

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random.normal([300]), name='b1')
W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random.normal([10]), name='b2')


A1 = tf.add(tf.matmul(x, W1), b1)
A1 = tf.nn.relu(A1)
X = tf.matmul(A1, W2)
A2 = tf.nn.softmax(tf.add(tf.matmul(A1, W2), b2))
A2 = tf.clip_by_value(A2, 1e-10, 0.9999999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum((y * tf.log(A2))+ ((1-y) * tf.log(1 - A2)), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

inti_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(A2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(inti_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


