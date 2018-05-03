import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# A Non-Convolutional Neural Network for number recognition of the mnist data


def create_subplot(a, b, A, B):
    axes = plt.gca()
    axes.set_xlim([0, 28])
    axes.set_ylim([0, 28])
    plt.xticks(np.arange(29, step=4))
    plt.yticks(np.arange(29, step=4))

    prob_tensor = sess.run([output_layer_softmax], feed_dict={x: A, y: B})
    prob_vec = np.asarray(prob_tensor[0])

    best_num = np.argmax(prob_vec[0])

    sm_list = (prob_vec[0]).tolist()
    sm_list.remove(max(prob_vec[0]))
    second_best_num = np.argmax(sm_list)

    plt.text(0.5, 24, ("{0:0.1f} with {1:0.3f}\n{2:0.1f} with {3:0.3f}".format(best_num, max(prob_vec[0]), second_best_num, max(sm_list))))
    for i in range(28):
        for j in range(28):
            plt.fill([j, j, j + 1, j + 1], [i, i + 1, i + 1, i],
                     color=[1 - a[27 - i, j], 1 - a[27 - i, j], 1 - a[27 - i, j]])  # [0, 0, 1, 1] [0, 1, 1, 0]
    plt.title("Real number: {0:1.0f}".format(get(b)))


def plot_six(acc, a1, b1, A1, B1, a2, b2, A2, B2, a3, b3, A3, B3, a4, b4, A4, B4, a5, b5, A5, B5, a6, b6, A6, B6):
    fig = plt.figure()
    plt.subplot(2, 3, 1)
    create_subplot(a1, b1, A1, B1)

    plt.subplot(2, 3, 2)
    create_subplot(a2, b2, A2, B2)

    plt.subplot(2, 3, 3)
    create_subplot(a3, b3, A3, B3)

    plt.subplot(2, 3, 4)
    create_subplot(a4, b4, A4, B4)

    plt.subplot(2, 3, 5)
    create_subplot(a5, b5, A5, B5)

    plt.subplot(2, 3, 6)
    create_subplot(a6, b6, A6, B6)

    plt.suptitle("Non-Convolutional Neural Network accuracy: %s" % acc, fontsize=16)
    plt.show()


def get(x):
    for i in range(len(x)):
        if x[i] != 0.0:
            return i


mnist = input_data.read_data_sets("first_time/datum", one_hot=True)

y = tf.placeholder(tf.float32, [None, 10])
x = tf.placeholder(tf.float32, [None, 784])

W_hidden = tf.Variable(tf.random_normal([784, 300], stddev=0.05))
b_hidden = tf.Variable(tf.random_normal([300]))

W_output = tf.Variable(tf.random_normal([300, 10], stddev=0.05))
b_output = tf.Variable(tf.random_normal([10]))

batch_size = 100
learning_rate = 0.01
epochs = 10

hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, W_hidden), b_hidden))
output_layer = tf.add(tf.matmul(hidden_layer, W_output), b_output)
output_layer_SCE = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)
output_layer_softmax = tf.nn.softmax(output_layer)
loss = tf.reduce_mean(output_layer_SCE)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    number_batches = int(len(mnist.train.labels)/batch_size)

    for epoch in range(epochs):
        avg_cost = 0
        for i in range(number_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})  # Training begins here
            avg_cost += c
        print("Total cost at epoch", epoch + 1, "is", avg_cost)

    matching = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
    matching_bool = sess.run([matching], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    for i in range(10000):
        if not (matching_bool[0])[i]:  # if not True <=> if False
            fail_index = i
            break

    accuracy = tf.reduce_mean(tf.cast(matching, tf.float32))
    accuracy = sess.run([accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})  # Testing

    rand_int = np.random.randint(0, 10000-1)  # Plotting starts here
    a1 = mnist.test.images[rand_int]
    A1 = np.asarray(a1).reshape((1, 784))
    b1 = mnist.test.labels[rand_int]
    B1 = np.asarray(b1).reshape((1, 10))
    a1 = np.asarray(a1).reshape((28, 28))

    rand_int = np.random.randint(0, 10000-1)
    a2 = mnist.test.images[rand_int]
    A2 = np.asarray(a2).reshape((1, 784))
    b2 = mnist.test.labels[rand_int]
    B2 = np.asarray(b2).reshape((1, 10))
    a2 = np.asarray(a2).reshape((28, 28))

    rand_int = np.random.randint(0, 10000-1)
    a3 = mnist.test.images[rand_int]
    A3 = np.asarray(a3).reshape((1, 784))
    b3 = mnist.test.labels[rand_int]
    B3 = np.asarray(b3).reshape((1, 10))
    a3 = np.asarray(a3).reshape((28, 28))

    rand_int = np.random.randint(0, 10000-1)
    a4 = mnist.test.images[rand_int]
    A4 = np.asarray(a4).reshape((1, 784))
    b4 = mnist.test.labels[rand_int]
    B4 = np.asarray(b4).reshape((1, 10))
    a4 = np.asarray(a4).reshape((28, 28))

    rand_int = np.random.randint(0, 10000-1)
    a5 = mnist.test.images[rand_int]
    A5 = np.asarray(a5).reshape((1, 784))
    b5 = mnist.test.labels[rand_int]
    B5 = np.asarray(b5).reshape((1, 10))
    a5 = np.asarray(a5).reshape((28, 28))

    a = mnist.test.images[fail_index]
    A = np.asarray(a).reshape((1, 784))
    b = mnist.test.labels[fail_index]
    B = np.asarray(b).reshape((1, 10))
    a = np.asarray(a).reshape((28, 28))

    plot_six(accuracy, a1, b1, A1, B1, a2, b2, A2, B2, a3, b3, A3, B3, a4, b4, A4, B4, a5, b5, A5, B5, a, b, A, B)


