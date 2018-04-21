import tensorflow as tf
W = tf.Variable([tf.random_normal([1])], name="Weight")  # Generate values using the Gaussian dist.
b = tf.Variable([tf.random_normal([1])], name="bias")
x = tf.placeholder(tf.float32, name='x-placeholder')
linear_model = tf.add(tf.multiply(W, x), b)  # y = W*x + b
y = tf.placeholder(tf.float32, name='y-placeholder')

init = tf.global_variables_initializer()

loss = tf.reduce_sum(tf.square(y - linear_model))
optimizer = tf.train.GradientDescentOptimizer(0.01)  # Use the GDO with a learning rate of .01
train = optimizer.minimize(loss)

with tf.Session() as sess:  # Let's begin the session!
    sess.run(init)
    writer = tf.summary.FileWriter('lin_mod_graph', sess.graph)  # Visualizing in Tensorboard
    print("Current weight: ", sess.run(W), " with bias: ", sess.run(b))
    print("Current outputs: ", sess.run(linear_model, {x: [1, 2, 3]}))
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3], y: [1, 2, 3]})  # Obv. W = 1 and b = 0
    print("Weight: ", sess.run([W]))
    print("Bias: ", sess.run([b]))
    writer.close()


