import tensorflow as tf
import numpy as np

logits = tf.Variable(np.array([[1, 2, 0], [4, 5, 6]]), dtype=tf.float32)
labels = tf.Variable(np.array([[1, 0, -2], [0, 1, 0]]), dtype=tf.float32)

factor = tf.constant([1,2])

sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

print("eccolo")
vect = tf.cast(
    tf.expand_dims(tf.not_equal(labels, -2), -1),
    sigmoid_loss.dtype)
print(vect)

loss = tf.reduce_mean(sigmoid_loss)


print([sigmoid_loss])
print([loss])