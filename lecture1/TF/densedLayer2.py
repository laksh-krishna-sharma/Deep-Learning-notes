import tensorflow as tf

layer = tf.keras.layers.Dense(units=2, activation="sigmoid")

inputs = tf.random.normal((5, 3))  # Batch of 5 samples, each with 3 features
output = layer(inputs)
print(output)