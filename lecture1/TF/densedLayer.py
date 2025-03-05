import tensorflow as tf

class DensedLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Initialize weight and bias with proper shape & initializer
        self.W = self.add_weight(
            shape=(input_dim, output_dim),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(output_dim,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        # Linear transformation
        z = tf.matmul(inputs, self.W) + self.b
        # Non-linear activation
        return tf.math.sigmoid(z)

# Example usage
if __name__ == "__main__":
    layer = DensedLayer(input_dim=3, output_dim=2)
    inputs = tf.random.normal((5, 3))  # Batch of 5 samples, each with 3 features
    output = layer(inputs)
    print(output.numpy())  # Should output a (5, 2) matrix

