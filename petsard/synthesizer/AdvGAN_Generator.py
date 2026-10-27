import tensorflow as tf


def build_generator(feature_dim):

    def custom_activation(x):

        return x

    inputs = tf.keras.Input(shape=(feature_dim,))   # Indicate input(original data) shape
    # inputs = tf.keras.Input(shape=(100,))     # Indicate input(noise) shape
    x = tf.keras.layers.Dense(256)(inputs)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
    outputs = tf.keras.layers.Dense(feature_dim)(x)
    # scale_factor = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # 讓 scale 在 (0,1) 之間
    # x = tf.keras.layers.Dense(feature_dim, activation='tanh')(x)
    # x = tf.keras.layers.Activation(custom_activation)(x)
    # outputs = x * (scale_factor * 10)  # 讓最大範圍到 [-10,10]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
