import tensorflow as tf


def build_discriminator(feature_dim):
    inputs = tf.keras.Input(shape=(feature_dim,))  # input original or perturbed data

    x = tf.keras.layers.Dense(128)(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # logits（未經過 sigmoid）
    logits = tf.keras.layers.Dense(1)(x)

    # prob（經過 sigmoid）
    prob = tf.keras.layers.Activation('sigmoid')(logits)

    model = tf.keras.Model(inputs=inputs, outputs=[logits, prob])
    return model
