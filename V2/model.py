import tensorflow as tf 
import numpy as np
from V2.preprocessing_imgs import preprocessing_image

class Model_ConvLSTM2D(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.conv2d_1 = tf.keras.layers.ConvLSTM2D(filters=64, 
                                                    data_format = "channels_last",
                                                    kernel_size=3, 
                                                    strides=(2,2),
                                                    activation="relu",
                                                    return_sequences=True)
                                            
        self.maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), 
                                                    padding='same', 
                                                    data_format='channels_last')

        self.conv2d_2 = tf.keras.layers.ConvLSTM2D(filters=16,
                                                    kernel_size=3,
                                                    strides=(2,2),
                                                    activation="relu",
                                                    return_sequences=True)

        self.maxpool_2 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),
                                                    padding='same', 
                                                    data_format='channels_last')

        self.flatten = tf.keras.layers.Flatten()

        self.layer_1 = tf.keras.layers.Dense(128, activation="relu")

        self.layer_2 = tf.keras.layers.Dense(64, activation="relu")

        self.value = tf.keras.layers.Dense(1, name="value")

        self.logits = tf.keras.layers.Dense(num_actions, name="policies")
        self.actions = tf.keras.layers.Softmax()

    def call(self, input):
        
        conv = self.conv2d_1(input)
        conv = self.maxpool_1(conv)
        conv = self.conv2d_2(conv)
        conv = self.maxpool_2(conv)

        flat = self.flatten(conv)

        nn = self.layer_1(flat)
        nn = self.layer_2(nn)
        
        value = self.value(nn)
        actions_logits = self.logits(nn)

        return actions_logits, value

    def action_value(self, obs):

        obs = preprocessing_image(obs)
        actions_logits, value = self.predict(obs)
        actions = self.actions(actions_logits)

        return np.argmax(np.squeeze(actions, axis=-1), axis=-1), np.squeeze(value, axis=-1)



