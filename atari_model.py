from random import random

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Concatenate
import numpy as np

class AtariModel:
    def __init__(self):
        inputs = Input(shape=(84, 84, 4))
        layer1 = Conv2D(32, [8, 8], strides=(4, 4), activation="relu")(inputs)
        layer2 = Conv2D(64, [4, 4], strides=(2, 2), activation="relu")(layer1)
        layer3 = Conv2D(64, [3, 3], strides=(1, 1), activation="relu")(layer2)
        layer4 = Flatten()(layer3)
        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(6)(layer5)
        self.model = Model(inputs=inputs, outputs=action)

    def mutate(self, mutation_power):
        weights = self.get_weights()
        for layer in weights:
            noise = np.random.normal(loc=0.0, scale=mutation_power, size=layer.shape)
            layer += noise
        self.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def determine_actions(self, inputs):
        actions = self.model(inputs)
        return [np.argmax(action_set) for action_set in actions]
