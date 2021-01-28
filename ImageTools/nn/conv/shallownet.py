from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:

    def build(width, height, depth, classes):

        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation('relu'))

        # >>> model = tf.keras.Sequential()
        # >>> model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
        # >>> model.output_shape
        # (None, 1, 10, 64)


        # >>> model.add(Flatten())
        # >>> model.output_shape
        # (None, 640)

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model