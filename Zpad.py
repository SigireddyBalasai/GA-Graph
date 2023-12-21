import numpy as np
import tensorflow as tf

class ZeroPadConcatLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ZeroPadConcatLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # resize along 2 and 3 along the height and width and concatenate along depth
        # inputs is a list of tensors
        # all tensors needs to be resized based on the max height and width
        # then concatenate along depth
        # return the concatenated tensor

        # get the max height and width
        max_width = np.max([x.shape[1] for x in inputs])
        max_height = np.max([x.shape[2] for x in inputs])
        # resize all the tensors
        resized_inputs = [tf.image.resize(x, [max_width, max_height]) for x in inputs]
        # concatenate along depth
        concatenated_tensor = tf.concat(resized_inputs, axis=3)
        #print(concatenated_tensor.shape, "concatenated_tensor.shape", max_width, max_height, "max_width, max_height", [x.shape for x in inputs], "inputs")
        return concatenated_tensor
