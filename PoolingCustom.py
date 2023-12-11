#custom maxpooling layer with error condition and no padding
import tensorflow as tf
class MaxPoolingCustom(tf.keras.layers.Layer):
    def __init__(self, pool_size, name):
        super(MaxPoolingCustom, self).__init__(name=name)
        self.pool_size = pool_size

    def call(self, inputs):
        print(inputs.shape , self.pool_size)
        if inputs.shape[1] % self.pool_size != 0:
            # add padding to make it compatable
            pad_size = self.pool_size - inputs.shape[1] % self.pool_size
            inputs = tf.pad(inputs, [[0, 0], [0, pad_size], [0, pad_size], [0, 0]])
        if(inputs.shape[1] < self.pool_size):
            # adding padding to make it compatable
            pad_size = self.pool_size - inputs.shape[1]
            inputs = tf.pad(inputs, [[0, 0], [0, pad_size], [0, pad_size], [0, 0]])
        return tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.pool_size, padding='VALID')

    def get_config(self):
        config = super(MaxPoolingCustom, self).get_config()
        config.update({'pool_size': self.pool_size})
        return config
    

class AveragePoolingCustom(tf.keras.layers.Layer):
    def __init__(self, pool_size, name):
        super(AveragePoolingCustom, self).__init__(name=name)
        self.pool_size = pool_size

    def call(self, inputs):
        if inputs.shape[1] % self.pool_size != 0:
            # add padding to make it compatable
            pad_size = self.pool_size - inputs.shape[1] % self.pool_size
            inputs = tf.pad(inputs, [[0, 0], [0, pad_size], [0, pad_size], [0, 0]])
        if(inputs.shape[1] < self.pool_size):
            # adding padding to make it compatable
            pad_size = self.pool_size - inputs.shape[1]
            inputs = tf.pad(inputs, [[0, 0], [0, pad_size], [0, pad_size], [0, 0]])
        #print(inputs.shape , self.pool_size)
        return tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=self.pool_size, padding='VALID')
    
    def get_config(self):
        config = super(AveragePoolingCustom, self).get_config()
        config.update({'pool_size': self.pool_size})
        return config