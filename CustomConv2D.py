import tensorflow as tf

class Conv2DPadLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='valid', strides=(1, 1), activation=None,trainable=True, **kwargs):
        super(Conv2DPadLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.trainable = trainable

    def build(self, input_shape):
        self.conv2d_layer = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            activation=self.activation
        )
        super(Conv2DPadLayer, self).build(input_shape)

    def call(self, inputs):
        print(inputs.shape)
        print(self.kernel_size)
        # Check if input size is less than kernel size
        if self.padding == 'valid' and inputs.shape[1] < self.kernel_size[0] and inputs.shape[2] < self.kernel_size[1]:
            pad_height = self.kernel_size[0] - inputs.shape[1]
            pad_width = self.kernel_size[1] - inputs.shape[2]
            inputs = tf.pad(inputs, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)))
        
        # Perform convolution
        output = self.conv2d_layer(inputs)
        
        return output
