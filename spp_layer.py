import tensorflow as tf
import math

class SPPLayer():
    def __init__(self,bins,feature_map_size):
        self.strides = []
        self.filters = []
#        print(type(feature_map_size))
        self.a = float(feature_map_size)
        self.bins = bins
        self.n = len(bins)

    def spatial_pyramid_pooling(self,data):
        self.input = data
        self.batch_size = self.input.get_shape().as_list()[0]
        for i in range(self.n):
            x = int(math.floor(self.a/float(self.bins[i])))
            self.strides.append(x)
            x = int (math.ceil(self.a/float(self.bins[i])))
            self.filters.append(x)

        self.pooled_out = []
        for i in range(self.n):
            self.pooled_out.append(tf.keras.layers.MaxPool2D(
                                   pool_size=(1, self.filters[i]),
                                   strides=(1, self.strides[i]),
                                   padding="valid")(self.input))

        for i in range(self.n):
            if self.batch_size is None:
                self.pooled_out[i] = tf.reshape(self.pooled_out[i], [-1,])
            else:
                self.pooled_out[i] = tf.reshape(self.pooled_out[i], [self.batch_size, -1])
        
        self.output = tf.concat(1, [self.pooled_out[0], self.pooled_out[1], self.pooled_out[2]])

        return self.output
