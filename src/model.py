import random 
import tensorflow as tf
import numpy as np

def weights(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.Variable(tf.random_normal(shape, stddev = xavier_stddev))

def bias(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


class EmbeddingBlock(tf.layers.Layer):
    def __init__(self, aa_size, seq_len, n_outputs, kernel_size, strides, dilation_rate, p_keep, name=None):
        super(EmbeddingBlock, self).__init__(name=name)        
        self.p_keep = p_keep
        self.n_outputs = n_outputs
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.nchan = n_outputs
        self.dilation_rate = dilation_rate
        self.aa_size = aa_size

    def build(self, input_shape):
        channel_dim = -1
        self.built = True
    
    def call(self, inputs, training=True):
        embedding_size = 100
        emb = tf.Variable(tf.random_uniform([self.aa_size, embedding_size], -1, 1), dtype=tf.float32)
        pads = tf.constant([[0,1], [0,0]])
        embeddings = tf.pad(emb, pads)
                
        w0 = weights([1, embedding_size, 1, self.nchan])
        b0 = bias([self.seq_len+1, 1, self.nchan])
        
        #print(inputs.get_shape().as_list())
        padding = 1
        pad = tf.constant([[0, 0], [padding, 0]]) #[batch_size, length]
        x = tf.pad(inputs, pad, constant_values=self.aa_size) #[batch_size, seq_len+padding]
        x = tf.nn.embedding_lookup(embeddings, x) #[batch_size, seq_len+padding, embedding_size]
        x = tf.reshape(x, [-1, self.seq_len+padding, embedding_size, 1])
        x = tf.nn.conv2d(x, w0, padding='VALID', strides=[1,1,1,1], dilations=[1,1,1,1], data_format='NHWC')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x+b0)
        x = tf.nn.dropout(x, self.p_keep)
        #print(x.get_shape().as_list()) 
        
        return x
    
    
class TCBlock(tf.layers.Layer):
    def __init__(self, seq_len, n_outputs, kernel_size, strides, p_keep, dilation_rate, name=None):
        super(TCBlock, self).__init__(name=name)        
        self.p_keep = p_keep
        self.n_outputs = n_outputs
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.nchan = n_outputs
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        channel_dim = -1
        self.built = True
    
    def call(self, inputs, training=True):
        w1 = weights([self.kernel_size, 1, self.nchan, self.nchan]) 
        
        padding = (self.kernel_size - 1) * self.dilation_rate
        pad = tf.constant([[0, 0], [padding, 0], [0,0], [0,0]]) #[batch_size, length, 1, channels]
        x = tf.pad(inputs, pad, constant_values=0)        
        x = tf.nn.conv2d(x, w1, padding='VALID', strides=[1,1,1,1], 
                             dilations=[1, self.dilation_rate, 1, 1], data_format='NHWC')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, self.p_keep)
        #print(x.get_shape().as_list())
        
        return x
    
    

class AttentionBlock(tf.layers.Layer):
    def __init__(self, aa_size, seq_len, n_outputs, p_keep, name=None):
        super(AttentionBlock, self).__init__(name=name)      
        self.n_outputs = n_outputs
        self.seq_len = seq_len
        self.nchan = n_outputs
        self.p_keep = p_keep
        self.aa_size = aa_size
        
    def build(self, input_shape):
        channel_dim = -1
        self.built = True
    
    def call(self, inputs, out0, training=True):
        wo = weights([2*self.nchan, self.aa_size])
        bo = bias([self.seq_len, self.aa_size])
        
        out0 = tf.squeeze(out0, [2])
        out0 = out0[:, :-1, :] #[batch_size, seq_len, nchan]
        x = tf.squeeze(inputs, [2])
        x = x[:, :-1, :] #[batch_size, seq_len, nchan]
        
        # Train attention filter
        wt = np.ones((self.seq_len, self.seq_len), dtype=np.float32) * (-9999)
        wt = np.triu(wt, 1) #Return a copy of a matrix with the elements below the diagonal zeroed
        wa = tf.Variable(wt) #this is being updated
        wa_n = tf.constant(np.array(range(1, self.seq_len+1), dtype=np.float32))        
        
        wa_s = tf.nn.softmax(wa, axis=1) #row direction
        # tf.multiply is element-wise multiplication
        wa_s = tf.transpose(tf.multiply(tf.transpose(wa_s), wa_n))
        con = tf.einsum('jk,ikl->ijl', wa_s, out0) #[batch_size, seq_len, nchan]
        
        out = tf.concat([x, con], 2) #[batch_size, seq_len, 2*nchan]
        out = tf.layers.batch_normalization(out)
        out = tf.nn.dropout(out, self.p_keep)
        out = tf.reshape(out, [-1, 2*self.nchan]) #[batch_size*seq_len, 2*nchan]

        out = tf.matmul(out, wo) #[batch_size*seq_len, aa_size]
        out = tf.reshape(out, [-1, self.seq_len, self.aa_size]) #[batch_size, seq_len, aa_size]
        out = out + bo
        #print(out.get_shape().as_list())
        
        return out, wa_s


    
class TemporalConvNet(tf.layers.Layer):
    def __init__(self, aa_size, seq_len, num_channels, kernel_size, p_keep, name=None):
        super(TemporalConvNet, self).__init__(name=name)
        self.layers = []
        self.num_levels = len(num_channels)
        out_channels = num_channels[0]
        
        self.layers.append(EmbeddingBlock(
                aa_size, seq_len, out_channels, kernel_size, strides=1, dilation_rate=1, 
                p_keep=p_keep, name="tblock_{}".format(0)))

        for i in range(0, self.num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[0]
            self.layers.append(TCBlock(
                seq_len, out_channels, kernel_size, strides=1, dilation_rate=2 ** i, 
                p_keep=p_keep, name="tblock_{}".format(i)))

        self.layers.append(AttentionBlock(
            aa_size, seq_len, out_channels, 
            p_keep=p_keep, name="tblock_{}".format(self.num_levels)))
        
    
    def call(self, inputs, training=True, **kwargs):
        outputs = inputs
        d={}
        for i in range(0, self.num_levels + 2): 
            #FirstBlock
            if i == 0: 
                outputs = self.layers[i](outputs, training = training)
                d['outputs_%d'%i] = outputs
                out0 = d['outputs_%d'%i]
                
            #AttentionBlock
            elif i == self.num_levels + 1: 
                outputs += d['outputs_%d'%(i-2)] #residual connection
                outputs, wa_s = self.layers[i](outputs, d['outputs_%d'%0], training = training)
                d['outputs_%d'%i] = outputs
                      
            #TemporalBlock
            else: 
                if i > 2:
                    outputs += d['outputs_%d'%(i-2)]
                outputs = self.layers[i](outputs, training = training)
                d['outputs_%d'%i] = outputs
        
        return outputs, wa_s
                           
                           
