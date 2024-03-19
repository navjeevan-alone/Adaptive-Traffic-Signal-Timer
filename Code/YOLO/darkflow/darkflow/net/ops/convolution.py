import tensorflow as tf
import numpy as np
from .baseop import BaseOp

class reorg(BaseOp):
    def forward(self):
        inp = self.inp.out
        s = self.lay.stride
        bs, h, w, c = inp.shape
        out = tf.reshape(inp, [bs, h // s, s, w // s, s, c])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        self.out = tf.reshape(out, [bs, h // s, w // s, c * s * s])

    def speak(self):
        args = [self.lay.stride] * 2
        msg = 'local flatten {}x{}'
        return msg.format(*args)

class local(BaseOp):
    def forward(self):
        pad = [[0, 0]] + [[self.lay.pad, self.lay.pad]] * 2 + [[0, 0]]
        temp = tf.pad(self.inp.out, pad)

        k = self.lay.w['kernels']
        out = []
        for i in range(self.lay.h_out):
            row_i = []
            for j in range(self.lay.w_out):
                kij = k[i * self.lay.w_out + j]
                slice_row = slice(i, i + self.lay.ksize)
                slice_col = slice(j, j + self.lay.ksize)
                tij = temp[:, slice_row, slice_col, :]
                conv = tf.nn.conv2d(tij, kij, strides=[1, 1, 1, 1], padding='VALID')
                row_i.append(conv)
            out.append(tf.concat(row_i, axis=2))
        self.out = tf.concat(out, axis=1)

    def speak(self):
        l = self.lay
        args = [l.ksize, l.ksize, l.pad, l.stride, l.activation]
        msg = 'loca {}x{}p{}_{}  {}'.format(*args)
        return msg

class convolutional(BaseOp):
    def forward(self):
        temp = tf.pad(self.inp.out, [[0, 0], [self.lay.pad, self.lay.pad], [self.lay.pad, self.lay.pad], [0, 0]])
        temp = tf.nn.conv2d(temp, self.lay.w['kernel'], strides=[1, self.lay.stride, self.lay.stride, 1], padding='VALID')
        if self.lay.batch_norm:
            temp = self.batchnorm(self.lay, temp)
        else:
            temp = tf.nn.bias_add(temp, self.lay.w['biases'])
        self.out = temp

    def batchnorm(self, layer, inp):
        if not self.var:
            temp = (inp - layer.w['moving_mean'])
            temp /= (np.sqrt(layer.w['moving_variance']) + layer.epsilon)
            temp *= layer.w['gamma']
            temp += layer.w['beta']
            return temp
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                center=False,
                scale=True,
                trainable=layer.h['is_training'],
                gamma_initializer=tf.constant_initializer(layer.w['gamma']),
                moving_mean_initializer=tf.constant_initializer(layer.w['moving_mean']),
                moving_variance_initializer=tf.constant_initializer(layer.w['moving_variance']),
                beta_initializer=tf.constant_initializer(layer.w['beta']),
            )(inp)

    def speak(self):
        l = self.lay
        batch_norm_part = '+bnorm' if l.batch_norm else ''
        args = [l.ksize, l.ksize, l.pad, l.stride, batch_norm_part, l.activation]
        msg = 'conv {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class conv_select(convolutional):
    def speak(self):
        l = self.lay
        batch_norm_part = '+bnorm' if l.batch_norm else ''
        args = [l.ksize, l.ksize, l.pad, l.stride, batch_norm_part, l.activation]
        msg = 'sele {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class conv_extract(convolutional):
    def speak(self):
        l = self.lay
        batch_norm_part = '+bnorm' if l.batch_norm else ''
        args = [l.ksize, l.ksize, l.pad, l.stride, batch_norm_part, l.activation]
        msg = 'extr {}x{}p{}_{}  {}  {}'.format(*args)
        return msg
