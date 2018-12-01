#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:34
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle, io


class LstmUnit(object):
    def __init__(self, hidden_size, input_size, scope_name):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [self.input_size+self.hidden_size, 4*self.hidden_size])
            self.b = tf.get_variable('b', [4*self.hidden_size],
                                     initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.params.update({'W':self.W, 'b':self.b})

    def __call__(self, x, s, finished = None):
        h_prev, c_prev = s

        x = tf.concat([x, h_prev], 1)
        i, j, f, o = tf.split(tf.nn.xw_plus_b(x, self.W, self.b), 4, 1)

        # Final Memory cell
        # add forget_bias (default: 1) the forget gate in order to reduce the scale of forgetting in the beginning of the training.
        c = tf.sigmoid(f+1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j)
        h = tf.sigmoid(o) * tf.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            '''
                tf.where(condition, x=None, y=None, name=None)
                Return the elements, either from x or y, depending on the condition.
                
                The condition tensor acts as a mask that chooses, based on the value at each element, whether the corresponding element / row in the output should be taken from x (if true) or y (if false).
                
                If condition is a vector and x and y are higher rank matrices, then it chooses which row (outer dimension) to copy from x and y. If condition has the same shape as x and y, then it chooses which element to copy from x and y.
            '''
            out = tf.where(finished, tf.zeros_like(h), h)
            state = (tf.where(finished, h_prev, h), tf.where(finished, c_prev, c))
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev), tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state

    def save(self, path):
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path, 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        param_values = pickle.load(io.open(path, 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])