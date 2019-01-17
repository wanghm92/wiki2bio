#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-12 下午10:47
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class tripleAttentionWrapper(object):
    def __init__(self, hidden_size, input_size, field_size, emb_size, hs, wds, fds, scope_name):
        self.hs = tf.transpose(hs, [1,0,2])  # input_len * batch * input_size
        self.fds = tf.transpose(fds, [1,0,2])
        self.wds = tf.transpose(wds, [1,0,2])
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            # project encoder hidden states
            self.Wh = tf.get_variable('Wh', [input_size, hidden_size])
            self.bh = tf.get_variable('bh', [hidden_size])

            self.Ws = tf.get_variable('Ws', [input_size, hidden_size])
            self.bs = tf.get_variable('bs', [hidden_size])

            # project encoder field embeddings
            self.Wf = tf.get_variable('Wf', [field_size, hidden_size])
            self.bf = tf.get_variable('bf', [hidden_size])

            self.Wr = tf.get_variable('Wr', [input_size, hidden_size])
            self.br = tf.get_variable('br', [hidden_size])

            # project encoder word embeddings
            self.We = tf.get_variable('We', [emb_size, hidden_size])
            self.be = tf.get_variable('be', [hidden_size])

            self.Ww = tf.get_variable('Ww', [input_size, hidden_size])
            self.bw = tf.get_variable('bw', [hidden_size])

            # project output vector
            self.Wo = tf.get_variable('Wo', [2*input_size, hidden_size])
            self.bo = tf.get_variable('bo', [hidden_size])

        self.params.update({'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
                            'bh': self.bh, 'bs': self.bs, 'bo': self.bo,
                            'Wf': self.Wf, 'Wr': self.Wr, 'We': self.We,
                            'bf': self.bf, 'br': self.br, 'be': self.be,
                            'Ww': self.Ww, 'bw': self.bw})

        hs2d = tf.reshape(self.hs, [-1, input_size])
        phi_hs2d = tf.tanh(tf.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))

        fds2d = tf.reshape(self.fds, [-1, field_size])
        phi_fds2d = tf.tanh(tf.nn.xw_plus_b(fds2d, self.Wf, self.bf))
        self.phi_fds = tf.reshape(phi_fds2d, tf.shape(self.hs))

        wds2d = tf.reshape(self.wds, [-1, emb_size])
        phi_wds2d = tf.tanh(tf.nn.xw_plus_b(wds2d, self.We, self.be))
        self.phi_wds = tf.reshape(phi_wds2d, tf.shape(self.hs))

    def __call__(self, x, coverage=None, finished=None):
        # hidden-state attentions
        alpha_h = tf.tanh(tf.nn.xw_plus_b(x, self.Ws, self.bs))  # batch * hidden_size
        weights = tf.reduce_sum(self.phi_hs * alpha_h, axis=2, keepdims=True)  # input_len * batch
        weights = tf.exp(weights - tf.reduce_max(weights, axis=0, keepdims=True))
        weights = tf.divide(weights, (1e-6 + tf.reduce_sum(weights, axis=0, keepdims=True)))

        # field (+pos/rpos) embedding attentions
        beta_h = tf.tanh(tf.nn.xw_plus_b(x, self.Wr, self.br))
        fd_weights = tf.reduce_sum(self.phi_fds * beta_h, axis=2, keepdims=True)
        fd_weights = tf.exp(fd_weights - tf.reduce_max(fd_weights, axis=0, keepdims=True))
        fd_weights = tf.divide(fd_weights, (1e-6 + tf.reduce_sum(fd_weights, axis=0, keepdims=True)))

        # word embedding attentions
        gamma_h = tf.tanh(tf.nn.xw_plus_b(x, self.Ww, self.bw))
        wd_weights = tf.reduce_sum(self.phi_wds * gamma_h, axis=2, keepdims=True)
        wd_weights = tf.exp(wd_weights - tf.reduce_max(wd_weights, axis=0, keepdims=True))
        wd_weights = tf.divide(wd_weights, (1e-6 + tf.reduce_sum(wd_weights, axis=0, keepdims=True)))

        # Aggregation
        weights_w = tf.divide(wd_weights * fd_weights, (1e-6 + tf.reduce_sum(wd_weights * fd_weights, axis=0, keepdims=True)))
        weights_h = tf.divide(weights * weights_w, (1e-6 + tf.reduce_sum(weights * weights_w, axis=0, keepdims=True)))

        context_h = tf.reduce_sum(self.hs * weights_h, axis=0)  # batch * input_size
        # context_w = tf.reduce_sum(self.wds * weights_w, axis=0)  # batch * input_size
        out = tf.tanh(tf.nn.xw_plus_b(tf.concat([x, context_h], -1), self.Wo, self.bo))

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
        return out, weights

    def save(self, path):
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path, 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        param_values = pickle.load(open(path, 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])
