#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:37
# @Author  : Tianyu Liu

from __future__ import print_function

import tensorflow as tf
import pickle, sys, time
from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit
from reward import get_reward, get_reward_bleu, get_reward_coverage, get_reward_coverage_v2
import numpy as np

# POS = 1.5


class SeqUnit(object):
  def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab,field_vocab, position_vocab,
               target_vocab, field_concat, position_concat, fgate_enc, dual_att, encoder_add_pos, dual_att_add_pos,
               learning_rate, scope_name, name, start_token=2, stop_token=2, max_length=150, mode='train', dp=0.8,
               rl=False, loss_alpha=1, beam_size=1, lp_alpha=0.9, scaled_coverage_rw=False, out_vocab_mask=False):
    '''
    batch_size, hidden_size, emb_size, field_size, pos_size:
      size of batch; hidden layer; word/field/position embedding
    source_vocab, target_vocab, field_vocab, position_vocab:
      vocabulary size of encoder words; decoder words; field types; position
    field_concat
      bool values, whether concat field embedding to word embedding for encoder inputs
    position_concat:
      bool values, whether concat position embedding to word embedding and field embedding for encoder inputs
    fgate_enc, dual_att:
      bool values, whether use field-gating / dual attention or not
    encoder_add_pos, dual_att_add_pos:
      bool values, whether add position embedding to field-gating encoder / decoder with dual attention or not
    '''
    self.batch_size 		  = batch_size
    self.hidden_size 		  = hidden_size
    self.emb_size 			  = emb_size
    self.field_size 		  = field_size
    self.pos_size 			  = pos_size
    self.uni_size 			  = emb_size if not field_concat else emb_size+field_size
    self.uni_size 			  = self.uni_size if not position_concat else self.uni_size+2*pos_size
    self.field_encoder_size   = field_size if not encoder_add_pos else field_size+2*pos_size
    self.field_att_size 	    = field_size if not dual_att_add_pos  else field_size+2*pos_size
    self.source_vocab 	 = source_vocab
    self.target_vocab 	 = target_vocab
    self.field_vocab 	   = field_vocab
    self.position_vocab  = position_vocab
    self.grad_clip 		 = 5.0
    self.start_token	 = start_token
    self.stop_token 	 = stop_token
    self.max_length 	 = max_length
    self.scope_name 	 = scope_name
    self.name 			   = name
    self.field_concat 	 = field_concat
    self.position_concat = position_concat
    self.fgate_enc 		   = fgate_enc
    self.dual_att 		   = dual_att
    self.encoder_add_pos = encoder_add_pos
    self.dual_att_add_pos = dual_att_add_pos
    self.loss_alpha_value = loss_alpha
    self.units  = {}
    self.params = {}

    self.encoder_input 	= tf.placeholder(tf.int32, [None, None])
    self.encoder_field 	= tf.placeholder(tf.int32, [None, None])
    self.encoder_pos 	  = tf.placeholder(tf.int32, [None, None])
    self.encoder_rpos 	= tf.placeholder(tf.int32, [None, None])
    self.decoder_input 	= tf.placeholder(tf.int32, [None, None])
    self.encoder_len 	  = tf.placeholder(tf.int32, [None])
    self.decoder_len 	  = tf.placeholder(tf.int32, [None])
    self.decoder_output = tf.placeholder(tf.int32, [None, None])
    self.enc_mask 		  = tf.sign(tf.to_float(self.encoder_pos))
    if rl:
      self.decoder_input_sampled = tf.placeholder(tf.int32, [None, None])
      self.decoder_len_sampled   = tf.placeholder(tf.int32, [None])
      self.decoder_output_sampled = tf.placeholder(tf.int32, [None, None])
      if scaled_coverage_rw:
        self.rewards = tf.placeholder(tf.float32, [None, None])
      else:
        self.rewards = tf.placeholder(tf.float32, [None])
      self.loss_alpha = tf.constant(self.loss_alpha_value, dtype=tf.float32)

    with tf.variable_scope(scope_name):
      if self.fgate_enc:
        print('field-gated encoder LSTM')
        self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size,
                        self.field_encoder_size, 'encoder_select')
      else:
        print('normal encoder LSTM')
        self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')
      self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
      self.dec_out  = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output', out_vocab_mask=out_vocab_mask)

    self.units.update({'encoder_lstm': self.enc_lstm,'decoder_lstm': self.dec_lstm,
               'decoder_output': self.dec_out})

    # ====== embeddings ====== #
    with tf.device('/cpu:0'):
      with tf.variable_scope(scope_name):
        self.embedding = tf.get_variable('embedding', [self.source_vocab, self.emb_size])
        self.encoder_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
        self.decoder_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)

        # apply dropout on word embedding
        self.encoder_embed = tf.layers.dropout(self.encoder_embed, rate=dp, training=(mode == 'train'))
        self.decoder_embed = tf.layers.dropout(self.decoder_embed, rate=dp, training=(mode == 'train'))

        if rl:
          self.decoder_embed_sampled = tf.nn.embedding_lookup(self.embedding, self.decoder_input_sampled)

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.dual_att_add_pos:
          self.fembedding  = tf.get_variable('fembedding', [self.field_vocab, self.field_size])
          self.field_embed = tf.nn.embedding_lookup(self.fembedding, self.encoder_field)
          # apply dropout on field embedding
          self.field_embed = tf.layers.dropout(self.field_embed, rate=dp, training=(mode == 'train'))

          self.field_pos_embed = self.field_embed

          if self.field_concat:
            self.encoder_embed = tf.concat([self.encoder_embed, self.field_embed], 2)

        if self.position_concat or self.encoder_add_pos or self.dual_att_add_pos:
          self.pembedding = tf.get_variable('pembedding', [self.position_vocab, self.pos_size])
          self.rembedding = tf.get_variable('rembedding', [self.position_vocab, self.pos_size])
          self.pos_embed  = tf.nn.embedding_lookup(self.pembedding, self.encoder_pos)
          self.rpos_embed = tf.nn.embedding_lookup(self.rembedding, self.encoder_rpos)

          if self.position_concat:
            self.encoder_embed   = tf.concat([self.encoder_embed, self.pos_embed, self.rpos_embed], 2)
            self.field_pos_embed = tf.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)
          elif self.encoder_add_pos or self.dual_att_add_pos:
            self.field_pos_embed = tf.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)

    if self.field_concat or self.fgate_enc:
      self.params.update({'fembedding': self.fembedding})
    if self.position_concat or self.encoder_add_pos or self.dual_att_add_pos:
      self.params.update({'pembedding': self.pembedding})
      self.params.update({'rembedding': self.rembedding})

    self.params.update({'embedding': self.embedding})


    # ====== encoder ====== #
    if self.fgate_enc:
      print('field gated encoder used')
      en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed, self.encoder_len)
    else:
      print('normal encoder used')
      en_outputs, en_state = self.encoder(self.encoder_embed, self.encoder_len)

    # ====== decoder ====== #
    if self.dual_att:
      print('dual attention mechanism used')
      with tf.variable_scope(scope_name):
        self.att_layer = dualAttentionWrapper(self.hidden_size,
                                              self.hidden_size,
                                              self.field_att_size,
                                              en_outputs,
                                              self.field_pos_embed,
                                              "attention")
        self.units.update({'attention': self.att_layer})
    else:
      print("normal attention used")
      with tf.variable_scope(scope_name):
        self.att_layer = AttentionWrapper(self.hidden_size,
                                          self.hidden_size,
                                          en_outputs,
                                          "attention")
        self.units.update({'attention': self.att_layer})

    # ------ decoder for training ------ #
    de_outputs, de_state = self.decoder_t(en_state, self.decoder_embed, self.decoder_len)

    # ------ greedy decoder for testing ------ #
    self.g_tokens, self.atts = self.decoder_g(en_state)

    # ------ sampling decoder for testing ------ #
    self.multinomial_tokens, self.multinomial_atts = self.decoder_s(en_state)

    # ------ beam search decoder ------ #
    self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs = self.decoder_beam(en_state, beam_size, lp_alpha)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=de_outputs, labels=self.decoder_output)
    mask = tf.sign(tf.to_float(self.decoder_output))
    losses = mask * losses
    self.loss_mle = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
    self.mean_loss = self.loss_mle

    if rl:
      de_outputs_sampled, de_state_sampled = self.decoder_t(en_state, self.decoder_embed_sampled, self.decoder_len_sampled)
      neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=de_outputs_sampled, labels=self.decoder_output_sampled)
      mask_rl = tf.sign(tf.to_float(self.decoder_output_sampled))
      neg_log_prob = mask_rl * neg_log_prob
      rewards = self.rewards if scaled_coverage_rw else tf.expand_dims(self.rewards, axis=-1)
      self.loss_rl = tf.reduce_mean(neg_log_prob * rewards)  # reward guided loss
      self.mean_loss = self.loss_alpha*self.loss_mle + (1-self.loss_alpha) * self.loss_rl

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def encoder(self, inputs, inputs_len):
    batch_size = tf.shape(self.encoder_input)[0]
    max_time = tf.shape(self.encoder_input)[1]
    hidden_size = self.hidden_size

    time = tf.constant(0, dtype=tf.int32)
    h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
        tf.zeros([batch_size, hidden_size], dtype=tf.float32))
    f0 = tf.zeros([batch_size], dtype=tf.bool)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)

    # inputs: embeddings [batch_size, max_time, embedding_size]
    # inputs_ta: [max_time, batch_size, embedding_size]
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

    '''
      tf.while_loop(cond, body, loop_vars):
        cond is a callable returning a boolean scalar tensor. 
      body is a callable returning a (possibly nested) tuple, namedtuple or list of
        tensors of the same arity (length and structure) and types as loop_vars. 
      loop_vars is a (possibly nested) tuple, namedtuple or list of tensors that is
        passed to both cond and body.
      cond and body both take as many arguments as there are loop_vars.
    '''

    def loop_fn(t, x_t, s_t, emit_ta, finished):
      o_t, s_nt = self.enc_lstm(x_t, s_t, finished)
      emit_ta = emit_ta.write(t, o_t)
      finished = tf.greater_equal(t+1, inputs_len)
      '''
        tf.cond: Return true_fn() if the predicate pred is true else false_fn()
        tf.reduce_all: "logical and" 
        tf.greater_equal: (x >= y) element-wise
      '''
      x_nt = tf.cond(tf.reduce_all(finished),
               lambda: tf.zeros([batch_size, self.uni_size], dtype=tf.float32),
               lambda: inputs_ta.read(t+1))
      return t+1, x_nt, s_nt, emit_ta, finished

    _, _, state, emit_ta, _ = tf.while_loop(
      cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
      body=loop_fn,
      loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))

    outputs = tf.transpose(emit_ta.stack(), [1,0,2])
    return outputs, state

  def fgate_encoder(self, inputs, fields, inputs_len):
    batch_size = tf.shape(self.encoder_input)[0]
    max_time = tf.shape(self.encoder_input)[1]
    hidden_size = self.hidden_size

    time = tf.constant(0, dtype=tf.int32)
    h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
        tf.zeros([batch_size, hidden_size], dtype=tf.float32))
    f0 = tf.zeros([batch_size], dtype=tf.bool)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
    fields_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    fields_ta = fields_ta.unstack(tf.transpose(fields, [1,0,2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

    def loop_fn(t, x_t, d_t, s_t, emit_ta, finished):
      o_t, s_nt = self.enc_lstm(x_t, d_t, s_t, finished)
      emit_ta = emit_ta.write(t, o_t)
      finished = tf.greater_equal(t+1, inputs_len)
      x_nt = tf.cond(tf.reduce_all(finished),
               lambda: tf.zeros([batch_size, self.uni_size], dtype=tf.float32),
               lambda: inputs_ta.read(t+1))
      d_nt = tf.cond(tf.reduce_all(finished),
                 lambda: tf.zeros([batch_size,self.field_att_size],dtype=tf.float32),
                 lambda: fields_ta.read(t+1))
      return t+1, x_nt, d_nt, s_nt, emit_ta, finished

    _, _, _, state, emit_ta, _ = tf.while_loop(
      cond=lambda _1, _2, _3, _4, _5, finished:tf.logical_not(tf.reduce_all(finished)),
      body=loop_fn,
      loop_vars=(time, inputs_ta.read(0), fields_ta.read(0), h0, emit_ta, f0))

    outputs = tf.transpose(emit_ta.stack(), [1,0,2])
    return outputs, state


  def decoder_t(self, initial_state, inputs, inputs_len):
    batch_size = tf.shape(self.decoder_input)[0]
    # max_time = tf.shape(self.decoder_input)[1]
    max_time = tf.shape(inputs)[1]
    encoder_len = tf.shape(self.encoder_input)[1]

    time = tf.constant(0, dtype=tf.int32)
    h0 = initial_state
    f0 = tf.zeros([batch_size], dtype=tf.bool)
    x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

    def loop_fn(t, x_t, s_t, emit_ta, finished):
      o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
      o_t, _ = self.att_layer(o_t)
      o_t = self.dec_out(o_t, finished)
      emit_ta = emit_ta.write(t, o_t)
      finished = tf.greater_equal(t, inputs_len)
      x_nt = tf.cond(tf.reduce_all(finished),
               lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
               lambda: inputs_ta.read(t))
      return t+1, x_nt, s_nt, emit_ta, finished

    _, _, state, emit_ta,  _ = tf.while_loop(
      cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
      body=loop_fn,
      loop_vars=(time, x0, h0, emit_ta, f0))

    outputs = tf.transpose(emit_ta.stack(), [1,0,2])
    return outputs, state

  def decoder_g(self, initial_state):
    batch_size  = tf.shape(self.encoder_input)[0]
    encoder_len = tf.shape(self.encoder_input)[1]

    time 	= tf.constant(0, dtype=tf.int32)
    h0 		= initial_state
    f0 		= tf.zeros([batch_size], dtype=tf.bool)
    x0 		= tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    att_ta 	= tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

    def loop_fn(t, x_t, s_t, emit_ta, att_ta, finished):
      o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
      o_t, w_t = self.att_layer(o_t)
      o_t = self.dec_out(o_t, finished)
      emit_ta = emit_ta.write(t, o_t)
      att_ta = att_ta.write(t, w_t)
      next_token = tf.argmax(o_t, 1)
      x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
      finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
      finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
      return t+1, x_nt, s_nt, emit_ta, att_ta, finished

    _, _, state, emit_ta, att_ta, _ = tf.while_loop(
      cond=lambda _1, _2, _3, _4, _5, finished:tf.logical_not(tf.reduce_all(finished)),
      body=loop_fn,
      loop_vars=(time, x0, h0, emit_ta, att_ta, f0))

    outputs = tf.transpose(emit_ta.stack(), [1,0,2])
    pred_tokens = tf.argmax(outputs, 2)
    atts = att_ta.stack()
    return pred_tokens, atts

  def decoder_s(self, initial_state):
    """categorical sampling decoder"""
    batch_size  = tf.shape(self.encoder_input)[0]
    encoder_len = tf.shape(self.encoder_input)[1]

    sample_shape = tf.constant(1, dtype=tf.int32)
    time 	= tf.constant(0, dtype=tf.int32)
    h0 		= initial_state
    f0 		= tf.zeros([batch_size], dtype=tf.bool)
    t0    = tf.fill([batch_size], self.start_token)
    x0 		= tf.nn.embedding_lookup(self.embedding, t0)
    pred_ta = tf.TensorArray(dtype=tf.int32, dynamic_size=True, size=0)
    att_ta 	= tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

    def loop_fn(t, x_t, s_t, pred_ta, att_ta, finished, prev_token):
      o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
      o_t, w_t = self.att_layer(o_t)
      o_t = self.dec_out(o_t, finished)
      att_ta = att_ta.write(t, w_t)
      sampler = tf.contrib.distributions.Categorical(logits=o_t)
      next_token = sampler.sample(sample_shape=sample_shape)
      next_token = tf.squeeze(next_token, 0)
      pred_ta = pred_ta.write(t, next_token)
      x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
      finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
      finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
      return t+1, x_nt, s_nt, pred_ta, att_ta, finished, next_token

    _, _, state, pred_ta, att_ta, _, next_token = tf.while_loop(
      cond=lambda _1, _2, _3, _4, _5, finished, _6:tf.logical_not(tf.reduce_all(finished)),
      body=loop_fn,
      loop_vars=(time, x0, h0, pred_ta, att_ta, f0, t0))

    pred_tokens = tf.transpose(pred_ta.stack(), [1,0])
    atts = att_ta.stack()
    return pred_tokens, atts

  def decoder_beam(self, initial_state, beam_size, lp_alpha):

    def beam_init():
      # return beam_seqs_1 beam_probs_1 cand_seqs_1 cand_prob_1 next_states time
      time_1 = tf.constant(1, dtype=tf.int32)
      beam_seqs_0 = tf.constant([[self.start_token]]*beam_size)
      beam_probs_0 = tf.constant([0.]*beam_size)

      cand_seqs_0 = tf.constant([[self.start_token]])
      cand_probs_0 = tf.constant([-3e38])

      beam_seqs_0.set_shape((None, None))
      beam_probs_0.set_shape((None,))
      cand_seqs_0.set_shape((None, None))
      cand_probs_0.set_shape((None,))

      inputs = [self.start_token]
      x_t = tf.nn.embedding_lookup(self.embedding, inputs)
      # print(x_t.get_shape().as_list())
      o_t, s_nt = self.dec_lstm(x_t, initial_state)
      o_t, w_t = self.att_layer(o_t) # attention weights
      o_t = self.dec_out(o_t)
      # print(s_nt[0].get_shape().as_list())
      # print(w_t.get_shape().as_list())
      # initial_state = tf.reshape(initial_state, [1,-1])
      logprobs2d = tf.nn.log_softmax(o_t)
      total_probs = logprobs2d + tf.reshape(beam_probs_0, [-1, 1])
      total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [1, self.stop_token]),
                                     tf.tile([[-3e38]], [1, 1]),
                                     tf.slice(total_probs, [0, self.stop_token + 1],
                                              [1, self.target_vocab - self.stop_token - 1])
                                     ], 1)
      flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
      # print(flat_total_probs.get_shape().as_list())

      beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)
      next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

      next_bases = tf.floordiv(top_indices, self.target_vocab)
      next_mods = tf.mod(top_indices, self.target_vocab)

      next_beam_seqs = tf.concat([tf.gather(beam_seqs_0, next_bases),
                    tf.reshape(next_mods, [-1, 1])], 1)

      cand_seqs_pad = tf.pad(cand_seqs_0, [[0, 0], [0, 1]])
      beam_seqs_EOS = tf.pad(beam_seqs_0, [[0, 0], [0, 1]])
      new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0)
      # print(new_cand_seqs.get_shape().as_list())

      EOS_probs = tf.slice(total_probs, [0, self.stop_token], [beam_size, 1])
      new_cand_probs = tf.concat([cand_probs_0, tf.reshape(EOS_probs, [-1])], 0)
      cand_k = tf.minimum(tf.size(new_cand_probs), beam_size)
      next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
      next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

      part_state_0 = tf.reshape(tf.stack([s_nt[0]]*beam_size),
                         [beam_size, self.hidden_size])
      part_state_1 = tf.reshape(tf.stack([s_nt[1]]*beam_size),
                         [beam_size, self.hidden_size])
      part_state_0.set_shape((None, None))
      part_state_1.set_shape((None, None))
      next_states = (part_state_0, part_state_1)
      # print(next_states[0].get_shape().as_list())
      return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_states, time_1

    beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1 = beam_init()
    beam_seqs_1.set_shape((None, None))
    beam_probs_1.set_shape((None,))
    cand_seqs_1.set_shape((None, None))
    cand_probs_1.set_shape((None,))
    # states_1.set_shape((2, None, self.hidden_size))
    def beam_step(beam_seqs, beam_probs, cand_seqs, cand_probs, states, time):
      '''
      beam_seqs : [beam_size, time]
      beam_probs: [beam_size, ]
      cand_seqs : [beam_size, time]
      cand_probs: [beam_size, ]
      states : [beam_size * hidden_size, beam_size * hidden_size]
      '''
      inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [beam_size, 1]), [beam_size])
      # print(inputs.get_shape().as_list()
      x_t = tf.nn.embedding_lookup(self.embedding, inputs)
      # print(x_t.get_shape().as_list())
      o_t, s_nt = self.dec_lstm(x_t, states)
      o_t, w_t = self.att_layer(o_t)
      o_t = self.dec_out(o_t)
      logprobs2d = tf.nn.log_softmax(o_t)
      # print(logprobs2d.get_shape().as_list())
      total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
      # print(total_probs.get_shape().as_list())
      total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [beam_size, self.stop_token]),
                                     tf.tile([[-3e38]], [beam_size, 1]),
                                     tf.slice(total_probs, [0, self.stop_token + 1],
                                              [beam_size,self.target_vocab - self.stop_token - 1])
                                     ], 1)
      # print(total_probs_noEOS.get_shape().as_list())
      flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
      # print(flat_total_probs.get_shape().as_list())

      beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)

      # length_penalty = tf.pow(tf.cast(time, tf.float32), lp_alpha)
      # flat_total_probs_normalized = flat_total_probs / length_penalty
      # next_beam_probs_normalized, top_indices = tf.nn.top_k(flat_total_probs_normalized, k=beam_k)
      # next_beam_probs = next_beam_probs_normalized * length_penalty

      next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)
      # print(next_beam_probs.get_shape().as_list())

      next_bases = tf.floordiv(top_indices, self.target_vocab)
      next_mods = tf.mod(top_indices, self.target_vocab)
      # print(next_mods.get_shape().as_list())

      next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases),
                    tf.reshape(next_mods, [-1, 1])], 1)
      next_states = (tf.gather(s_nt[0], next_bases), tf.gather(s_nt[1], next_bases))
      # print(next_beam_seqs.get_shape().as_list())
      next_beam_seqs.set_shape((None, None))

      cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
      beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
      new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0)
      # print(new_cand_seqs.get_shape().as_list())
      new_cand_seqs.set_shape((None, None))

      EOS_probs = tf.slice(total_probs, [0, self.stop_token], [beam_size, 1])
      new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], 0)
      cand_k = tf.minimum(tf.size(new_cand_probs), beam_size)
      next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
      next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

      return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_states, time+1

    def beam_cond(beam_probs, beam_seqs, cand_probs, cand_seqs, state, time):
      length = (tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs))
      return tf.logical_and(length, tf.less(time, 60))
      # return tf.less(time, 18)

    loop_vars = [beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1]
    ret_vars = tf.while_loop(
      cond=beam_cond,
      body=beam_step,
      loop_vars=loop_vars,
      shape_invariants=[
        tf.TensorShape([None, None]),
        beam_probs_1.get_shape(),
        tf.TensorShape([None, None]),
        cand_probs_1.get_shape(),
        (tf.TensorShape([None, None]), tf.TensorShape([None, None])),
        time_1.get_shape()
      ],
      back_prop=False)
    beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all, _, time_all = ret_vars

    return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

  def train(self, x, sess, train_box_val, bc,
            rl=False, vocab=None, neg=False, discount=0.0,
            sampling=False, self_critic=False,
            accumulator=None, accumulator_sampled=None, counter=0,
            bleu_rw=False, coverage_rw=False, positive_reward_only=False, scaled_coverage_rw=False):
    if rl:
      return self.train_rl(x, sess, train_box_val, bc,
                         vocab=vocab, neg=neg, discount=discount,
                         sampling=sampling, self_critic=self_critic,
                         accumulator=accumulator, accumulator_sampled=accumulator_sampled, counter=counter,
                         bleu_rw=bleu_rw, coverage_rw=coverage_rw,
                         positive_reward_only=positive_reward_only, scaled_coverage_rw=scaled_coverage_rw)
    else:
      return True, self.train_mle(x, sess), 0, 0, None, None, 0


  def train_mle(self, x, sess):
    feed_dict = {self.encoder_input:  x['enc_in'],
                 self.encoder_len:    x['enc_len'],
                 self.decoder_input:  x['dec_in'],
                 self.decoder_len: 	  x['dec_len'],
                 self.decoder_output: x['dec_out']}

    if self.field_concat:
      feed_dict[self.encoder_field] = x['enc_fd']

    if self.position_concat:
      feed_dict[self.encoder_pos] = x['enc_pos']
      feed_dict[self.encoder_rpos] = x['enc_rpos']

    loss, _ = sess.run([self.mean_loss, self.train_op], feed_dict=feed_dict)

    return loss

  def train_rl(self, batch_data, sess, train_box_val, bc,
               vocab=None, neg=False, discount=0.0,
               sampling=False, self_critic=False,
               accumulator=None, accumulator_sampled=None, counter=0,
               bleu_rw=False, coverage_rw=False, positive_reward_only=False,
               scaled_coverage_rw=False):

    # start_time = time.time()
    if vocab is None: raise ValueError("vocab cannot be None")
    if not (bleu_rw or coverage_rw): raise ValueError("one of bleu_rw and coverage_rw has to be true")
    if self_critic: sampling = True

    box_ids          = batch_data['enc_in']
    sample_indices   = batch_data['indices']
    train_box_batch = train_box_val[sample_indices]

    def _replace_unk(target_prediction_ids, target_atts):
      """replace UNK with input word with highest attention weight"""
      batch = 0
      target_atts = np.squeeze(target_atts)
      real_sum_list, summary_len, real_ids_list = [], [], []
      for pred in np.array(target_prediction_ids):

        real_sum, real_ids = [], []

        pred = list(pred)  # 2 is eos, trim the output to the end of sentence
        if 2 in pred:
          pred = pred[:pred.index(2)] if pred[0] != 2 else [2]

        summary_len.append(len(pred))

        for idx, tid in enumerate(pred):
          if tid == 3:
            box_length = len(train_box_batch[batch])
            max_att = np.argmax(target_atts[idx, :box_length, batch])
            sub_id = box_ids[batch][max_att]
            sub = train_box_batch[batch][max_att]

            real_sum.append(sub)
            real_ids.append(sub_id)
          else:
            real_sum.append(vocab.id2word(tid))
            real_ids.append(tid)

        real_sum_list.append([x for x in real_sum])
        real_ids_list.append(real_ids)

        batch += 1

      summary_len = np.array(summary_len, dtype=np.int32)
      max_summary_len = max([len(x) for x in real_ids_list])
      dec_in_sampled = np.array([ids + [0] * (max_summary_len - len(ids)) for ids in real_ids_list], dtype=np.float32)
      dec_out_sampled = np.array([ids + [2] + [0] * (max_summary_len - len(ids)) for ids in real_ids_list], dtype=np.float32)

      return real_sum_list, real_ids_list, summary_len, dec_in_sampled, dec_out_sampled, max_summary_len

    rewards_bleu = np.zeros(box_ids.shape[0], dtype=np.float32)
    rewards_cov = np.zeros(box_ids.shape[0], dtype=np.float32)
    gold_summary_tks = batch_data['summaries']
    coverage_labels = batch_data['coverage_labels']

    '''predictions: [batch, length_decoder], atts: [length_decoder, length_encoder, batch]'''
    prediction_ids, atts = self.generate(batch_data, sess, sampling=sampling)
    real_sum_list, real_ids_list, summary_len, dec_in_sampled, dec_out_sampled, max_summary_len = _replace_unk(prediction_ids, atts)

    if bleu_rw:
      '''bleu_rewards'''
      bleu_rewards = get_reward_bleu(gold_summary_tks, real_sum_list)
      rewards_bleu += bleu_rewards
    if coverage_rw:
      '''coverage_rewards'''
      if scaled_coverage_rw:
        coverage_rewards, coverage_rewards_matrix = get_reward_coverage_v2(train_box_batch, gold_summary_tks,
                                                                           coverage_labels, real_sum_list,
                                                                           max_summary_len, bc)
      else:
        coverage_rewards = get_reward_coverage(gold_summary_tks, coverage_labels, real_sum_list, bc)
      rewards_cov += coverage_rewards

    if self_critic:
      '''predictions: [batch, length_decoder], atts: [length_decoder, length_encoder, batch]'''
      prediction_ids_greedy, atts_greedy = self.generate(batch_data, sess, sampling=False)
      real_sum_list_greedy, _, _, _, _, max_summary_len_greedy = _replace_unk(prediction_ids_greedy, atts_greedy)

      if bleu_rw:
        '''bleu_rewards'''
        bleu_rewards_baseline = get_reward_bleu(gold_summary_tks, real_sum_list_greedy)
        rewards_bleu -= bleu_rewards_baseline
      if coverage_rw:
        '''coverage_rewards'''
        if scaled_coverage_rw:
          coverage_rewards_baseline, _ = get_reward_coverage_v2(train_box_batch, gold_summary_tks, coverage_labels,
                                                                real_sum_list, max_summary_len, bc)
        else:
          coverage_rewards_baseline = get_reward_coverage(gold_summary_tks, coverage_labels, real_sum_list_greedy, bc)
        rewards_cov -= coverage_rewards_baseline

      rewards = rewards_bleu + rewards_cov
      if positive_reward_only:
        '''train with instances with positive rewards for self-critic'''

        # accumulate number of training instances
        counter += len([r for r in rewards if r > 0.0])

        # add gold instance pairs with positive rewards to accumulator
        for k, v in batch_data.iteritems():
          for l, r in zip(v.tolist(), rewards):
            if r > 0.0:
              accumulator[k].append(l)

        # add sampled instance pairs with positive rewards to accumulator_sampled
        for idx, (r, rb, rc, real_ids, l) in enumerate(zip(rewards, rewards_bleu, rewards_cov, real_ids_list, summary_len)):
          if r > 0.0:
            accumulator_sampled['real_ids_list'].append(real_ids)
            accumulator_sampled['summary_len'].append(l)
            accumulator_sampled['rewards'].append(r)
            if scaled_coverage_rw:
              accumulator_sampled['rewards_bleu'].append(rb)
              accumulator_sampled['rewards_cov'].append(rc)
              accumulator_sampled['reward_matrix'].append(coverage_rewards_matrix[idx])

        # go back and fetch the next batch if instance with positive rewards do not make a full batch yet
        if counter < self.batch_size:
          return False, 0.0, 0.0, 0.0, accumulator, accumulator_sampled, counter
        else:
          # padding sampled instances
          max_summary_len = max(accumulator_sampled['summary_len'])
          summary_len = np.array(accumulator_sampled['summary_len'], dtype=np.int32)
          real_ids_list = accumulator_sampled['real_ids_list']
          coverage_rewards_matrix = accumulator_sampled['reward_matrix']
          dec_in_sampled = np.array([ids + [0] * (max_summary_len - len(ids)) for ids in real_ids_list], dtype=np.float32)
          dec_out_sampled = np.array([ids + [2] + [0] * (max_summary_len - len(ids)) for ids in real_ids_list], dtype=np.float32)
          rewards = np.array(accumulator_sampled['rewards'], dtype=np.float32)
          if scaled_coverage_rw:
            # print(max_summary_len)
            # print([len(x) for x in coverage_rewards_matrix])
            # coverage_rewards_matrix = np.array([x + [0] * (max_summary_len + 1 - len(ids)) for ids in coverage_rewards_matrix], dtype=np.float32)
            coverage_rewards_matrix = np.array([np.pad(x, (0, max_summary_len + 1 - len(x)), 'constant') for x in coverage_rewards_matrix], dtype=np.float32)
            rewards_cov_exp = np.expand_dims(np.array(accumulator_sampled['rewards_cov'], dtype=np.float32), axis=-1)
            rewards_cov_scaled = coverage_rewards_matrix * rewards_cov_exp
            rewards_bleu_tiled = np.tile(np.array(accumulator_sampled['rewards_bleu'], dtype=np.float32),
                                       [coverage_rewards_matrix.shape[-1], 1]).T
            rewards = rewards_bleu_tiled + rewards_cov_scaled

          # padding gold instances: encoder side
          max_encoder_len_padded = max([len(v) for v in accumulator['enc_in']])
          max_encoder_len = max(accumulator['enc_len'])
          for key in ['enc_in', 'enc_fd', 'enc_pos', 'enc_rpos']:
            value = accumulator[key]
            value_padded = [v + [0] * (max_encoder_len - len(v)) for v in value]
            accumulator[key] = value_padded
          if max_encoder_len_padded > max_encoder_len:
            for key in ['enc_in', 'enc_fd', 'enc_pos', 'enc_rpos']:
              value = accumulator[key]
              value_cropped = [v[:max_encoder_len] for v in value]
              accumulator[key] = value_cropped

          # padding gold instances: decoder side
          max_decoder_len = max(accumulator['dec_len'])
          max_decoder_len_padded = max([len(i) for i in accumulator['dec_in']])
          value = accumulator['dec_in']
          value_padded = [v + [0] * (max_decoder_len - len(v)) for v in value]
          accumulator['dec_in'] = value_padded
          value = accumulator['dec_out']
          value_padded = [v + [0] * (max_decoder_len + 1 - len(v)) for v in value]
          accumulator['dec_out'] = value_padded
          if max_decoder_len_padded > max_decoder_len:
            value = accumulator['dec_in']
            value_cropped = [v[:max_decoder_len] for v in value]
            accumulator['dec_in'] = value_cropped
            value = accumulator['dec_out']
            value_cropped = [v[:max_decoder_len+1] for v in value]
            accumulator['dec_out'] = value_cropped

          # convert to numpy arrays (not necessary since tf does that inside)
          accumulator_np = {}
          for k, v in accumulator.iteritems():
            accumulator_np[k] = np.array(v)

          loss, loss_mle, loss_rl,  _ = sess.run([self.mean_loss, self.loss_mle, self.loss_rl, self.train_op],
                                                 {self.encoder_input:  accumulator_np['enc_in'],
                                                  self.encoder_field:  accumulator_np['enc_fd'],
                                                  self.encoder_pos:    accumulator_np['enc_pos'],
                                                  self.encoder_rpos:   accumulator_np['enc_rpos'],
                                                  self.decoder_input:  accumulator_np['dec_in'],
                                                  self.encoder_len:    accumulator_np['enc_len'],
                                                  self.decoder_len: 	 accumulator_np['dec_len'],
                                                  self.decoder_output: accumulator_np['dec_out'],
                                                  self.decoder_input_sampled: dec_in_sampled,
                                                  self.decoder_len_sampled: summary_len,
                                                  self.decoder_output_sampled: dec_out_sampled,
                                                  self.rewards: rewards,
                                                  })

          # reset accumulator and accumulator_sampled
          accumulator = {'enc_in': [], 'enc_fd': [], 'enc_pos': [], 'enc_rpos': [], 'enc_len': [],
                         'dec_in': [], 'dec_len': [], 'dec_out': [],
                         'indices': [], 'summaries': [], 'coverage_labels': []}
          accumulator_sampled = {'rewards': [], 'real_ids_list': [], 'summary_len': [],
                                 'reward_matrix': [], 'rewards_bleu': [], 'rewards_cov': []}

          return True, loss, loss_mle, loss_rl, accumulator, accumulator_sampled, 0

    loss, loss_mle, loss_rl, _ = sess.run([self.mean_loss, self.loss_mle, self.loss_rl, self.train_op],
                                          {self.encoder_input:  batch_data['enc_in'],
                                           self.encoder_field:  batch_data['enc_fd'],
                                           self.encoder_pos:    batch_data['enc_pos'],
                                           self.encoder_rpos:   batch_data['enc_rpos'],
                                           self.decoder_input:  batch_data['dec_in'],
                                           self.encoder_len:    batch_data['enc_len'],
                                           self.decoder_len:    batch_data['dec_len'],
                                           self.decoder_output: batch_data['dec_out'],
                                           self.decoder_input_sampled: dec_in_sampled,
                                           self.decoder_len_sampled: summary_len,
                                           self.decoder_output_sampled: dec_out_sampled,
                                           self.rewards: rewards,
                                           })

    return True, loss, loss_mle, loss_rl, None, None, 0

  def evaluate(self, x, sess):

    feed_dict = {self.encoder_input:  x['enc_in'],
                 self.encoder_len:    x['enc_len'],
                 self.decoder_input:  x['dec_in'],
                 self.decoder_len: 	  x['dec_len'],
                 self.decoder_output: x['dec_out']}

    if self.field_concat:
      feed_dict[self.encoder_field] = x['enc_fd']

    if self.position_concat:
      feed_dict[self.encoder_pos] = x['enc_pos']
      feed_dict[self.encoder_rpos] = x['enc_rpos']

    return sess.run([self.loss_mle], feed_dict=feed_dict)

  def generate(self, x, sess, sampling=False):
    ops = [self.g_tokens, self.atts] if not sampling else [self.multinomial_tokens, self.multinomial_atts]

    feed_dict = {self.encoder_input:  x['enc_in'],
                 self.encoder_len:    x['enc_len']}

    if self.field_concat:
      feed_dict[self.encoder_field] = x['enc_fd']

    if self.position_concat:
      feed_dict[self.encoder_pos] = x['enc_pos']
      feed_dict[self.encoder_rpos] = x['enc_rpos']

    predictions, atts = sess.run(ops, feed_dict=feed_dict)

    return predictions, atts

  def generate_beam(self, x, sess):
    # beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

    feed_dict = {self.encoder_input: x['enc_in'],
                 self.encoder_len: x['enc_len']}

    if self.field_concat:
      feed_dict[self.encoder_field] = x['enc_fd']

    if self.position_concat:
      feed_dict[self.encoder_pos] = x['enc_pos']
      feed_dict[self.encoder_rpos] = x['enc_rpos']

    beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all = sess.run(
             [self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs], feed_dict = feed_dict)

    return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

  def save(self, path):
    for u in self.units:
      self.units[u].save(path+u+".pkl")
    param_values = {}
    for param in self.params:
      param_values[param] = self.params[param].eval()
    with open(path+self.name+".pkl", 'wb') as f:
      pickle.dump(param_values, f, True)

  def load(self, path):
    for u in self.units:
      self.units[u].load(path+u+".pkl")
    param_values = pickle.load(open(path+self.name+".pkl", 'rb'))
    for param in param_values:
      self.params[param].load(param_values[param])
