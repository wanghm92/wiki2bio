#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import tensorflow as tf
import time
import numpy as np


class DataLoader(object):
  def __init__(self, data_dir, data_dir_ori, limits):
    self.train_data_path = [data_dir + '/train/train.summary.id',
                            data_dir + '/train/train.box.val.id',
                            data_dir + '/train/train.box.lab.id',
                            data_dir + '/train/train.box.pos',
                            data_dir + '/train/train.box.rpos',
                            data_dir_ori + '/train.summary']

    self.test_data_path  = [data_dir + '/test/test.summary.id',
                            data_dir + '/test/test.box.val.id',
                            data_dir + '/test/test.box.lab.id',
                            data_dir + '/test/test.box.pos',
                            data_dir + '/test/test.box.rpos',
                            data_dir_ori + '/test.summary']

    self.dev_data_path   = [data_dir + '/valid/valid.summary.id',
                            data_dir + '/valid/valid.box.val.id',
                            data_dir + '/valid/valid.box.lab.id',
                            data_dir + '/valid/valid.box.pos',
                            data_dir + '/valid/valid.box.rpos',
                            data_dir_ori + '/valid.summary']
    self.limits 	  = limits
    self.man_text_len = 100
    start_time 		  = time.time()

    print('Reading datasets ...')
    print(data_dir)
    print(data_dir_ori)
    self.train_set = self.load_data(self.train_data_path)
    self.test_set  = self.load_data(self.test_data_path)
    self.dev_set   = self.load_data(self.dev_data_path)
    print('Reading datasets consumes %.3f seconds' % (time.time() - start_time))

  def load_data(self, path):
    summary_id_path, text_path, field_path, pos_path, rpos_path, summary_tk_path = path

    summary_ids = open(summary_id_path, 'r').read().strip().split('\n')
    summary_tks = open(summary_tk_path, 'r').read().strip().split('\n')
    texts 	  = open(text_path,    'r').read().strip().split('\n')
    fields 	  = open(field_path,   'r').read().strip().split('\n')
    poses 	  = open(pos_path,     'r').read().strip().split('\n')
    rposes 	  = open(rpos_path,    'r').read().strip().split('\n')

    if self.limits > 0:
      summary_ids = summary_ids[:self.limits]
      summary_tks = summary_tks[:self.limits]
      texts 	  = texts[:self.limits]
      fields    = fields[:self.limits]
      poses 	  = poses[:self.limits]
      rposes 	  = rposes[:self.limits]

    summary_ids = [list(map(int, summary.strip().split(' '))) for summary in summary_ids]
    summary_tks = [summary.strip().split(' ') for summary in summary_tks]
    texts 	  = [list(map(int, text.strip().split(' ')))    for text in texts]
    fields 	  = [list(map(int, field.strip().split(' ')))   for field in fields]
    poses 	  = [list(map(int, pos.strip().split(' ')))     for pos in poses]
    rposes 	  = [list(map(int, rpos.strip().split(' ')))    for rpos in rposes]

    return summary_ids, texts, fields, poses, rposes, summary_tks

  def batch_iter(self, data, batch_size, shuffle):
    summary_ids, texts, fields, poses, rposes, summary_tks = data
    data_size 	= len(summary_ids)
    num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
                          else int(data_size / batch_size) + 1

    print('num_batches = %d'%num_batches)
    indices = np.arange(data_size)
    if shuffle:
      indices = np.random.permutation(indices)
      summary_ids = np.array(summary_ids)[indices]
      summary_tks = np.array(summary_tks)[indices]
      texts 			= np.array(texts)[indices]
      fields 			= np.array(fields)[indices]
      poses 			= np.array(poses)[indices]
      rposes 			= np.array(rposes)[indices]

    for batch_num in range(num_batches):
      start_index 	= batch_num * batch_size
      end_index 		= min((batch_num + 1) * batch_size, data_size)

      # pad to max length within a batch
      max_summary_len = max([len(s) for s in summary_ids[start_index:end_index]])
      max_text_len 	= max([len(s) for s in texts[start_index:end_index]])
      batch_data 		= {'enc_in': [], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[],
                       'enc_len':[], 'dec_in':[], 'dec_len':[], 'dec_out': [],
                       'indices':[], 'summaries':[]}

      for summary_id, text, field, pos, rpos, idxes, summary_tk in zip(summary_ids[start_index:end_index],
                                                                       texts[start_index:end_index],
                                                                       fields[start_index:end_index],
                                                                       poses[start_index:end_index],
                                                                       rposes[start_index:end_index],
                                                                       indices[start_index:end_index],
                                                                       summary_tks[start_index:end_index]):
        summary_len = len(summary_id)
        text_len 	= len(text)
        pos_len 	= len(pos)
        rpos_len 	= len(rpos)
        assert summary_len 	== len(summary_tk)
        assert text_len == len(field)
        assert pos_len 	== len(field)
        assert rpos_len == pos_len
        gold 	  = summary_id + [2] + [0] * (max_summary_len - summary_len)
        summary_id = summary_id + [0] * (max_summary_len - summary_len)
        text 	  = text 	  + [0] * (max_text_len - text_len)
        field 	= field   + [0] * (max_text_len - text_len)
        pos 	  = pos 	  + [0] * (max_text_len - text_len)
        rpos  	= rpos 	  + [0] * (max_text_len - text_len)

        if max_text_len > self.man_text_len:
          text 	 = text[:self.man_text_len]
          field  = field[:self.man_text_len]
          pos    = pos[:self.man_text_len]
          rpos 	 = rpos[:self.man_text_len]
          text_len = min(text_len, self.man_text_len)

        batch_data['enc_in'].append(text)
        batch_data['enc_len'].append(text_len)
        batch_data['enc_fd'].append(field)
        batch_data['enc_pos'].append(pos)
        batch_data['enc_rpos'].append(rpos)
        batch_data['dec_in'].append(summary_id)
        batch_data['dec_len'].append(summary_len)
        batch_data['dec_out'].append(gold)
        batch_data['indices'].append(idxes)
        batch_data['summaries'].append(summary_tk)

      batch_data_np = {}
      for k, v in batch_data.iteritems():
        batch_data_np[k] = np.array(v)

      yield batch_data_np