#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from DataLoader import DataLoader
import numpy as np

MAX = 64
MIN = 2

class DataLoader_s2t(object):
  def __init__(self, data_dir, data_dir_ori, limits):
    self.train_data_path = [data_dir + '/train/train.summary.id',
                            data_dir + '/train/skeleton/train.summary.id.skeleton',
                            data_dir_ori + '/train.summary']

    self.test_data_path  = [data_dir + '/test/test.summary.id',
                            data_dir + '/test/skeleton/test.summary.id.skeleton',
                            data_dir_ori + '/test.summary']

    self.dev_data_path   = [data_dir + '/valid/valid.summary.id',
                            data_dir + '/valid/skeleton/valid.summary.id.skeleton',
                            data_dir_ori + '/valid.summary']

    self.limits 	  = limits
    self.man_text_len = 100
    start_time 		  = time.time()

    print('Reading datasets from ...')
    print(data_dir)
    self.train_set = self.load_data(self.train_data_path)
    self.test_set  = self.load_data(self.test_data_path)
    self.dev_set   = self.load_data(self.dev_data_path)
    print('Reading datasets consumes %.3f seconds' % (time.time() - start_time))

  def load_data(self, path, filter=True):
      print(path)
      summary_id_path, text_path, summary_tk_path = path

      summary_ids = open(summary_id_path, 'r').read().strip().split('\n')
      summary_tks = open(summary_tk_path, 'r').read().strip().split('\n')
      texts = open(text_path, 'r').read().strip().split('\n')

      if self.limits > 0:
          summary_ids = summary_ids[:self.limits]
          summary_tks = summary_tks[:self.limits]
          texts = texts[:self.limits]

      reserved_indices = []
      if filter:
          summary_ids_out, texts_out, summary_tks_out = [], [], []

          for idx, (summary_id, summary_tk, text) in enumerate(zip(summary_ids, summary_tks, texts)):

              length = len(text.strip().split(' '))

              if (length > MAX or length < MIN):
                  continue
              else:
                  summary_ids_out.append(list(map(int, summary_id.strip().split(' '))))
                  summary_tks_out.append(summary_tk.strip().split(' '))
                  texts_out.append(list(map(int, text.strip().split(' '))))
                  reserved_indices.append(idx)
      else:
          summary_ids_out = [list(map(int, summary.strip().split(' '))) for summary in summary_ids]
          summary_tks_out = [summary.strip().split(' ') for summary in summary_tks]
          texts_out = [list(map(int, text.strip().split(' '))) for text in texts]

      return summary_ids_out, texts_out, summary_tks_out, reserved_indices

  def batch_iter(self, data, batch_size, shuffle=False):
    summary_ids, texts, summary_tks, _ = data

    data_size 	= len(summary_ids)
    print('data_size = %d'%data_size)
    num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
                          else int(data_size / batch_size) + 1

    print('num_batches = %d'%num_batches)
    indices = np.arange(data_size)
    if shuffle:
      indices = np.random.permutation(indices)
      summary_ids = np.array(summary_ids)[indices]
      summary_tks = np.array(summary_tks)[indices]
      texts 			= np.array(texts)[indices]

    for batch_num in range(num_batches):
      start_index 	= batch_num * batch_size
      end_index 		= min((batch_num + 1) * batch_size, data_size)

      # pad to max length within a batch
      max_summary_len = max([len(s) for s in summary_ids[start_index:end_index]])
      max_text_len 	= max([len(s) for s in texts[start_index:end_index]])
      batch_data 		= {'enc_in': [], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[],
                       'enc_len':[], 'dec_in':[], 'dec_len':[], 'dec_out': [],
                       'indices':[], 'summaries':[], 'coverage_labels':[]}

      for summary_id, text, idxes, summary_tk in zip(
              summary_ids[start_index:end_index],
              texts[start_index:end_index],
              indices[start_index:end_index],
              summary_tks[start_index:end_index]):

        summary_len = len(summary_id)
        text_len 	= len(text)
        assert summary_len 	== len(summary_tk)
        gold 	  = summary_id + [2] + [0] * (max_summary_len - summary_len)
        summary_id = summary_id + [0] * (max_summary_len - summary_len)
        text 	  = text 	  + [0] * (max_text_len - text_len)

        if max_text_len > self.man_text_len:
          text 	 = text[:self.man_text_len]
          text_len = min(text_len, self.man_text_len)

        batch_data['enc_in'].append(text)
        batch_data['enc_len'].append(text_len)
        batch_data['dec_in'].append(summary_id)
        batch_data['dec_len'].append(summary_len)
        batch_data['dec_out'].append(gold)
        batch_data['indices'].append(idxes)
        batch_data['summaries'].append(summary_tk)

      batch_data_np = {}
      for k, v in batch_data.iteritems():
        batch_data_np[k] = np.array(v)

      yield batch_data_np