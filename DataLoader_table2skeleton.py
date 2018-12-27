#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import time
from DataLoader import DataLoader

class DataLoader_t2s(DataLoader):
  def __init__(self, data_dir, data_dir_ori, limits):
    self.train_data_path = [data_dir + '/train/skeleton/train.summary.id.skeleton',
                            data_dir + '/train/train.box.val.id',
                            data_dir + '/train/train.box.lab.id',
                            data_dir + '/train/train.box.pos',
                            data_dir + '/train/train.box.rpos',
                            data_dir + '/train/train.coverage',
                            data_dir + '/train/skeleton/train.summary.skeleton']

    self.test_data_path  = [data_dir + '/test/skeleton/test.summary.id.skeleton',
                            data_dir + '/test/test.box.val.id',
                            data_dir + '/test/test.box.lab.id',
                            data_dir + '/test/test.box.pos',
                            data_dir + '/test/test.box.rpos',
                            data_dir + '/test/test.coverage',
                            data_dir + '/test/skeleton/test.summary.skeleton']

    self.dev_data_path   = [data_dir + '/valid/skeleton/valid.summary.id.skeleton',
                            data_dir + '/valid/valid.box.val.id',
                            data_dir + '/valid/valid.box.lab.id',
                            data_dir + '/valid/valid.box.pos',
                            data_dir + '/valid/valid.box.rpos',
                            data_dir + '/valid/valid.coverage',
                            data_dir + '/valid/skeleton/valid.summary.skeleton']

    self.limits 	  = limits
    self.man_text_len = 100
    start_time 		  = time.time()

    print('Reading datasets from ...')
    print(data_dir)
    self.train_set = self.load_data(self.train_data_path)
    self.test_set  = self.load_data(self.test_data_path)
    self.dev_set   = self.load_data(self.dev_data_path)
    print('Reading datasets consumes %.3f seconds' % (time.time() - start_time))

  def load_data(self, path, filter=False):
      return super(DataLoader_t2s, self).load_data(path, filter=True)

