#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from DataLoader import DataLoader

class DataLoader_t2s(DataLoader):
  def __init__(self, data_dir, data_dir_ori, limits):
    self.train_data_path = [data_dir + '/train/skeleton/train.summary.id.skeleton',
                            data_dir + '/train/skeleton/train.box.val.id.filtered',
                            data_dir + '/train/skeleton/train.box.lab.id.filtered',
                            data_dir + '/train/skeleton/train.box.pos.filtered',
                            data_dir + '/train/skeleton/train.box.rpos.filtered',
                            data_dir + '/train/skeleton/train.coverage.filtered',
                            data_dir + '/train/skeleton/train.summary.skeleton',
                            data_dir + '/train/skeleton/train.box.val.filtered']

    self.test_data_path  = [data_dir + '/test/skeleton/test.summary.id.skeleton',
                            data_dir + '/test/skeleton/test.box.val.id.filtered',
                            data_dir + '/test/skeleton/test.box.lab.id.filtered',
                            data_dir + '/test/skeleton/test.box.pos.filtered',
                            data_dir + '/test/skeleton/test.box.rpos.filtered',
                            data_dir + '/test/skeleton/test.coverage.filtered',
                            data_dir + '/test/skeleton/test.summary.skeleton',
                            data_dir + '/test/skeleton/test.box.val.filtered']

    self.dev_data_path   = [data_dir + '/valid/skeleton/valid.summary.id.skeleton',
                            data_dir + '/valid/skeleton/valid.box.val.id.filtered',
                            data_dir + '/valid/skeleton/valid.box.lab.id.filtered',
                            data_dir + '/valid/skeleton/valid.box.pos.filtered',
                            data_dir + '/valid/skeleton/valid.box.rpos.filtered',
                            data_dir + '/valid/skeleton/valid.coverage.filtered',
                            data_dir + '/valid/skeleton/valid.summary.skeleton',
                            data_dir + '/valid/skeleton/valid.box.val.filtered']

    self.limits 	  = limits
    self.man_text_len = 100
    start_time 		  = time.time()

    print('Reading datasets from ...')
    print(data_dir)
    self.train_set = self.load_data(self.train_data_path)
    self.test_set  = self.load_data(self.test_data_path)
    self.dev_set   = self.load_data(self.dev_data_path)
    print('Reading datasets consumes %.3f seconds' % (time.time() - start_time))

  # def load_data(self, path, filter=False):
  #     return super(DataLoader_t2s, self).load_data(path, filter=True)

