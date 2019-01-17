from __future__ import print_function

import re, time, os, sys

suffix='data'
from os.path import expanduser
HOME = expanduser("~")
prepro_in = '%s/table2text_nlg/data/fieldgate_data/original_%s'%(HOME, suffix)
# prepro_in = '%s/table2text_nlg/data/dkb/wikibio_format_tokenized'%HOME

# TODO: replace open() with io.open(encoding='utf-8'), non-breaking space exists

def split_infobox():
  """
  extract box content, field type and position information from infoboxes from original_data
  *.box.val is the box content (token)
  *.box.lab is the field type for each token
  *.box.pos is the position counted from the begining of a field
  """
  bwfile = ["%s/train/train.box.val"%prepro_out,
        "%s/valid/valid.box.val"%prepro_out,
        "%s/test/test.box.val"%prepro_out]
  bffile = ["%s/train/train.box.lab"%prepro_out,
        "%s/valid/valid.box.lab"%prepro_out,
        "%s/test/test.box.lab"%prepro_out]
  bpfile = ["%s/train/train.box.pos"%prepro_out,
        "%s/valid/valid.box.pos"%prepro_out,
        "%s/test/test.box.pos"%prepro_out]
  boxes = ["%s/train.box"%prepro_in, "%s/valid.box"%prepro_in, "%s/test.box"%prepro_in]

  mixb_word, mixb_label, mixb_pos = [], [], []

  # datasets
  for fboxes in boxes:
    box = open(fboxes, "r").read().strip().split('\n')
    box_word, box_label, box_pos = [], [], []

    # each box
    for ib in box:
      item = ib.strip().split('\t')
      box_single_word, box_single_label, box_single_pos = [], [], []

      # within a box
      for it in item:
        # field, value pairs have already been splited with starting postion
        if len(it.split(':')) > 2: continue

        prefix, word = it.split(':')

        # ignore none values and empty field or values
        if '<none>' in word or word.strip()=='' or prefix.strip()=='': continue

        # '*' matches 0 or more repetitions;'$' matches the end of the string
        # re.sub(pattern, repl, string)
        new_label = re.sub("_[1-9]\d*$", "", prefix)

        if new_label.strip() == "": continue

        box_single_word.append(word)
        box_single_label.append(new_label)
        if re.search("_[1-9]\d*$", prefix):
          field_id = int(prefix.split('_')[-1])
          box_single_pos.append(field_id if field_id<=30 else 30)
        else:
          box_single_pos.append(1)

      box_word.append(box_single_word)
      box_label.append(box_single_label)
      box_pos.append(box_single_pos)
    mixb_word.append(box_word)
    mixb_label.append(box_label)
    mixb_pos.append(box_pos)

  for k, m in enumerate(mixb_word):
    with open(bwfile[k], "w+") as h:
      for items in m:
        for sens in items:
          h.write(str(sens) + " ")
        h.write('\n')
  for k, m in enumerate(mixb_label):
    with open(bffile[k], "w+") as h:
      for items in m:
        for sens in items:
          h.write(str(sens) + " ")
        h.write('\n')
  for k, m in enumerate(mixb_pos):
    with open(bpfile[k], "w+") as h:
      for items in m:
        for sens in items:
          h.write(str(sens) + " ")
        h.write('\n')

def reverse_pos():
  # get the position counted from the end of a field
  bpfile = ["%s/train/train.box.pos"%prepro_out,
        "%s/valid/valid.box.pos"%prepro_out,
        "%s/test/test.box.pos"%prepro_out]
  bwfile = ["%s/train/train.box.rpos"%prepro_out,
        "%s/valid/valid.box.rpos"%prepro_out,
        "%s/test/test.box.rpos"%prepro_out]
  for k, pos in enumerate(bpfile):
    box = open(pos, "r").read().strip().split('\n')
    reverse_pos = []
    for bb in box:
      pos = bb.split()
      tmp_pos = []
      single_pos = []
      for p in pos:
        if int(p) == 1 and len(tmp_pos) != 0:
          single_pos.extend(tmp_pos[::-1])
          tmp_pos = []
        tmp_pos.append(p)
      single_pos.extend(tmp_pos[::-1])
      reverse_pos.append(single_pos)
    with open(bwfile[k], 'w+') as bw:
      for item in reverse_pos:
        bw.write(" ".join(item) + '\n')

def check_generated_box():
  ftrain = ["%s/train/train.box.val"%prepro_out,
        "%s/train/train.box.lab"%prepro_out,
        "%s/train/train.box.pos"%prepro_out,
        "%s/train/train.box.rpos"%prepro_out]
  ftest  = ["%s/test/test.box.val"%prepro_out,
        "%s/test/test.box.lab"%prepro_out,
        "%s/test/test.box.pos"%prepro_out,
        "%s/test/test.box.rpos"%prepro_out]
  fvalid = ["%s/valid/valid.box.val"%prepro_out,
        "%s/valid/valid.box.lab"%prepro_out,
        "%s/valid/valid.box.pos"%prepro_out,
        "%s/valid/valid.box.rpos"%prepro_out]
  for case in [ftrain, ftest, fvalid]:
    vals = open(case[0], 'r').read().strip().split('\n')
    labs = open(case[1], 'r').read().strip().split('\n')
    poses = open(case[2], 'r').read().strip().split('\n')
    rposes = open(case[3], 'r').read().strip().split('\n')
    assert len(vals) == len(labs)
    assert len(poses) == len(labs)
    assert len(rposes) == len(poses)
    for val, lab, pos, rpos in zip(vals, labs, poses, rposes):
      vval = val.strip().split(' ')
      llab = lab.strip().split(' ')
      ppos = pos.strip().split(' ')
      rrpos = rpos.strip().split(' ')
      if len(vval) != len(llab) or len(llab) != len(ppos) or len(ppos) != len(rrpos):
        print(case)
        print(val)
        print(len(vval))
        print(len(llab))
        print(len(ppos))
        print(len(rrpos))
      assert len(vval) == len(llab)
      assert len(llab) == len(ppos)
      assert len(ppos) == len(rrpos)


def split_summary_for_rouge():
  bpfile = ["%s/test.summary"%prepro_in, "%s/valid.summary"%prepro_in]
  bwfile = ["%s/test/test_split_for_rouge/"%prepro_out,
        "%s/valid/valid_split_for_rouge/"%prepro_out]
  for i, fi in enumerate(bpfile):
    fread = open(fi, 'r')
    k = 0
    for line in fread:
      with open(bwfile[i] + 'gold_summary_' + str(k), 'w') as sw:
        sw.write(line.strip() + '\n')
      k += 1
    fread.close()



class Vocab(object):
  """vocabulary for words and field types"""
  def __init__(self):
    vocab = dict()
    vocab['PAD'] = 0
    vocab['START_TOKEN'] = 1
    vocab['END_TOKEN'] = 2
    vocab['UNK_TOKEN'] = 3
    cnt = 4
    with open("%s/word_vocab.txt"%prepro_in, "r") as v:
      for line in v:
        # remove '' from word list
        try:
          word = line.strip().split()[0]
        except IndexError:
          continue
        vocab[word] = cnt
        cnt += 1
    self._word2id = vocab
    self._id2word = {value: key for key, value in vocab.items()}

    key_map = dict()
    key_map['PAD'] = 0
    key_map['START_TOKEN'] = 1
    key_map['END_TOKEN'] = 2
    key_map['UNK_TOKEN'] = 3
    cnt = 4
    with open("%s/field_vocab.txt"%prepro_in, "r") as v:
      for line in v:
        key = line.strip().split()[0]
        key_map[key] = cnt
        cnt += 1
    self._key2id = key_map
    self._id2key = {value: key for key, value in key_map.items()}

  def word2id(self, word):
    ans = self._word2id[word] if word in self._word2id else 3
    return ans

  def id2word(self, id):
    ans = self._id2word[int(id)]
    return ans

  def key2id(self, key):
    ans = self._key2id[key] if key in self._key2id else 3
    return ans

  def id2key(self, id):
    ans = self._id2key[int(id)]
    return ans

def table2id():
  fsums = ['%s/train.summary'%prepro_in,
       '%s/test.summary'%prepro_in,
       '%s/valid.summary'%prepro_in]

  fvals = ['%s/train/train.box.val'%prepro_out,
       '%s/test/test.box.val'%prepro_out,
       '%s/valid/valid.box.val'%prepro_out]
  flabs = ['%s/train/train.box.lab'%prepro_out,
       '%s/test/test.box.lab'%prepro_out,
       '%s/valid/valid.box.lab'%prepro_out]
  fvals2id = ['%s/train/train.box.val.id'%prepro_out,
        '%s/test/test.box.val.id'%prepro_out,
        '%s/valid/valid.box.val.id'%prepro_out]
  flabs2id = ['%s/train/train.box.lab.id'%prepro_out,
        '%s/test/test.box.lab.id'%prepro_out,
        '%s/valid/valid.box.lab.id'%prepro_out]
  fsums2id = ['%s/train/train.summary.id'%prepro_out,
        '%s/test/test.summary.id'%prepro_out,
        '%s/valid/valid.summary.id'%prepro_out]

  vocab = Vocab()
  for k, ff in enumerate(fvals):
    fi = open(ff, 'r')
    fo = open(fvals2id[k], 'w')
    for line in fi:
      items = line.strip().split()
      fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
    fi.close()
    fo.close()
  for k, ff in enumerate(flabs):
    fi = open(ff, 'r')
    fo = open(flabs2id[k], 'w')
    for line in fi:
      items = line.strip().split()
      fo.write(" ".join([str(vocab.key2id(key)) for key in items]) + '\n')
    fi.close()
    fo.close()
  for k, ff in enumerate(fsums):
    fi = open(ff, 'r')
    fo = open(fsums2id[k], 'w')
    for line in fi:
      items = line.strip().split()
      fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
    fi.close()
    fo.close()


def preprocess():
  """
  We use a triple <f, p+, p-> to represent the field information of a token in the specific field.
  p+&p- are the position of the token in that field counted from the begining and the end of the field.
  For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
  (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
  """
  print("extracting token, field type and position info from original data ...")
  time_start = time.time()
  split_infobox()
  reverse_pos()
  duration = time.time() - time_start
  print("extract finished in %.3f seconds" % float(duration))

  print("spliting test and valid summaries for ROUGE evaluation ...")
  time_start = time.time()
  split_summary_for_rouge()
  duration = time.time() - time_start
  print("split finished in %.3f seconds" % float(duration))

  print("turning words and field types to ids ...")
  time_start = time.time()
  # field valuse and output sentences use the same embedding
  table2id()
  duration = time.time() - time_start
  print("idlization finished in %.3f seconds" % float(duration))

def make_dirs():
  dirs = ["results/", "results/res/", "results/evaluation/",
      "%s/"%prepro_out,
      "%s/train/"%prepro_out, "%s/test/"%prepro_out, "%s/valid/"%prepro_out,
      "%s/test/test_split_for_rouge/"%prepro_out,
      "%s/valid/valid_split_for_rouge/"%prepro_out]
  for d in dirs:
    if not os.path.isdir(d):
      os.mkdir(d)

if __name__ == '__main__':
  # prepro_out = 'processed_small'
  prepro_out = '%s/table2text_nlg/data/fieldgate_data/processed_dkb_tk'%HOME
  make_dirs()
  preprocess()
  check_generated_box()
  print("check done")
