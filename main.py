#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu
from __future__ import print_function

import sys, os, time, logging, json, tqdm, io, subprocess, math
import tensorflow as tf
import numpy as np
from SeqUnit import *
from DataLoader import DataLoader
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import *
from os.path import expanduser
HOME = expanduser("~")
sys.path.append('{}/bert_src/bert_as_service'.format(HOME))
# from service.client import BertClient
# bc = BertClient()
bc = None
last_best = 0.0
file_paths = {}

suffix='data'
prepro_in = '%s/table2text_nlg/data/fieldgate_data/original_%s'%(HOME, suffix)
prepro_out = '%s/table2text_nlg/data/fieldgate_data/processed_%s'%(HOME, suffix)

tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 20003, 'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 1480, 'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31, 'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 20003, 'vocabulary size')
tf.app.flags.DEFINE_integer("report", 500, 'report losses after some steps')
tf.app.flags.DEFINE_integer("eval_multi", 5, 'report valid results after some steps')
tf.app.flags.DEFINE_integer("max_to_keep", 5, 'maximum number of checkpoints to save')
tf.app.flags.DEFINE_float("learning_rate", 0.0003, 'learning rate')

tf.app.flags.DEFINE_float("alpha", 0.9, 'percentage of MLE loss')
tf.app.flags.DEFINE_float("discount", 0.0, 'discount factor for cummulative rewards, default: not use')
tf.app.flags.DEFINE_integer("beam", 1, "width of beam search")
tf.app.flags.DEFINE_boolean("rl", False, 'whether to add policy gradient')
tf.app.flags.DEFINE_boolean("neg", False, 'whether to use negative rewards')
tf.app.flags.DEFINE_boolean("sampling", False, 'whether to use multinomial categorical sampling for rl')
tf.app.flags.DEFINE_boolean("self_critic", False, 'whether to use self-critic')

tf.app.flags.DEFINE_boolean("load", False, 'whether to load model parameters')
tf.app.flags.DEFINE_boolean("rouge", False, 'whether to evaluate on ROUGE for validation')
tf.app.flags.DEFINE_integer("cnt", 0, 'directory k to load model from')
tf.app.flags.DEFINE_integer("limits", 0, 'max data set size')
tf.app.flags.DEFINE_string("mode", 'train', 'train or test')
tf.app.flags.DEFINE_string("prefix", 'temp', 'name your model')
tf.app.flags.DEFINE_string("dir", prepro_out, 'data set directory')
tf.app.flags.DEFINE_string("dir_ori", prepro_in, 'data set directory')

tf.app.flags.DEFINE_boolean("dual_attention",True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", False,'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", True,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position",True,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos",True,'position info in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos",True,'position info in dual attention decoder')

FLAGS = tf.app.flags.FLAGS

prefix = FLAGS.prefix
save_dir = '%s/table2text_nlg/output/fieldgate_output/results/res/%s/'%(HOME, prefix)
load_dir = save_dir + 'models/%s/'%FLAGS.cnt
save_file_dir = save_dir + 'src/'
# ckpt_dir = save_dir + 'ckpt/'
pred_dir = '%s/table2text_nlg/output/fieldgate_output/results/evaluation/%s/'%(HOME, prefix)

if not os.path.exists(save_dir):
  os.mkdir(save_dir)
if not os.path.exists(pred_dir):
  os.mkdir(pred_dir)
if not os.path.exists(save_file_dir):
  os.mkdir(save_file_dir)
pred_path = pred_dir + 'pred_summary_'
pred_beam_path = pred_dir + 'beam_summary_'

if FLAGS.rouge:
  gold_path_test = '%s/test/test_split_for_rouge/gold_summary_'%prepro_out
  gold_path_valid = '%s/valid/valid_split_for_rouge/gold_summary_'%prepro_out
else:
  gold_path_test = '%s/test.summary'%prepro_in
  gold_path_valid = '%s/valid.summary'%prepro_in

log_file   = save_dir + 'log'
flag_file  = save_dir + 'flags'
rank_file  = save_dir + 'ranking'
valid_path = "%s/valid/valid.box.val"%prepro_out
test_path  = "%s/test/test.box.val"%prepro_out
train_path  = "%s/train/train.box.val"%prepro_out
tb_path	   = save_dir + 'event/'
tfwriter = tf.summary.FileWriter(tb_path)

write_log('save_dir: %s'%save_dir, log_file)
write_log('pred_dir: %s'%pred_dir, log_file)

file_paths['gold_path_test']  = gold_path_test
file_paths['gold_path_valid'] = gold_path_valid
file_paths['save_dir']		    = save_dir
file_paths['save_file_dir']   = save_file_dir
file_paths['pred_dir']		    = pred_dir
file_paths['pred_path'] 	    = pred_path
file_paths['pred_beam_path']  = pred_beam_path
file_paths['log_file'] 		    = log_file
file_paths['flag_file'] 	    = flag_file
file_paths['rank_file'] 	    = rank_file
file_paths['valid_path'] 	    = valid_path
file_paths['test_path'] 	    = test_path

def train(sess, dataloader, model, saver, rl=FLAGS.rl):
  cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
  write_log("Start time: %s"%cur_time, log_file)
  write_log("#######################################################", log_file)
  current_flags = [(f,str(FLAGS.__getattr__(f))) for f in FLAGS.__flags]
  all_flags = merge_two_dicts(dict(current_flags), file_paths)
  with open(flag_file, 'w+') as fout:
    json.dump(all_flags, fout, sort_keys=True, indent=4)
    L.info('Flags written to %s'%flag_file)
  for f, v in current_flags:
    write_log(f + " = " + v, log_file)
  write_log("#######################################################", log_file)

  v = Vocab()
  loss, start_time = 0.0, time.time()
  loss_mle_sum, loss_rl_sum = 0.0, 0.0
  trainset = dataloader.train_set
  batch = 1

  if FLAGS.load:
    # batch = FLAGS.cnt*FLAGS.report + 1
    best_bleu = load_rankings(rank_file)
    print('Rankings loaded from %s'%rank_file)
    print('initial batch = %d'%(FLAGS.cnt*FLAGS.report))
  else:
    best_bleu = dict([(i, ('ep0', 0, 0)) for i in range(FLAGS.max_to_keep)])

  train_box_val = None
  if rl:
    train_box_val = open(train_path, 'r').read().strip().split('\n')
    train_box_val = [list(t.strip().split()) for t in train_box_val]

  for e in range(FLAGS.epoch):
    L.info('Training Epoch --%2d--\n'%e)
    for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
      model_returns = model.train(x, sess, train_box_val, bc,
                                  rl=rl, vocab=v, neg=FLAGS.neg, discount=FLAGS.discount,
                                  sampling=FLAGS.sampling, self_critic=FLAGS.self_critic)
      if rl:
        loss_mean, loss_mle, loss_rl = model_returns
        loss_mle_sum += loss_mle
        loss_rl_sum += loss_rl
      else:
        loss_mean = model_returns
      loss += loss_mean
      progress_bar(batch%(FLAGS.report * FLAGS.eval_multi), (FLAGS.report * FLAGS.eval_multi))

      if (batch % FLAGS.report == 0):
        cnt = batch//FLAGS.report + FLAGS.cnt
        cost_time = time.time() - start_time
        avg_loss = loss / (FLAGS.report*1.0)
        write_log("%d : avg_loss = %.3f, time = %.3f"% (cnt, avg_loss, cost_time), log_file)
        tfwriter.add_summary(tf_summary_entry('train/loss', avg_loss), cnt)
        loss, start_time = 0.0, time.time()
        if rl:
          avg_loss_mle = loss_mle_sum / (FLAGS.report*1.0)
          avg_loss_rl = loss_rl_sum / (FLAGS.report*1.0)
          write_log("%d : avg_loss_mle = %.3f, avg_loss_rl = %.3f, time = %.3f"
                  %(cnt, avg_loss_mle, avg_loss_rl, cost_time), log_file)
          tfwriter.add_summary(tf_summary_entry('train/loss_mle', avg_loss_mle), cnt)
          tfwriter.add_summary(tf_summary_entry('train/loss_rl', avg_loss_rl), cnt)
          loss_mle_sum, loss_rl_sum = 0.0, 0.0

        if batch % (FLAGS.report * FLAGS.eval_multi) == 0:
          cnt = batch // (FLAGS.report * FLAGS.eval_multi) + FLAGS.cnt
          ksave_dir = save_model(model, save_dir, cnt)
          r, b_unk, b_cpy = evaluate(sess, dataloader, model, ksave_dir, 'valid', v)
          write_log(r, log_file)
          tfwriter.add_summary(tf_summary_entry('valid/BLEU', b_cpy), cnt)
          # tfwriter.add_summary(tf_summary_entry('train/loss', avg_loss), cnt)
          # tfwriter.add_summary(tf_summary_entry('valid/loss', l_v), cnt)

          for i in range(FLAGS.max_to_keep):
            if b_cpy >= best_bleu[i][1]:
              best_bleu[i] = ('ep%d'%cnt, b_cpy, b_unk)
              save_ckpt(sess, saver, save_dir, cnt)
              with open(rank_file, 'w+') as fout:
                json.dump(best_bleu, fout, sort_keys=True, indent=4)
              break
      batch += 1

def bleu_score(labels_file, predictions_path):
    bleu_script = '%s/onmt-tf-whm/third_party/multi-bleu.perl'%HOME
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [bleu_script, labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)

    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        tf.logging.warning(
            "{} script returned non-zero exit code: {}".format(bleu_script, msg))
      return None

def evaluate(*args):
  return evaluate_both(*args) if FLAGS.rouge else evaluate_bleu(*args)

def evaluate_bleu(sess, dataloader, model, ksave_dir, mode='valid', vocab=None):
  L.info('Begin evaluating (ROUGE=%s) in %s mode ...'%(FLAGS.rouge, mode))
  if mode == 'valid':
    texts_path = valid_path
    gold_path = gold_path_valid
    evalset = dataloader.dev_set
  else:
    texts_path = test_path
    gold_path = gold_path_test
    evalset = dataloader.test_set

  # for copy words from the infoboxes
  texts = open(texts_path, 'r').read().strip().split('\n')
  texts = [list(t.strip().split()) for t in texts]

  # with copy
  pred_list, pred_list_copy, gold_list = [], [], []
  pred_unk, pred_mask = [], []

  k = 0 # instance counter
  b = 0 # batch counter
  # loss = 0.0
  data_size = len(evalset[0])
  num_batches = int(data_size / FLAGS.batch_size) if data_size % FLAGS.batch_size == 0 \
                            else int(data_size / FLAGS.batch_size) + 1
  L.info('Evaluating by batches ...')
  for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
    predictions, atts = model.generate(x, sess)
    # loss += l
    atts = np.squeeze(atts)
    idx = 0 #idx within a batch
    b += 1
    for summary in np.array(predictions):
      with open(pred_path + str(k), 'w') as sw:
        summary = list(summary)
        if 2 in summary:
          summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
        real_sum, unk_sum, mask_sum = [], [], []
        for tk, tid in enumerate(summary):
          if tid == 3:
            sub = texts[k][np.argmax(atts[tk, :len(texts[k]), idx])]
            real_sum.append(sub)
            mask_sum.append("**" + str(sub) + "**")
          else:
            real_sum.append(vocab.id2word(tid))
            mask_sum.append(vocab.id2word(tid))
          unk_sum.append(vocab.id2word(tid))
        sw.write(" ".join([str(x) for x in real_sum]) + '\n')
        pred_list.append([str(x) for x in real_sum])
        pred_mask.append([str(x) for x in mask_sum])
        pred_unk.append([str(x) for x in unk_sum])
        k += 1
        idx += 1
    progress_bar(b, num_batches)
  write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
  write_word(pred_list, ksave_dir, mode + "_summary_copy.clean.txt")
  write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")
  # loss = loss/(data_size*1.0)

  bleu_unk = bleu_score(gold_path, ksave_dir + mode + "_summary_unk.txt")
  nocopy_result = "without copy BLEU: %.4f\n"%bleu_unk
  bleu_copy = bleu_score(gold_path, ksave_dir + mode + "_summary_copy.clean.txt")
  copy_result = "with copy BLEU: %.4f\n" %bleu_copy

  result 	= copy_result + nocopy_result

  return result, bleu_unk, bleu_copy

def evaluate_both(sess, dataloader, model, ksave_dir, mode='valid', vocab=None):
  L.info('Begin evaluating (ROUGE=%s) in %s mode ...'%(FLAGS.rouge, mode))
  if mode == 'valid':
    texts_path = valid_path
    gold_path = gold_path_valid
    evalset = dataloader.dev_set
  else:
    texts_path = test_path
    gold_path = gold_path_test
    evalset = dataloader.test_set

  # for copy words from the infoboxes
  texts = open(texts_path, 'r').read().strip().split('\n')
  texts = [list(t.strip().split()) for t in texts]

  # with copy
  pred_list, pred_list_copy, gold_list = [], [], []
  pred_unk, pred_mask = [], []

  k = 0
  # loss = 0.0
  data_size = len(evalset[0])
  L.info('data_size = {}'.format(data_size))
  L.info('Evaluating by batches ...')
  for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
    predictions, atts = model.generate(x, sess)
    # loss += l
    atts = np.squeeze(atts)
    idx = 0 #batch_idx
    for summary in np.array(predictions):
      with open(pred_path + str(k), 'w') as sw:
        summary = list(summary)
        if 2 in summary:
          summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
        real_sum, unk_sum, mask_sum = [], [], []
        for tk, tid in enumerate(summary):
          if tid == 3:
            sub = texts[k][np.argmax(atts[tk, :len(texts[k]), idx])]
            real_sum.append(sub)
            mask_sum.append("**" + str(sub) + "**")
          else:
            real_sum.append(vocab.id2word(tid))
            mask_sum.append(vocab.id2word(tid))
          unk_sum.append(vocab.id2word(tid))
        sw.write(" ".join([str(x) for x in real_sum]) + '\n')
        pred_list.append([str(x) for x in real_sum])
        pred_mask.append([str(x) for x in mask_sum])
        pred_unk.append([str(x) for x in unk_sum])
        k += 1
        idx += 1
  write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
  write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")
  # loss = loss/(data_size*1.0)

  for tk in range(k):
    with open(gold_path + str(tk), 'r') as g:
      gold_list.append([g.read().strip().split()])

  gold_set = [[gold_path + str(i)] for i in range(k)]
  pred_set = [pred_path + str(i) for i in range(k)]

  L.info('Calculating ROUGE with copy ...')
  recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)

  L.info('Calculating BLEU with copy ...')
  bleu_copy = corpus_bleu(gold_list, pred_list)
  copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
          (str(F_measure), str(recall), str(precision), str(bleu_copy))

  for tk in range(k):
    with open(pred_path + str(tk), 'w') as sw:
      sw.write(" ".join(pred_unk[tk]) + '\n')

  L.info('Calculating ROUGE without copy ...')
  recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
  L.info('Calculating BLEU without copy ...')
  bleu_unk = corpus_bleu(gold_list, pred_unk)
  nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
          (str(F_measure), str(recall), str(precision), str(bleu_unk))
  result 	= copy_result + nocopy_result

  return result, bleu_unk, bleu_copy

def evaluate_beam(sess, dataloader, model, ksave_dir, mode='valid', vocab=None, beam_size=1):
  L.info('Begin evaluating (ROUGE=%s) in %s mode ...'%(FLAGS.rouge, mode))
  if mode == 'valid':
    texts_path = valid_path
    gold_path = gold_path_valid
    evalset = dataloader.dev_set
  else:
    texts_path = test_path
    gold_path = gold_path_test
    evalset = dataloader.test_set

  # for copy words from the infoboxes
  texts = open(texts_path, 'r').read().strip().split('\n')
  texts = [list(t.strip().split()) for t in texts]

  counter = 0
  data_size = len(evalset[0])
  L.info('Evaluating by samples ...')

  all_summary = {k: v for k, v in zip(range(1,beam_size+1), [[] for _ in range(1,beam_size+1)])}

  for x in dataloader.batch_iter(evalset, 1, False):
    counter += 1

    beam_seqs_all, beam_probs_all, beam_predictions, cand_probs_all = model.generate_beam(x, sess)
    beam_predictions = beam_predictions.tolist()

    real_tokens = [[token_id for token_id in pred[1:] if token_id > 0] for pred in beam_predictions]

    for beam, summary in enumerate(real_tokens, start=1):
      summary = [str(vocab.id2word(token_id)) for token_id in summary]
      all_summary[beam].append(summary)

    progress_bar(counter, data_size)

  nocopy_result = []
  for beam in range(1,beam_size+1):
    write_word(all_summary[beam], ksave_dir, mode + "_summary_unk.beam.{}.txt".format(beam))
    L.info('Calculating BLEU for beam #{}'.format(beam))
    bleu_unk = bleu_score(gold_path, ksave_dir + mode + "_summary_unk.beam.{}.txt".format(beam))
    nocopy_result.append("without copy BLEU for beam #%d: %.4f\n"%(beam, bleu_unk))
  nocopy_result = ''.join(nocopy_result)
  return nocopy_result, bleu_unk, 0.0


def test(sess, dataloader, model, saver, beam_size=FLAGS.beam):
  print("beam_size={}".format(beam_size))
  v = Vocab()
  if beam_size > 0:
    result, _, _ = evaluate_beam(sess, dataloader, model, save_dir, 'test', vocab=v, beam_size=beam_size)
  else:
    result, _, _ = evaluate(sess, dataloader, model, save_dir, 'test', v)
  print(result)

def main():
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    copy_file(save_file_dir)
    dataloader = DataLoader(FLAGS.dir, FLAGS.dir_ori, FLAGS.limits)
    model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size,
            emb_size=FLAGS.emb_size, field_size=FLAGS.field_size,
            pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
            source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
            target_vocab=FLAGS.target_vocab,
            scope_name="seq2seq", name="seq2seq",
            field_concat=FLAGS.field, position_concat=FLAGS.position,
            fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention,
            decoder_add_pos=FLAGS.decoder_pos,
            encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate,
            rl=FLAGS.rl, loss_alpha=FLAGS.alpha,
            beam_size=FLAGS.beam)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1000)

    if FLAGS.load:
      L.info('Loading Model from %s ...'%load_dir)
      model.load(load_dir)
      L.info('Model loaded from %s'%load_dir)
    if FLAGS.mode == 'train':
      train(sess, dataloader, model, saver)
    else:
      test(sess, dataloader, model, saver)


if __name__=='__main__':
  program = os.path.basename(sys.argv[0])
  print(program)
  L = logging.getLogger(program)
  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  L.info("Running %s" % ' '.join(sys.argv))
  write_log(' '.join(sys.argv), log_file)
  main()
