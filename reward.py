from __future__ import unicode_literals
import spacy, sys
import scipy.signal as signal
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
chencherry = SmoothingFunction().method1

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'tokenizer'])
tokenization_to_bracket = {'-lrb-': '(', '-rrb-': ')', '-lcb-': '{', '-rcb-': '}', '-lsb-': '[', '-rsb-': ']', }

def _convert_back_to_brackets(input_str):
    for k, v in tokenization_to_bracket.items():
        input_str = input_str.replace(k, v)
    return input_str

def get_reward(train_box_batch, gold_summary_tks, real_sum_list, max_summary_len, bert_server, neg=False, discount=0.0):

    # predicted in + UNK --> +1

    POS = 1.5
    NEU = 1
    NEG = 0

    reward_matrix = []

    pred_sentences = [' '.join([_convert_back_to_brackets(y.decode('utf-8')) for y in x]) for x in real_sum_list]
    # print(pred_sentences)
    result, mask = bert_server.encode(pred_sentences) # [batch, 64]
    assert result.shape == mask.shape

    def _check_in_box(label, token, lemma, box_dict, box_lemma_dict):
        if label != 1:
            return NEU
        else:
            # print([token,lemma])
            has_token = box_dict.has_key(token)
            has_lemma = box_lemma_dict.has_key(lemma)
            # print(has_token)
            # print(has_lemma)
            return POS if has_token or has_lemma else NEG

    def _discounted_cumulative_reward_v1(rewards, discount=0.9):
        """unused"""
        rewards = np.array(rewards, dtype=np.float32)
        future_cumulative_reward = 0
        cumulative_rewards = np.empty_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
            future_cumulative_reward = cumulative_rewards[i]
        return cumulative_rewards

    def _discounted_cumulative_reward_v2(rewards, discount=0.9):

        """
        C[i] = R[i] + discount * C[i+1]
        signal.lfilter(b, a, x, axis=-1, zi=None)
        a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                              - a[1]*y[n-1] - ... - a[N]*y[n-N]
        """
        r = rewards[::-1]
        a = [1, -discount]
        b = [1]
        y = signal.lfilter(b, a, x=r)
        return y[::-1]

    for batch, (labels, binary_switch) in enumerate(zip(result, mask)):
        train_box = [x.decode('utf-8') for x in train_box_batch[batch]]
        box_dict = dict.fromkeys(train_box)
        box_lemmas = [nlp(x)[0].lemma_ for x in train_box]
        box_lemma_dict = dict.fromkeys(box_lemmas)
        # print(box_dict)
        # print(box_lemma_dict)

        binary_labels = [l for l, m in zip(labels, binary_switch) if m == 1]
        sampled_tokens = [x.decode('utf-8') for x in real_sum_list[batch]]
        sampled_lemmas = [nlp(x)[0].lemma_ for x in sampled_tokens]
        # print(labels)
        # print(binary_switch)

        rewards = [_check_in_box(l, tk, lm, box_dict, box_lemma_dict)
                  for l, tk, lm in zip(binary_labels, sampled_tokens, sampled_lemmas)]
        rewards.append(NEU) # for eos
        # print(train_box)
        # print(box_lemmas)
        # print(sampled_tokens)
        # print(sampled_lemmas)
        # print(binary_labels)
        # print(rewards)
        if discount > 0:
            discounted_rewards = _discounted_cumulative_reward_v2(rewards)
            # print(len(discounted_rewards))
            reward_matrix.append(discounted_rewards)
        else:
            reward_matrix.append(rewards)

    reward_matrix = [np.pad(ids, (0, max_summary_len + 1 - len(ids)), 'constant') for ids in reward_matrix]

    reward_matrix = np.array(reward_matrix, dtype=np.float32)

    return reward_matrix


def get_reward_coverage(gold_summary_tks, coverage_labels, real_sum_list, bert_server):
    """ content word coverage F1 score as reward """
    pred_token_lists = [[_convert_back_to_brackets(y.decode('utf-8')) for y in x] for x in real_sum_list]
    pred_sentences = [' '.join(x) for x in pred_token_lists]
    # print(pred_sentences)
    result, mask = bert_server.encode(pred_sentences) # [batch, 64]
    assert result.shape == mask.shape

    rewards = []
    for batch, (gold, coverage, pred_tokens, labels, binary_switch) \
            in enumerate(zip(gold_summary_tks, coverage_labels, pred_token_lists, result, mask)):

        binary_labels = [l for l, m in zip(labels, binary_switch) if m == 1]
        try:
            assert len(pred_tokens) == len(binary_labels)
        except AssertionError:
            rewards.append(0.0)
            continue

        gold_set = set([x for x, y in zip(gold, coverage) if y == 1])
        pred_set = set([x for x, y in zip(pred_tokens, binary_labels) if y == 1])

        intersection = gold_set.intersection(pred_set)
        precision = float(len(intersection)) / float(len(pred_set)) if len(pred_set) > 0 else 0.0
        recall = float(len(intersection)) / float(len(gold_set)) if len(gold_set) > 0 else 0.0
        denominator = precision + recall
        f1 = 2*precision*recall/denominator if denominator > 0.0 else 0.0
        rewards.append(f1)

    return np.array(rewards, dtype=np.float32)


def get_reward_coverage_v2(train_box_batch, gold_summary_tks, coverage_labels, real_sum_list, max_summary_len, bert_server):

    POS = 1.0
    NEU = 0.0
    NEG = 0.0

    def _check_in_box(label, token, lemma, box_dict, box_lemma_dict):
        if label != 1:
            return NEU
        else:
            has_token = box_dict.has_key(token)
            has_lemma = box_lemma_dict.has_key(lemma)
            return POS if has_token or has_lemma else NEG

    pred_token_lists = [[_convert_back_to_brackets(y.decode('utf-8')) for y in x] for x in real_sum_list]
    pred_sentences = [' '.join(x) for x in pred_token_lists]
    # print("pred_sentences")
    # for p, b in zip(pred_sentences, train_box_batch):
        # print(b)
        # print(p)
    result, mask = bert_server.encode(pred_sentences) # [batch, 64]
    assert result.shape == mask.shape

    reward_list = []
    reward_matrix = []

    for batch, (gold, coverage, pred_tokens, labels, binary_switch) \
            in enumerate(zip(gold_summary_tks, coverage_labels, pred_token_lists, result, mask)):

        train_box = [x.decode('utf-8') for x in train_box_batch[batch]]
        # print('train_box')
        # print(train_box)
        box_dict = dict.fromkeys(train_box)
        box_lemmas = [nlp(x)[0].lemma_ for x in train_box]
        box_lemma_dict = dict.fromkeys(box_lemmas)
        # print(box_dict)
        # print(box_lemma_dict)

        binary_labels = [l for l, m in zip(labels, binary_switch) if m == 1]
        # print('binary_labels')
        # print(binary_labels)
        try:
            assert len(pred_tokens) == len(binary_labels)
        except AssertionError:
            reward_list.append(0.0)
            reward_matrix.append([NEG])

            # print('pred_tokens')
            # print(len(pred_tokens))
            # print(pred_tokens)
            # print('binary_labels')
            # print(len(binary_labels))
            # print(binary_labels)
            continue

        gold_set = set([x for x, y in zip(gold, coverage) if y == 1])
        pred_set = set([x for x, y in zip(pred_tokens, binary_labels) if y == 1])
        # print('gold_set')
        # print(gold_set)
        # print('pred_set')
        # print(pred_set)

        intersection = gold_set.intersection(pred_set)
        precision = float(len(intersection)) / float(len(pred_set)) if len(pred_set) > 0 else 0.0
        recall = float(len(intersection)) / float(len(gold_set)) if len(gold_set) > 0 else 0.0
        denominator = precision + recall
        f1 = 2.0*precision*recall/denominator if denominator > 0.0 else 0.0
        reward_list.append(f1)

        sampled_tokens = [x.decode('utf-8') for x in real_sum_list[batch]]
        sampled_lemmas = [nlp(x)[0].lemma_ for x in sampled_tokens]
        # print('sampled_tokens')
        # print(sampled_tokens)
        # print('sampled_lemmas')
        # print(sampled_lemmas)
        rewards = [_check_in_box(l, tk, lm, box_dict, box_lemma_dict)
                  for l, tk, lm in zip(binary_labels, sampled_tokens, sampled_lemmas)]
        rewards.append(NEG) # for eos

        # print('rewards')
        # print(rewards)
        # print('-'*50+'\n')
        reward_matrix.append(rewards)

    # reward_matrix = [np.pad(ids, (0, max_summary_len + 1 - len(ids)), 'constant') for ids in reward_matrix]

    # reward_matrix = np.array(reward_matrix, dtype=np.float32)
    # print('reward_matrix')
    # print(reward_matrix.shape)
    # print(reward_matrix)
    reward_list = np.array(reward_list, dtype=np.float32)
    # reward_list_exp = np.expand_dims(reward_list, axis=-1)
    # print('reward_list')
    # print(reward_list.shape)
    # print(reward_list)

    # final_rewards = reward_matrix*reward_list_exp
    return reward_list, reward_matrix


def get_reward_bleu(gold_summary_tks, real_sum_list):
    """ BLEU score as reward """
    return np.array([sentence_bleu([r], s, smoothing_function=chencherry)
                     for r, s in zip(gold_summary_tks, real_sum_list)],
                    dtype=np.float32)