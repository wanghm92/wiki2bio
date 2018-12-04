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

def get_reward_bleu(gold_summary_tks, real_sum_list):
    """ BLEU score as reward """
    return np.array([sentence_bleu([r], s, smoothing_function=chencherry)
                     for r, s in zip(gold_summary_tks, real_sum_list)],
                    dtype=np.float32)
