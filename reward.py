from __future__ import unicode_literals
import spacy, sys
import numpy as np
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'tokenizer'])
tokenization_to_bracket = {'-lrb-': '(', '-rrb-': ')', '-lcb-': '{', '-rcb-': '}', '-lsb-': '[', '-rsb-': ']', }

def convert_back_to_brackets(input_str):
    for k, v in tokenization_to_bracket.items():
        input_str = input_str.replace(k, v)
    return input_str

POS=1
NEU=0
NEG=0

def get_reward(train_box_batch, real_sum_list, max_summary_len, bert_server):

    # predicted in + UNK --> +1

    reward_matrix = []

    pred_sentences = [' '.join([convert_back_to_brackets(y.decode('utf-8')) for y in x]) for x in real_sum_list]
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

        reward = [_check_in_box(l, tk, lm, box_dict, box_lemma_dict)
                  for l, tk, lm in zip(binary_labels, sampled_tokens, sampled_lemmas)]
        # print(train_box)
        # print(box_lemmas)
        # print(sampled_tokens)
        # print(sampled_lemmas)
        # print(binary_labels)
        # print(reward)

        reward_matrix.append(reward)

    reward_matrix = np.array([ids + [0] * (max_summary_len+1 - len(ids)) for ids in reward_matrix], dtype=np.float32)

    return reward_matrix