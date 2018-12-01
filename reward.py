
import spacy
import numpy as np
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'tokenizer'])
# tokenization_to_bracket = {'-lrb-': '(', '-rrb-': ')', '-lcb-': '{', '-rcb-': '}', '-lsb-': '[', '-rsb-': ']', }

# def convert_back_to_brackets(input_str):
#     for k, v in tokenization_to_bracket.items():
#         input_str = input_str.replace(k, v)
#     return input_str

POS=1
NEU=0
NEG=-1

def get_reward(self, train_box_batch, pred_sentences, real_ids_list, result, mask, unkmask_list, max_gold_len):

    # predicted in + UNK --> +1
    #TODO:
      # get_reward()
      # pad
      # pass to placeholder
    reward_matrix = []

    assert real_ids_list.shape == unkmask_list.shape
    assert result.shape == mask.shape

    sequence_lengths = [len(x.split()) for x in pred_sentences]
    print(sequence_lengths)

    def _check_in_box(label, token, lemma, train_box):
        box_dict = dict.fromkeys(train_box)
        box_lemmas = [nlp(x)[0].lemma_ for x in train_box]
        box_lemma_dict = dict.fromkeys(box_lemmas)

        if label != 1:
            return NEU
        else:
            return POS if box_dict.has_key(token) or box_lemma_dict.has_key(lemma) else NEG


    for batch, (labels, binary_switch) in enumerate(zip(result, mask)):
        train_box = train_box_batch[batch]
        binary_labels = [l for l, m in zip(labels, binary_switch) if m == 1]
        sampled_tokens = pred_sentences[batch]
        sampled_lemmas = [nlp(x)[0].lemma_ for x in sampled_tokens.split()]

        reward = [_check_in_box(l, tk, lm, train_box)
                  for l, tk, lm in zip(binary_labels, sampled_lemmas.split(), sampled_tokens.split())]
        reward_matrix.append(reward)

    reward_matrix = np.array([ids + [0] * (max_gold_len - len(ids)) for ids in reward_matrix], dtype=np.float32)

    return reward_matrix