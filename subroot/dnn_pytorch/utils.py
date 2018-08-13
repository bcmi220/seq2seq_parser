import os
import re
import io
import itertools
import codecs
import time
import numpy as np
import collections
from torch.autograd import Variable
from dnn_utils import LongTensor,FloatTensor

try:
    import _pickle as cPickle
except ImportError:
    import cPickle

models_path = "./models"


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word(inputs, seq_len):
    # get the max sequence length in the batch
    max_len = seq_len[0]

    padding = np.zeros_like([inputs[0][0]]).tolist()

    padded_inputs = []
    for item in inputs:
        padded_inputs.append(item + padding * (max_len - len(item)))

    return padded_inputs


def pad_chars(inputs):
    chained_chars = list(itertools.chain.from_iterable(inputs))

    char_index_mapping, chars = zip(
        *[item for item in sorted(
            enumerate(chained_chars), key=lambda x: len(x[1]), reverse=True
        )]
    )
    char_index_mapping = {v: i for i, v in enumerate(char_index_mapping)}

    char_len = [len(c) for c in chars]

    chars = pad_word(chars, char_len)

    # pad chars to length of 25 if max char len less than 25
    # char CNN layer requires at least 25 chars
    if len(chars[0]) < 25:
        chars = [c + [0]*(25-len(c)) for c in chars]

    return chars, char_index_mapping, char_len


def create_input(data, parameters, add_label=True):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    # sort data by sequence length
    seq_index_mapping, data = zip(*[item for item in sorted(enumerate(data), key=lambda x: len(x[1]['words']), reverse=True)])
    seq_index_mapping = {v: i for i, v in enumerate(seq_index_mapping)}

    inputs = collections.defaultdict(list)
    seq_len = []

    for d in data:
        words = d['words']

        seq_len.append(len(words))

        chars = d['chars']

        if parameters['word_dim']:
            inputs['words'].append(words)
        if parameters['char_dim']:
            inputs['chars'].append(chars)
        if parameters['cap_dim']:
            caps = d['caps']
            inputs['caps'].append(caps)

        # boliang: add expectation features into input
        if d['feats']:
            inputs['feats'].append(d['feats'])

        if add_label:
            tags = d['tags']
            inputs['tags'].append(tags)

    char_index_mapping = []
    char_len = []
    for k, v in inputs.items():
        if k == 'chars':
            padded_chars, char_index_mapping, char_len = pad_chars(v)
            inputs[k] = padded_chars
        else:
            inputs[k] = pad_word(v, seq_len)

    # convert inputs and labels to Variable
    for k, v in inputs.items():
        inputs[k] = Variable(LongTensor(v))

    return inputs, seq_index_mapping, char_index_mapping, seq_len, char_len

def count_sentence_predicate(sentence):
    count = 0
    for item in sentence:
        if item[-2] == 'Y':
            count += 1
    return count

def evaluate(phase, preds, dataset, id_to_tag, eval_out_dir=None):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)

    tp = 0
    fp = 0
    fn = 0
    correct = 0
    total = 0
    
    output = []
    for d, p in zip(dataset, preds):

        assert len(d['words']) == len(p)
        str_words = d['str_words']
        p_tags = [id_to_tag[y_pred] for y_pred in p]
        r_tags = [id_to_tag[y_real] for y_real in d['tags']]

        block = []
        for i in range(len(p_tags)):
            if r_tags[i]!='0' and p_tags[i] == r_tags[i]:
                tp += 1
            if r_tags[i]!='0' and p_tags[i] != r_tags[i]:
                fp += 1
            if r_tags[i]=='0' and p_tags[i] != r_tags[i]:
                fn += 1
            if p_tags[i] == r_tags[i]:
                correct += 1
            total += 1
            block.append([r_tags[i],p_tags[i]])
        output.append(block)
        
        p = tp / (tp + fp + 1e-13)

        r = tp / (tp + fn + 1e-13)

        f1 = 2 * p * r / ( p + r + 1e-13)
    
        acc = correct / total

    # Global accuracy
    print("Acc:%.5f%% P:%.5f R:%.5f F1:%.5f" % (
        acc * 100, p * 100, r * 100, f1 * 100
    ))

    if eval_out_dir is not None:
        output_file = os.path.join(eval_out_dir,'{}_predicate_{:.2f}.pred'.format(phase,p*100))
        with open(output_file, 'w') as fout:
            for block in output:
                for line in block:
                    fout.write('\t'.join(line))
                    fout.write('\n')
                fout.write('\n')

    return f1, acc


########################################################################################################################
# temporal script below
#
def load_exp_feats(fp):
    bio_feats_fp = fp
    res = []
    for sent in io.open(bio_feats_fp, 'r', -1, 'utf-8').read().split('\n\n'):
        sent_feats = []
        for line in sent.splitlines():
            feats = line.split('\t')[1:]
            sent_feats.append(feats)
        res.append(sent_feats)

    return res


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()




