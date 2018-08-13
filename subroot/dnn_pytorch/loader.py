import os
import re
import codecs

try:
    import lzma
except ImportError:
    pass

from utils import create_dico, create_mapping, zero_digits


def load_sentences(path, lower=False, zeros=False):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(path, 'r'):
        line = line.strip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            word[0] = zero_digits(word[0]) if zeros else word[0]
            # assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


# def update_tag_scheme(sentences, tag_scheme):
#     """
#     Check and update sentences tagging scheme to IOB2.
#     Only IOB1 and IOB2 schemes are accepted.
#     """
#     for i, s in enumerate(sentences):
#         tags = [w[-1] for w in s]
#         if tag_scheme == 'classification':
#             for word, new_tag in zip(s, tags):
#                 word[-1] = new_tag
#         else:
#             # Check that tags are given in the IOB format
#             if not iob2(tags):
#                 s_str = '\n'.join(' '.join(w) for w in s)
#                 raise Exception('Sentences should be given in IOB format! ' +
#                                 'Please check sentence %i:\n%s' % (i, s_str))
#             if tag_scheme == 'iob':
#                 # If format was IOB1, we convert to IOB2
#                 for word, new_tag in zip(s, tags):
#                     word[-1] = new_tag
#             elif tag_scheme == 'iobes':
#                 new_tags = iob_iobes(tags)
#                 for word, new_tag in zip(s, new_tags):
#                     word[-1] = new_tag
#             else:
#                 raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def feats_mapping(sentences, feat_column):
    """
    Boliang
    Create a list of dictionary and a list of mappings of features, sorted by frequency.
    """
    assert all(
        [[len(word) == sentences[0][0] for word in s] for s in sentences]
    ), 'features length are not consistent for all instances.'

    dico_list = []
    feat_to_id_list = []
    id_to_feat_list = []

    feature_len = len(sentences[0][0]) - feat_column - 1

    for i in range(feature_len):
        feats = [[word[i+feat_column] for word in s] for s in sentences]
        dico = create_dico(feats)
        dico['<UNK>'] = 10000000
        feat_to_id, id_to_feat = create_mapping(dico)

        dico_list.append(dico)
        feat_to_id_list.append(feat_to_id)
        id_to_feat_list.append(id_to_feat)

    return dico_list, feat_to_id_list, id_to_feat_list


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(sentence, feat_column,
                     word_to_id, char_to_id, tag_to_id, feat_to_id_list,
                     lower=False, is_train=True):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    max_sent_len = 300
    max_word_len = 50
    if is_train:
        sentence = sentence[:max_sent_len]
    str_words = [w[0] for w in sentence]
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    # set max word len for char embedding layer to prevent memory problem
    if is_train:
        chars = [[char_to_id[c] if c in char_to_id else 0 for c in w[:max_word_len]]
                 for w in str_words]
    else:
        chars = [
            [char_to_id[c] if c in char_to_id else 0 for c in w]
            for w in str_words
            ]
    caps = [cap_feature(w) for w in str_words]
    tags = []
    if is_train:
        for w in sentence:
            if w[-1] in tag_to_id:
                tags.append(tag_to_id[w[-1]])
            else:
                tags.append(0)

    # features
    sent_feats = []
    for token in sentence:
        s_feats = []
        for j, feat in enumerate(token[feat_column:-1]):
            if feat not in feat_to_id_list[j]:
                s_feats.append(feat_to_id_list[j]['<UNK>'])
            else:
                s_feats.append(feat_to_id_list[j][feat])
        if s_feats:
            sent_feats.append(s_feats)

    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'tags': tags,
        'feats': sent_feats
    }


def prepare_dataset(sentences, feat_column,
                    word_to_id, char_to_id, tag_to_id, feat_to_id_list,
                    lower=False, is_train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for i, s in enumerate(sentences):
        data.append(
            prepare_sentence(s, feat_column,
                             word_to_id, char_to_id, tag_to_id, feat_to_id_list,
                             lower, is_train=is_train)
        )

    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # debug embeddings
    # index = 0
    # for line in codecs.open(ext_emb_path, 'r', 'utf-8'):
    #     index += 1
    #     print(index)

    # Load pretrained embeddings from file
    pretrained = []
    if len(ext_emb_path) > 0:
        for line in load_embedding(ext_emb_path):
            if not line.strip():
                continue
            if type(line) == bytes:
                try:
                    pretrained.append(str(line, 'utf-8').rstrip().split()[0].strip())
                except UnicodeDecodeError:
                    continue
            else:
                pretrained.append(line.rstrip().split()[0].strip())

    pretrained = set(pretrained)

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def load_embedding(pre_emb):
    if os.path.basename(pre_emb).endswith('.xz'):
        return lzma.open(pre_emb)
    else:
        return codecs.open(pre_emb, 'r', 'utf-8', errors='replace')
