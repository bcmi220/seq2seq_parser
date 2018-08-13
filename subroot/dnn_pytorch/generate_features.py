import os
import sys
import tempfile
import subprocess
import itertools
import collections
import argparse
import shutil
from loader import load_sentences


#
# generate external features
#
def generate_features(sentences, parameters):
    feats = []
    stem = []
    if parameters['upenn_stem']:
        stem, affix = stem_feature(sentences, parameters)
        feats.append(affix)
    if parameters['pos_model']:
        feats.append(postag_feature(sentences, parameters))
    if parameters['cluster']:
        feats.append(clustering_feature(sentences, parameters))
    if parameters['ying_stem']:
        stem = ying_stem_feature(sentences, parameters)
    if parameters['gaz']:
        feats.append(gaz_features(sentences, parameters))

    feats = merge_features(feats)

    return feats, stem


#
# clustering features
#
def clustering_feature(sentences, parameters):
    def f(x): return x.lower() if parameters['lower'] else x
    cluster_path_file = parameters['cluster']
    #
    # parse cluster path
    #
    print('=> parsing cluster path...')
    cluster_path = dict()
    num_lines_loaded = 0
    path_len = 0
    for line in open(cluster_path_file).read().splitlines():
        if not line:
            continue
        cluster, word, frequency = line.split('\t')

        cluster_path[f(word)] = cluster

        if len(cluster) > path_len:
            path_len = len(cluster)

        num_lines_loaded += 1
        if num_lines_loaded % 50 == 0:
            sys.stdout.write("%d word paths loaded.\r" % num_lines_loaded )
            sys.stdout.flush()

    # padding short paths
    for k, v in cluster_path.items():
        v += '0' * (path_len - len(v))
        cluster_path[k] = v

    print('=> %d words with cluster path are loaded in total.' %
          len(cluster_path))
    print('max path len is %d' % path_len)

    #
    # sentence input
    #
    sentence_clusters = []
    unk_cluster_path = ['0' * int(0.4*path_len),
                        '0' * int(0.6*path_len),
                        '0' * int(0.8*path_len),
                        '0' * path_len]
    cluster_coverage = collections.defaultdict(int)
    token_count = collections.defaultdict(int)
    token_with_labels = collections.defaultdict(int)
    token_with_labels_covered = collections.defaultdict(int)
    for i, s in enumerate(sentences):
        if i % 100 == 0:
            sys.stdout.write('%d sentences processed.\r' % i)
            sys.stdout.flush()

        s_cluster_path = []
        for token in s:
            text = f(token[0])
            label = token[-1]

            token_count[text] += 1
            if label != 'O':
                token_with_labels[text] += 1
            if text in cluster_path:
                c_path = cluster_path[text]
                cluster_coverage[text] += 1
                if label != 'O':
                    token_with_labels_covered[text] += 1
            else:
                c_path = '0' * path_len

            s_cluster_path.append([c_path[:int(0.4*path_len)],
                                   c_path[:int(0.6*path_len)],
                                   c_path[:int(0.8*path_len)],
                                   c_path])

        s = []
        # add prev and next word cluster path
        for j in range(len(s_cluster_path)):
            if j == 0:
                prev_cp = unk_cluster_path
            else:
                prev_cp = s_cluster_path[j-1]
            if j == len(s_cluster_path)-1:
                next_cp = unk_cluster_path
            else:
                next_cp = s_cluster_path[j+1]
            s.append(prev_cp + s_cluster_path[j] + next_cp)

        sentence_clusters.append(s)

    print('%d / %d (%.2f) tokens have clusters.' % (sum(cluster_coverage.values()),
                                                    sum(token_count.values()),
                                                    sum(cluster_coverage.values()) / sum(token_count.values())))
    print('%d / %d (%.2f) unique tokens have clusters.' % (len(cluster_coverage),
                                                           len(token_count),
                                                           len(cluster_coverage) / len(token_count)))

    print('%d / %d (%.2f) labeled tokens have clusters.' % (
        sum(token_with_labels_covered.values()),
        sum(token_with_labels.values()),
        sum(token_with_labels_covered.values()) / (sum(token_with_labels.values())+1)
    ))
    print('%d / %d (%.2f) labeled unique tokens have clusters.' % (
        len(token_with_labels_covered),
        len(token_with_labels),
        len(token_with_labels_covered) / (len(token_with_labels)+1)
    ))

    return sentence_clusters


#
# postagging features
#
def postag_feature(sentences, parameters):
    postag_model = parameters['pos_model']
    print('=> running postagger...')
    tagger_script = '/nas/data/m1/zhangb8/ml/theano/wrapper/dnn_tagger.py'
    print('tagger script: %s' % tagger_script)
    tmp_output = tempfile.mktemp()

    tmp_input = tempfile.mktemp()
    with open(tmp_input, 'w') as f:
        res = []
        for s in sentences:
            sent = []
            for token in s:
                sent.append(token[0])
            res.append('\n'.join(sent))

        f.write('\n\n'.join(res))

    cmd = ['python', tagger_script, tmp_input, tmp_output, postag_model, '-b',
           '--core_num', '5']
    print('command: %s' % ' '.join(cmd))
    subprocess.call(cmd)

    # parse pos-tagger output
    res = []
    for line in open(tmp_output).read().split('\n\n'):
        line = line.strip()
        if not line:
            continue
        sent = []
        for token in line.splitlines():
            items = token.split()
            if len(items) == 2 and ':' not in items[1]:
                items = [items[0], 'no_offset', items[1]]
            sent.append([items[-1]])
        res.append(sent)

    return res


#
# upenn stem feature
#
def stem_feature(sentences, parameters):
    def f(x): return x.lower() if parameters['lower'] else x
    upenn_morph_file = parameters['upenn_stem']
    #
    # parse upenn morph str
    #
    print('=> loading upenn morphology analysis...')
    upenn_morph = dict()
    num_pair_loaded = 0
    for line in open(upenn_morph_file).read().splitlines():
        if not line:
            continue
        word, morph, suffix = line.split('\t')

        # select the longest portion as stem
        affix = morph.split(' ')
        stem = sorted(affix, key=lambda x: len(x), reverse=True)[0]
        stem_index = affix.index(stem)
        prefix = affix[:stem_index]
        suffix = affix[stem_index+1:]
        if word != stem and stem.strip():
            upenn_morph[f(word)] = [f(stem),
                                    f(''.join(prefix)) if prefix else 'no_prefix',
                                    f(''.join(suffix)) if suffix else 'no_suffix']
            num_pair_loaded += 1
            if num_pair_loaded % 100 == 0:
                sys.stdout.write("%d morph-stem pairs loaded.\r" % num_pair_loaded)
                sys.stdout.flush()
    print('=> %d morph-stem pairs loaded in total.' % len(upenn_morph))

    #
    # sentence input
    #
    stem_res = []
    affix_res = []
    num_stem = 0
    num_stem_label = 0
    num_label = 0
    num_token = 0
    for s in sentences:
        stem_sent = []
        affix_sent = []
        for i, t in enumerate(s):
            if num_token % 100 == 0:
                sys.stdout.write('%d tokens processed.\r' % num_token)
                sys.stdout.flush()

            num_token += 1

            text = f(t[0])
            label = t[-1]
            if label != 'O':
                num_label += 1

            if text in upenn_morph:
                stem, prefix, suffix = upenn_morph[text]
                num_stem += 1
                if label != 'O':
                    num_stem_label += 1
            else:
                stem, prefix, suffix = text, 'no_prefix', 'no_suffix'

            stem_sent.append(stem)
            affix_sent.append([prefix, suffix])

        stem_res.append(stem_sent)
        affix_res.append(affix_sent)

    print('=> %.2f%% (%d/%d) tokens stemmed.' % (
        num_stem / num_token * 100, num_stem, num_token
    ))
    print('=> %.2f%% (%d/%d) tokens with label are stemmed.' %
          (num_stem_label/num_label*100, num_stem_label, num_label))

    return stem_res, affix_res


#
# Ying stem feature
#
def ying_stem_feature(sentences, parameters):
    def f(x): return x.lower() if parameters['lower'] else x

    ying_stem = parameters['ying_stem']
    # parse ying stem
    print('=> parsing ying stem...')
    ying_stems = dict()
    orm_freq_shreshold = 5
    stem_freq_shreshold = 5
    num_tokens = 0
    for line in open(ying_stem):
        line = line.strip()
        if not line:
            continue

        num_tokens += 1

        orm, stem, orm_freq, stem_freq = line.split('\t')

        if orm == stem:
            continue
        # if int(orm_freq) < orm_freq_shreshold and \
        #                 int(stem_freq) > stem_freq_shreshold:
        #     stems[orm] = stem
        ying_stems[f(orm)] = f(stem)
    print('%d tokens parsed.' % num_tokens)
    print('%d tokens added to stem dict.' % len(ying_stems))

    # stem sentences
    stem_res = []
    num_stem = 0
    num_stem_label = 0
    num_label = 0
    num_tokens = 0
    for s in sentences:
        stem_sent = []
        for i, t in enumerate(s):
            if num_tokens % 100 == 0:
                sys.stdout.write('%d tokens processed.\r' % num_tokens)
                sys.stdout.flush()
            num_tokens += 1

            text = f(t[0])
            label = t[-1]
            if label != 'O':
                num_label += 1

            if text in ying_stems:
                stem = ying_stems[text]
                num_stem += 1
                if label != 'O':
                    num_stem_label += 1
            else:
                stem = text

            stem_sent.append(stem)

        stem_res.append(stem_sent)

    print('=> %.2f%% (%d/%d) tokens stemmed.' % (
        num_stem / num_tokens * 100, num_stem, num_tokens
    ))
    print('=> %.2f%% (%d/%d) tokens with label are stemmed.' %
          (num_stem_label / num_label * 100, num_stem_label, num_label))

    return stem_res


#
# gaz features
#
def gaz_features(sentence, parameters):
    def f(x): return x.lower() if parameters['lower'] else x

    print('=> generating gaz features...')
    print('loading gaz...')
    # parsing gaz
    gaz = []
    max_name_len = []
    for gaz_file in parameters['gaz']:
        assert os.path.exists(gaz_file), 'gaz file not exists: %s' % gaz_file
        g = set()
        max_len = 0
        for line in open(gaz_file):
            line = line.strip()
            if not line:
                continue
            name = tuple(f(line).split())
            g.add(name)

            if len(name) > max_len:
                max_len = min(7, len(name))  # set max name len to 7

        max_name_len.append(max_len)
        gaz.append(g)

    # generate gaz features for each token
    gaz_feats = []
    num_token_matched = collections.defaultdict(int)
    num_labeled_token_match = collections.defaultdict(int)
    for s_index, s in enumerate(sentence):
        s_gaz_feats = []
        for i, token in enumerate(s):
            token_gaz_feats = []
            token_text = f(token[0])
            token_label = token[-1]
            for j, g in enumerate(gaz):
                context_window = int((max_name_len[j] - 1) / 2)
                token_index = i

                if token_index - context_window < 0:
                    context_tokens = [
                        f(item[0]) for item in s[:token_index + context_window + 1]
                        ]
                else:
                    context_tokens = [
                        f(item[0]) for item in s[token_index - context_window:
                        token_index + context_window + 1]
                        ]

                valid_context_tokens = []

                # if len(token.text) >= 5:
                #     min_context_len = 1
                # else:
                #     min_context_len = 2

                # min_context_len = 2

                for i in range(1, len(context_tokens) + 1):
                    combinations = list(
                        itertools.combinations(context_tokens, i)
                    )
                    for c in combinations:
                        # make sure the combination is a sublist and contains
                        # the token.
                        if token_text not in c:
                            continue
                        if any(list(c) == context_tokens[j:j+len(c)]
                               for j in range(len(context_tokens))):
                            valid_context_tokens.append(c)

                # sort context by length
                sorted_contexts = sorted(valid_context_tokens, key=len, reverse=True)

                is_in_g = False
                for c in sorted_contexts:
                    if c in g:
                        token_gaz_feats.append('1')
                        is_in_g = True
                        num_token_matched[j] += 1
                        if token_label != 'O':
                            num_labeled_token_match[j] += 1
                        break
                if not is_in_g:
                    token_gaz_feats.append('0')
            s_gaz_feats.append(token_gaz_feats)

        gaz_feats.append(s_gaz_feats)

        sys.stdout.write('%d sentences are processed...\r' % s_index)
        sys.stdout.flush()

    num_token = sum([len(s) for s in sentence])
    print('%d sentences and %d tokens are processed in total.' % (len(sentence), num_token))
    for i in range(len(gaz)):
        print('gaz %d: %d tokens match.' % (i, num_token_matched[i]))
        print('gaz %d: %d labeled tokens match.' % (i, num_labeled_token_match[i]))

    return gaz_feats


def merge_features(sent_features):
    res = []
    for s_feat in zip(*sent_features):
        merged_sent_f = []
        for t_feat in zip(*s_feat):
            merged_t_feat = list(itertools.chain.from_iterable(t_feat))
            merged_sent_f.append(merged_t_feat)
        res.append(merged_sent_f)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bio_input',
                        help='bio file that needs to generate features')
    parser.add_argument('bio_output',
                        help='path of the result')
    parser.add_argument('--feat_column', type=int, default=1,
                        help='the number of the column where the features '
                             'start. default is 1, the 2nd column.')
    parser.add_argument(
        "--lower", default='0',
        type=int, help="Lowercase words"
    )
    #
    # external features
    #
    parser.add_argument(
        "--upenn_stem", default="",
        help="path of upenn morphology analysis result."
    )
    parser.add_argument(
        "--pos_model", default="",
        help="path of pos tagger model."
    )
    parser.add_argument(
        "--cluster", default="",
        help="path of brown cluster paths."
    )
    parser.add_argument(
        "--ying_stem", default="",
        help="path of Ying's stemming result."
    )
    parser.add_argument(
        "--gaz", default="", nargs="+",
        help="gazetteers paths."
    )
    args = parser.parse_args()

    # external features
    parameters = dict()
    parameters['lower'] = args.lower == 1
    parameters['upenn_stem'] = args.upenn_stem
    parameters['pos_model'] = args.pos_model
    parameters['cluster'] = args.cluster
    parameters['ying_stem'] = args.ying_stem
    parameters['gaz'] = args.gaz

    sentences = load_sentences(args.bio_input)

    feats, stem = generate_features(sentences, parameters)

    # output bio with features
    if feats:
        bio = []
        for i, s in enumerate(sentences):
            bio_s = []
            for j, w in enumerate(s):
                bio_s.append(' '.join(w[:args.feat_column] + feats[i][j] +
                                      w[args.feat_column:]))
            bio.append('\n'.join(bio_s))
        with open(args.bio_output, 'w') as f:
            f.write('\n\n'.join(bio))
    else:
        shutil.copy(args.bio_input, args.bio_output)


