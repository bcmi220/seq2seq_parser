import os
import sys
import argparse
import time
import random
import torch
import itertools
import numpy as np
import torch.optim as optim
from collections import OrderedDict

from nn import SeqLabeling
from utils import create_input, Tee
from utils import evaluate
from loader import word_mapping, char_mapping, tag_mapping, feats_mapping
from loader import prepare_dataset, load_sentences
from loader import augment_with_pretrained
from dnn_utils import exp_lr_scheduler

# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", default="./subroot/data/predicate-train.txt",
    help="Train set location"
)
parser.add_argument(
    "--dev", default="./srl/subroot/predicate-dev.txt",
    help="Dev set location"
)
parser.add_argument(
    "--test", default="./srl/subroot/predicate-test.txt",
    help="Test set location"
)
parser.add_argument(
    "--model_dp", default="./model/",
    help="model directory path"
)
parser.add_argument(
    "--result_dp", default="./result/",
    help="result directory path"
)
parser.add_argument(
    "--lower", default='1',
    type=int, help="Lowercase words (this will not affect character inputs)"
)
parser.add_argument(
    "--zeros", default="1",
    type=int, help="Replace digits with 0"
)
parser.add_argument(
    "--char_dim", default="0", # 25
    type=int, help="Char embedding dimension"
)
parser.add_argument(
    "--char_lstm_dim", default="0", # 25
    type=int, help="Char LSTM hidden layer size"
)
parser.add_argument(
    "--char_conv_channel", default="0", # 1
    type=int, help="Use CNN to generate character embeddings. (0 to disable)"
)
parser.add_argument(
    "--word_dim", default="100",
    type=int, help="Token embedding dimension"
)
parser.add_argument(
    "--word_lstm_dim", default="512",
    type=int, help="Token LSTM hidden layer size"
)
parser.add_argument(
    "--word_lstm_layer", default="4",
    type=int, help="Token LSTM hidden layer number"
)
parser.add_argument(
    "--hidden_layer1_size", default="256",
    type=int, help="Token LSTM hidden layer number"
)
parser.add_argument(
    "--hidden_layer2_size", default="128",
    type=int, help="Token LSTM hidden layer number"
)
parser.add_argument(
    "--hidden_layer3_size", default="64",
    type=int, help="Token LSTM hidden layer number"
)
parser.add_argument(
    "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
parser.add_argument(
    "--all_emb", default="0",
    type=int, help="Load all embeddings"
)
parser.add_argument(
    "--cap_dim", default="0",
    type=int, help="Capitalization feature dimension (0 to disable)"
)
parser.add_argument(
    "--feat_dim", default="100",
    type=int, help="dimension for each feature."
)
parser.add_argument(
    '--feat_column',
    type=int, default=2,
    help='the number of the column where features start. default is 1, '
         'the 2nd column.'
)
parser.add_argument(
    "--crf", default="1",
    type=int, help="Use CRF (0 to disable)"
)
parser.add_argument(
    "--dropout", default="0.1",
    type=float, help="Droupout on the input (0 = no dropout)"
)
parser.add_argument(
    "--lr_method", default="adam-init_lr=.001-lr_decay_epoch=100", #sgd-lr_.005 sgd-init_lr=.001-lr_decay_epoch=100
    help="Learning method (SGD, Adadelta, Adam..)"
)
parser.add_argument(
    "--reload", default="0",
    type=int, help="Reload the last saved model"
)
parser.add_argument(
    "--num_epochs", default="100",
    type=int, help="Number of training epochs"
)
parser.add_argument(
    "--batch_size", default="64",
    type=int, help="Batch size."
)
parser.add_argument(
    "--gpu", default="0",
    type=int, help="default is 0. set 1 to use gpu."
)


args = parser.parse_args()

# parse model dir
model_dir = args.model_dp
result_dir = args.result_dp

# Parse parameters
parameters = OrderedDict()
parameters['lower'] = args.lower == 1
parameters['zeros'] = args.zeros == 1
parameters['char_dim'] = args.char_dim
parameters['char_lstm_dim'] = args.char_lstm_dim
parameters['char_conv_channel'] = args.char_conv_channel
parameters['word_dim'] = args.word_dim
parameters['word_lstm_dim'] = args.word_lstm_dim
parameters['word_lstm_layer'] = args.word_lstm_layer
parameters['hidden_layer1_size'] = args.hidden_layer1_size
parameters['hidden_layer2_size'] = args.hidden_layer2_size
parameters['hidden_layer3_size'] = args.hidden_layer3_size
parameters['pre_emb'] = args.pre_emb
parameters['all_emb'] = args.all_emb == 1
parameters['cap_dim'] = args.cap_dim
parameters['feat_dim'] = args.feat_dim
parameters['feat_column'] = args.feat_column
parameters['crf'] = args.crf == 1
parameters['dropout'] = args.dropout
parameters['lr_method'] = args.lr_method
parameters['num_epochs'] = args.num_epochs
parameters['batch_size'] = args.batch_size

# generate model name
model_dir = os.path.join(model_dir, str(time.time()))

print('Workspace path:{}'.format(os.getcwd()))

# Check parameters validity
assert os.path.isfile(args.train)
assert os.path.isfile(args.dev)
assert os.path.isfile(args.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert parameters['word_lstm_layer'] > 0
assert parameters['hidden_layer1_size'] > 0 and parameters['hidden_layer2_size'] > 0 and parameters['hidden_layer3_size'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation folders
os.makedirs(model_dir, exist_ok=True)

# register logger to save print(messages to both stdout and disk)
training_log_path = os.path.join(model_dir, 'training_log.txt')
if os.path.exists(training_log_path):
    os.remove(training_log_path)
f = open(training_log_path, 'w')
sys.stdout = Tee(sys.stdout, f)

print('Training data: %s' % args.train)
print('Dev data: %s' % args.dev)
print('Test data: %s' % args.test)
print("Model location: %s" % model_dir)
print('Model parameters:')
for k, v in parameters.items():
    print('%s=%s' % (k, v))

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']

# Load sentences
train_sentences = load_sentences(args.train, lower, zeros)
dev_sentences = load_sentences(args.dev, lower, zeros)
test_sentences = load_sentences(args.test, lower, zeros)

# train_sentences = train_sentences[:50]
# dev_sentences = dev_sentences[:50]
# test_sentences = test_sentences[:50]

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
# create a dictionary and a mapping for each feature
dico_feats_list, feat_to_id_list, id_to_feat_list = feats_mapping(
    train_sentences, parameters['feat_column']
)

parameters['label_size'] = len(id_to_tag)
parameters['word_vocab_size'] = len(id_to_word)
parameters['char_vocab_size'] = len(id_to_char)
parameters['feat_vocab_size'] = [len(item) for item in id_to_feat_list]

# Index data
dataset = dict()
dataset['train'] = prepare_dataset(
    train_sentences, parameters['feat_column'],
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, lower
)
dataset['dev'] = prepare_dataset(
    dev_sentences, parameters['feat_column'],
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, lower
)
dataset['test'] = prepare_dataset(
    test_sentences, parameters['feat_column'],
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(dataset['train']), len(dataset['dev']), len(dataset['test'])))

# initialize model
print('model initializing...')
model = SeqLabeling(parameters)
model.load_pretrained(id_to_word, **parameters)

# Parse optimization method parameters
lr_method = parameters['lr_method']
if "-" in lr_method:
    lr_method_name = lr_method[:lr_method.find('-')]
    lr_method_parameters = {}
    for x in lr_method[lr_method.find('-') + 1:].split('-'):
        split = x.split('=')
        assert len(split) == 2
        lr_method_parameters[split[0]] = float(split[1])
else:
    lr_method_name = lr_method
    lr_method_parameters = {}

# initialize optimizer function
if lr_method_name == 'sgd':
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
else:
    print('unknown optimization method.')

since = time.time()
best_model = model
best_dev = 0.0
best_test = 0.0
metric = 'f1'
num_epochs = args.num_epochs
batch_size = args.batch_size

for epoch in range(num_epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    time_epoch_start = time.time()  # epoch start time

    # Each epoch has a training and validation phase
    for phase in ['train', 'dev', 'test'][:]:
        if phase == 'train':
            optimizer = exp_lr_scheduler(optimizer_ft, epoch,
                                         **lr_method_parameters)
            model.train(True)  # Set model to training mode
            random.shuffle(dataset[phase])
        else:
            model.train(False)  # Set model to evaluate mode

        epoch_loss = []

        # Iterate over data.
        preds = []
        for i in range(0, len(dataset[phase]), batch_size):
            inputs, seq_index_mapping, char_index_mapping, seq_len, char_len = \
                create_input(dataset[phase][i:i+batch_size], parameters)

            # forward
            outputs, loss = model.forward(inputs, seq_len, char_len, char_index_mapping)
            try:
                epoch_loss.append(loss.data[0])
            except AttributeError:
                pass

            # backward + optimize only if in training phase
            if phase == 'train':
                # zero the parameter gradients
                optimizer.zero_grad()

                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem
                # in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(model.parameters(), 5)

                optimizer.step()

                sys.stdout.write(
                    '%d instances processed. current batch loss: %f\r' %
                    (i, np.mean(epoch_loss))
                )
                sys.stdout.flush()
            else:
                if parameters['crf']:
                    preds += [outputs[seq_index_mapping[j]].data
                              for j in range(len(outputs))]
                else:
                    _, _preds = torch.max(outputs.data, 2)

                    preds += [
                        _preds[seq_index_mapping[j]][:seq_len[seq_index_mapping[j]]]
                        for j in range(len(seq_index_mapping))
                        ]

        if phase == 'train':
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            print('{} Loss: {:.4f}\n'.format(phase, epoch_loss))
        else:
            epoch_f1, epoch_acc = evaluate(phase, preds, dataset[phase], id_to_tag, result_dir)
            if metric == 'f1':
                epoch_score = epoch_f1
            elif metric == 'acc':
                epoch_score = epoch_acc
            print(
                '{} F1: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_f1, epoch_acc)
            )

        # deep copy the model
        if phase == 'dev' and epoch_score > best_dev:
            best_dev = epoch_score
            print('new best score on dev: %.4f' % best_dev)
            print('saving the current model to disk...')

            state = {
                'epoch': epoch + 1,
                'parameters': parameters,
                'mappings': {
                    'id_to_word': id_to_word,
                    'id_to_char': id_to_char,
                    'id_to_tag': id_to_tag,
                    'id_to_feat_list': id_to_feat_list  # boliang
                },
                'state_dict': model.state_dict(),
                'best_prec1': best_dev,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(model_dir, 'best_model.pth.tar'))
            with open(os.path.join(model_dir, 'best_dev.result'), 'w') as f:
                f.write('F1: {:.4f} Acc: {:.4f}\n'.format(epoch_f1, epoch_acc))

        if phase == 'test' and epoch_score > best_test:
            best_test = epoch_score
            print('new best score on test: %.4f' % best_test)
            with open(os.path.join(model_dir, 'best_test.result'), 'w') as f:
                f.write('F1: {:.4f} Acc: {:.4f}\n'.format(epoch_f1, epoch_acc))

    time_epoch_end = time.time()  # epoch end time
    print('epoch training time: %f seconds' % round(
        (time_epoch_end - time_epoch_start), 2))

