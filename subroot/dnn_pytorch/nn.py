import torch
import re
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from loader import load_embedding
from dnn_utils import init_param, init_variable, log_sum_exp, sequence_mask
from dnn_utils import LongTensor, FloatTensor


class SeqLabeling(nn.Module):
    def __init__(self, model_param):
        super(SeqLabeling, self).__init__()

        #
        # model parameters
        #
        self.model_param = model_param
        word_vocab_size = model_param['word_vocab_size']
        char_vocab_size = model_param['char_vocab_size']
        feat_vocab_size = model_param['feat_vocab_size']
        word_dim = model_param['word_dim']
        word_lstm_dim = model_param['word_lstm_dim']
        char_dim = model_param['char_dim']
        char_lstm_dim = model_param['char_lstm_dim']
        feat_dim = model_param['feat_dim']
        crf = model_param['crf']
        dropout = model_param['dropout']
        char_conv_channel = model_param['char_conv_channel']
        label_size = model_param['label_size']

        # initialize word lstm input dim to 1 because of flags
        word_lstm_input_dim = 0
        
        #
        # word embedding layer
        #
        self.word_emb = nn.Embedding(word_vocab_size, word_dim)
        word_lstm_input_dim += word_dim

        #
        # char embedding layer
        #
        if char_dim:
            self.char_dim = char_dim
            self.char_emb = nn.Embedding(char_vocab_size, char_dim)

        #
        # bi-lstm char layer
        #
        if char_lstm_dim:
            self.char_lstm_init_hidden = (
                Parameter(torch.randn(2, 1, char_lstm_dim).type(
                    FloatTensor)),
                Parameter(torch.randn(2, 1, char_lstm_dim).type(
                    FloatTensor))
            )
            self.char_lstm_dim = char_lstm_dim
            self.char_lstm = nn.LSTM(char_dim, char_lstm_dim, 1,
                                     bidirectional=True, batch_first=True)
            word_lstm_input_dim += 2 * char_lstm_dim

        # cnn char layer
        if char_conv_channel:
            max_length = 25
            out_channel = 50
            kernel_sizes = [2, 3, 4]
            kernel_shape = []
            for i in range(len(kernel_sizes)):
                kernel_shape.append([char_dim, out_channel, kernel_sizes[i]])
            kernel_shape = np.array(kernel_shape)
            pool_sizes = [max_length - 2 + 1,
                          max_length - 3 + 1,
                          max_length - 4 + 1]
            self.multi_convs = MultiLeNetConv1dLayer(kernel_shape, pool_sizes)
            word_lstm_input_dim += out_channel * len(kernel_sizes)

        #
        # feat dim
        #
        if feat_vocab_size:
            self.feat_emb = [nn.Embedding(v, feat_dim) for v in feat_vocab_size]
            word_lstm_input_dim += len(self.feat_emb) * feat_dim
        else:
            self.feat_emb = []

        #
        # dropout for word bi-lstm layer
        #
        if dropout:
            self.word_lstm_dropout = nn.Dropout(p=dropout)

        #
        # word bi-lstm layer
        #
        self.word_lstm_init_hidden = (
            Parameter(torch.randn(2, 1, word_lstm_dim).type(FloatTensor)),
            Parameter(torch.randn(2, 1, word_lstm_dim).type(FloatTensor)),
        )
        self.word_lstm_dim = word_lstm_dim
        self.word_lstm = nn.LSTM(word_lstm_input_dim, word_lstm_dim, 1,
                                 bidirectional=True, batch_first=True)

        #
        # tanh layer
        #
        tanh_layer_input_dim = 2 * word_lstm_dim
        self.tanh_linear = nn.Linear(tanh_layer_input_dim,
                                     word_lstm_dim)

        #
        # linear layer before loss
        #
        self.linear = nn.Linear(word_lstm_dim, label_size)

        #
        # loss
        #
        if crf:
            self.criterion = CRFLoss(label_size)
        else:
            self.softmax = nn.Softmax()
            self.criterion = CrossEntropyLoss()

        #
        # initialize weights of each layer
        #
        self.init_weights()

    def init_weights(self):
        init_param(self.word_emb)

        if self.model_param['char_dim']:
            init_param(self.char_emb)
        if self.model_param['char_lstm_dim']:
            init_param(self.char_lstm)
            self.char_lstm.flatten_parameters()
        if self.model_param['char_conv_channel']:
            init_param(self.multi_convs)
        if self.feat_emb:
            for f_e in self.feat_emb:
                init_param(f_e)

        init_param(self.word_lstm)
        self.word_lstm.flatten_parameters()

        init_param(self.tanh_linear)

        init_param(self.linear)

    def load_pretrained(self, id_to_word, pre_emb, word_dim, **kwargs):
        if not pre_emb:
            return

        # Initialize with pretrained embeddings
        new_weights = self.word_emb.weight.data
        print('Loading pretrained embeddings from %s...' % pre_emb)
        pretrained = {}
        emb_invalid = 0
        for i, line in enumerate(load_embedding(pre_emb)):
            if type(line) == bytes:
                try:
                    line = str(line, 'utf-8')
                except UnicodeDecodeError:
                    continue
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                pretrained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            print('WARNING: %i invalid lines' % emb_invalid)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        # Lookup table initialization
        for i in range(len(id_to_word)):
            word = id_to_word[i]
            if word in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[word])
                c_found += 1
            elif word.lower() in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[word.lower()])
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pretrained:
                new_weights[i] = torch.from_numpy(
                    pretrained[re.sub('\d', '0', word.lower())]
                )
                c_zeros += 1
        self.word_emb.weight = nn.Parameter(new_weights)

        print('Loaded %i pretrained embeddings.' % len(pretrained))
        print('%i / %i (%.4f%%) words have been initialized with '
              'pretrained embeddings.' % (
                  c_found + c_lower + c_zeros, len(id_to_word),
                  100. * (c_found + c_lower + c_zeros) / len(id_to_word)
              ))
        print('%i found directly, %i after lowercasing, '
              '%i after lowercasing + zero.' % (
                  c_found, c_lower, c_zeros
              ))

    def forward(self, inputs, seq_len, char_len, char_index_mapping):
        seq_len = np.array(seq_len)
        char_len = np.array(char_len)
        batch_size = len(seq_len)

        word_lstm_input = []
        #
        # word embeddings
        #
        words = inputs['words']

        word_emb = self.word_emb(words.type(LongTensor))

        word_lstm_input.append(word_emb)

        #
        # char embeddings
        #
        char_repr = []
        if self.model_param['char_dim']:
            chars = inputs['chars']
            char_emb = self.char_emb(chars.type(LongTensor))

        #
        # char bi-lstm embeddings
        #
        if self.model_param['char_lstm_dim']:
            lstm_char_emb = char_emb[:, :char_len[0]]
            char_lstm_dim = self.model_param['char_lstm_dim']
            char_lstm_init_hidden = (
                self.char_lstm_init_hidden[0].expand(2, len(char_len), char_lstm_dim).contiguous(),
                self.char_lstm_init_hidden[1].expand(2, len(char_len), char_lstm_dim).contiguous(),
            )
            lstm_char_emb = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_char_emb, char_len, batch_first=True
            )
            char_lstm_out, char_lstm_h = self.char_lstm(
                lstm_char_emb, char_lstm_init_hidden
            )
            char_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                char_lstm_out, batch_first=True
            )
            char_lstm_h = char_lstm_h[0].permute(1, 0, 2).contiguous().view(len(char_len), 2*char_lstm_dim)
            char_repr.append(char_lstm_h)

        #
        # char CNN embeddings
        #
        if self.model_param['char_conv_channel']:
            cnn_char_emb = char_emb[:, :25]
            char_cnn_out = self.multi_convs(cnn_char_emb)
            char_repr += char_cnn_out

        if char_repr:
            char_repr = torch.cat(char_repr, dim=1)

            char_index_mapping = LongTensor([char_index_mapping[k] for k in range(len(char_len))])
            char_repr = char_repr[char_index_mapping]

            char_repr_padded_seq = nn.utils.rnn.PackedSequence(data=char_repr, batch_sizes=seq_len.tolist())
            char_repr, _ = nn.utils.rnn.pad_packed_sequence(
                char_repr_padded_seq
            )
            word_lstm_input.append(char_repr)

        #
        # feat input
        #
        if self.feat_emb:
            feat_emb = []
            for i, f_e in enumerate(self.feat_emb):
                feat = inputs['feats'][:, :, i]
                feat_emb.append(f_e(feat.type(LongTensor)))
            word_lstm_input += feat_emb

        #
        # bi-directional lstm
        #
        word_lstm_dim = self.model_param['word_lstm_dim']
        word_lstm_input = torch.cat(word_lstm_input, dim=2)
        # dropout
        if self.model_param['dropout']:
            word_lstm_input = self.word_lstm_dropout(word_lstm_input)

        word_lstm_init_hidden = (
            self.word_lstm_init_hidden[0].expand(2, batch_size, word_lstm_dim).contiguous(),
            self.word_lstm_init_hidden[1].expand(2, batch_size, word_lstm_dim).contiguous()
        )

        word_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            word_lstm_input, seq_len, batch_first=True
        )
        word_lstm_out, word_lstm_h = self.word_lstm(
            word_lstm_input, word_lstm_init_hidden
        )
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            word_lstm_out, batch_first=True
        )

        #
        # tanh layer
        #
        tanh_out = nn.Tanh()(self.tanh_linear(word_lstm_out))

        #
        # fully connected layer
        #
        linear_out = self.linear(tanh_out)

        #
        # softmax or crf layer
        #
        if type(self.criterion) == CrossEntropyLoss:
            outputs = torch.stack(
                [self.softmax(linear_out[i]) for i in range(batch_size)], 0
            )
        elif type(self.criterion) == CRFLoss and not self.training:
            preds = linear_out
            outputs = self.criterion(
                preds, None, seq_len, viterbi=True, return_best_sequence=True
            )
        else:
            outputs = None

        #
        # compute batch loss
        #
        loss = 0
        if self.training:
            if type(self.criterion) == CrossEntropyLoss:
                preds = outputs
            elif type(self.criterion) == CRFLoss:
                preds = linear_out
            reference = inputs['tags']

            loss = self.criterion(preds, reference, seq_len)
            loss /= batch_size

        return outputs, loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, ref, seq_len):
        batch_size = pred.size(0)
        max_seq_len = pred.size(1)

        mask = sequence_mask(seq_len)
        mask = Variable(torch.from_numpy(mask).type(FloatTensor))

        # compute cross entropy loss
        loss = - torch.log(pred)[
            torch.from_numpy(
                np.array([np.arange(batch_size)] * max_seq_len).transpose()
            ).type(LongTensor),
            torch.from_numpy(
                np.array([np.arange(max_seq_len)] * batch_size)
            ).type(LongTensor),
            ref.data
        ]
        loss = torch.sum(loss * mask)

        return loss


class CRFLoss(nn.Module):
    def __init__(self, num_labels):
        super(CRFLoss, self).__init__()

        self.num_labels = num_labels

        # trans_array = init_variable((num_labels+2, num_labels+2))
        # trans_array[:, self.num_labels] = -1000
        # trans_array[self.num_labels+1, :] = -1000

        self.transitions = Parameter(
            torch.from_numpy(init_variable((num_labels+2, num_labels+2))).type(FloatTensor)
        )

    def forward(self, pred, ref, seq_len,
                viterbi=False, return_best_sequence=False):
        # get batch info
        batch_size = pred.size(0)
        max_seq_len = pred.size(1)
        label_size = pred.size(2)

        # add padding to observations.
        small = -1000
        b_s_array = np.array(
            [[[small] * self.num_labels + [0, small]]] * batch_size
        ).astype(np.float32)
        b_s = Variable(torch.from_numpy(b_s_array).type(FloatTensor))
        right_padding_array = np.array(
            [[[0] * self.num_labels + [small, small]]] * batch_size
        ).astype(np.float32)
        right_padding = Variable(
            torch.from_numpy(right_padding_array).type(FloatTensor)
        )
        observations = torch.cat(
            [pred,
             Variable(
                 small * torch.ones((batch_size, max_seq_len, 2)).type(FloatTensor)
             )],
            dim=2
        )
        observations = torch.cat(
            [b_s, observations, right_padding],
            dim=1
        )

        # because of various length in batch, add e_s to the real end of each
        # sequence
        e_s = np.array([small] * self.num_labels + [0, 1000]).astype(np.float32)
        e_s_mask = np.zeros(observations.size())
        for i in range(batch_size):
            e_s_mask[i][seq_len[i]+1] = e_s
        observations += Variable(torch.from_numpy(e_s_mask).type(FloatTensor))

        # compute all path scores
        paths_scores = Variable(
            FloatTensor(max_seq_len+1, batch_size, label_size+2)
        )
        paths_indices = Variable(
            LongTensor(max_seq_len+1, batch_size, label_size+2)
        )
        previous = observations[:, 0]
        for i in range(1, observations.size(1)):
            obs = observations[:, i]
            _previous = torch.unsqueeze(previous, 2)
            _obs = torch.unsqueeze(obs, 1)
            if viterbi:
                scores = _previous + _obs + self.transitions
                out, out_indices = scores.max(dim=1)
                if return_best_sequence:
                    paths_indices[i-1] = out_indices
                paths_scores[i-1] = out
                previous = out
            else:
                previous = log_sum_exp(_previous + _obs + self.transitions,
                                       dim=1)
                paths_scores[i-1] = previous

        paths_scores = paths_scores.permute(1, 0, 2)
        paths_indices = paths_indices.permute(1, 0, 2)

        all_paths_scores = log_sum_exp(
            paths_scores[
                torch.from_numpy(np.arange(batch_size)).type(LongTensor),
                torch.from_numpy(seq_len).type(LongTensor)
            ],
            dim=1
        ).sum()

        # return indices of best paths.
        if return_best_sequence:
            sequence = []
            for i in range(len(paths_indices)):
                p_indices = paths_indices[i][:seq_len[i]+1]
                p_score = paths_scores[i][:seq_len[i]+1]
                _, previous = p_score[-1].max(dim=0)
                seq = []
                for j in reversed(range(len(p_score))):
                    s = p_indices[j]
                    previous = s[previous]
                    seq.append(previous)

                seq = torch.cat(seq[::-1]+[p_score[-1].max(dim=0)[1]])

                sequence.append(seq[1:-1])

            return sequence

        # compute real path score if reference is provided
        if ref is not None:
            # Score from tags
            real_path_mask = Variable(
                torch.from_numpy(sequence_mask(seq_len))
            ).type(FloatTensor)
            real_path_score = pred[
                torch.from_numpy(
                    np.array([np.arange(batch_size)]*max_seq_len).transpose()
                ).type(LongTensor),
                torch.from_numpy(
                    np.array([np.arange(max_seq_len)]*batch_size)
                ).type(LongTensor),
                ref.data
            ]
            real_path_score = torch.sum(real_path_score * real_path_mask)

            # Score from transitions
            b_id = Variable(
                torch.from_numpy(
                    np.array([[self.num_labels]] * batch_size)
                ).type(LongTensor)
            )
            right_padding = Variable(torch.zeros(b_id.size())).type(LongTensor)

            padded_tags_ids = torch.cat([b_id, ref, right_padding], dim=1)

            # because of various length in batch, add e_id to the real end of
            # each sequence
            e_id = np.array([self.num_labels+1])
            e_id_mask = np.zeros(padded_tags_ids.size())
            for i in range(batch_size):
                e_id_mask[i][seq_len[i] + 1] = e_id

            padded_tags_ids += Variable(
                torch.from_numpy(e_id_mask)
            ).type(LongTensor)

            # mask out padding in batch
            transition_score_mask = Variable(
                torch.from_numpy(sequence_mask(seq_len+1))
            ).type(FloatTensor)
            real_transition_score = self.transitions[
                padded_tags_ids[
                :, torch.from_numpy(np.arange(max_seq_len + 1)).type(LongTensor)
                ].data,
                padded_tags_ids[
                :, torch.from_numpy(np.arange(max_seq_len + 1) + 1).type(LongTensor)
                ].data
            ]
            real_path_score += torch.sum(
                real_transition_score * transition_score_mask
            )

            # compute loss
            loss = all_paths_scores - real_path_score

            return loss


class MultiLeNetConv1dLayer(nn.Module):
    def __init__(self, kernel_shape, pool_sizes):
        super(MultiLeNetConv1dLayer, self).__init__()

        num_conv = kernel_shape.shape[0]
        in_channels = kernel_shape[:, 0]
        out_channels = kernel_shape[:, 1]
        kernel_size = kernel_shape[:, 2]

        self.conv_nets = []
        self.max_pool1d = []
        for i in range(num_conv):
            conv = nn.Conv1d(int(in_channels[i]), int(out_channels[i]), int(kernel_size[i]))
            self.conv_nets.append(conv)

            max_pool1d = nn.MaxPool1d(pool_sizes[i])
            self.max_pool1d.append(max_pool1d)
        self.conv_nets = nn.ModuleList(self.conv_nets)
        self.max_pool1d = nn.ModuleList(self.max_pool1d)

    def forward(self, input):
        conv_out = []
        input = input.permute(0, 2, 1)
        for conv in self.conv_nets:
            conv_out.append(conv(input))

        pooling_out = []
        for i, pool in enumerate(self.max_pool1d):
            pooling_out.append(pool(conv_out[i]).squeeze())

        return pooling_out