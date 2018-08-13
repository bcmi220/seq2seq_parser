import argparse
import time
import torch

from nn import SeqLabeling
from utils import create_input, iobes_iob
from loader import prepare_dataset, load_sentences


# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="",
    help="Model location"
)
parser.add_argument(
    "--input", default="",
    help="Input bio file location"
)
parser.add_argument(
    "--output", default="",
    help="Output bio file location"
)
parser.add_argument(
    "--batch_size", default="50",
    type=int, help="batch size"
)
parser.add_argument(
    "--gpu", default="0",
    type=int, help="default is 0. set 1 to use gpu."
)
args = parser.parse_args()

print('loading model from:', args.model)
if args.gpu:
    state = torch.load(args.model)
else:
    state = torch.load(args.model, map_location=lambda storage, loc: storage)

parameters = state['parameters']
mappings = state['mappings']

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [mappings['id_to_word'], mappings['id_to_char'], mappings['id_to_tag']]
    ]
feat_to_id_list = [
    {v: k for k, v in id_to_feat.items()}
    for id_to_feat in mappings['id_to_feat_list']
    ]

# eval sentences
eval_sentences = load_sentences(
    args.input,
    parameters['lower'],
    parameters['zeros']
)

eval_dataset = prepare_dataset(
    eval_sentences, parameters['feat_column'],
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, parameters['lower'],
    is_train=False
)

print("%i sentences in eval set." % len(eval_dataset))

# initialize model
model = SeqLabeling(parameters)
model.load_state_dict(state['state_dict'])
model.train(False)

since = time.time()
batch_size = args.batch_size
f_output = open(args.output, 'w')

# Iterate over data.
print('tagging...')
for i in range(0, len(eval_dataset), batch_size):
    inputs, seq_index_mapping, char_index_mapping, seq_len, char_len = \
        create_input(eval_dataset[i:i+batch_size], parameters, add_label=False)

    # forward
    outputs, loss = model.forward(inputs, seq_len, char_len, char_index_mapping)
    if parameters['crf']:
        preds = [outputs[seq_index_mapping[j]].data
                 for j in range(len(outputs))]
    else:
        _, _preds = torch.max(outputs.data, 2)

        preds = [
            _preds[seq_index_mapping[j]][:seq_len[seq_index_mapping[j]]]
            for j in range(len(seq_index_mapping))
            ]
    for j, pred in enumerate(preds):
        pred = [mappings['id_to_tag'][p] for p in pred]
        # Output tags in the IOB2 format
        if parameters['tag_scheme'] == 'iobes':
            pred = iobes_iob(pred)
        # Write tags
        assert len(pred) == len(eval_sentences[i+j])
        f_output.write('%s\n\n' % '\n'.join('%s%s%s' % (' '.join(w), ' ', z)
                                            for w, z in zip(eval_sentences[i+j],
                                                            pred)))
        if (i + j + 1) % 500 == 0:
            print(i+j+1)

end = time.time()  # epoch end time
print('time elapssed: %f seconds' % round(
    (end - since), 2))

