# we merge the golden subroot feature into train dataset for training
# and we use the predict subroot (by BiLSTM+CRF) feature into dev/train dataset

import os

def merge_train(input_file, origin_file, output_file):
    with open(input_file, 'r') as f:
        input_data = f.readlines()
    
    input_data = [line.split() for line in input_data if len(line.strip())>0]

    with open(origin_file, 'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []

    for i in range(len(data)):
        if len(data[i].strip()) > 0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    assert len(input_data) == len(origin_data)

    with open(output_file, 'w') as f:
        for i in range(len(input_data)):
            assert len(input_data[i]) == len(origin_data[i])
            line = []
            for j in range(len(input_data[i])):
                if int(origin_data[i][j][6]) == 0:
                    line.append(input_data[i][j]+'|1')
                else:
                    line.append(input_data[i][j]+'|0')
            f.write(' '.join(line))
            f.write('\n')


def merge_pred(input_file, subroot_pred_file, output_file):
    with open(input_file, 'r') as f:
        input_data = f.readlines()
    
    input_data = [line.split() for line in input_data if len(line.strip())>0]

    with open(subroot_pred_file, 'r') as f:
        data = f.readlines()

    pred_data = []
    sentence = []

    for i in range(len(data)):
        if len(data[i].strip()) > 0:
            sentence.append(data[i].strip().split('\t'))
        else:
            pred_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        pred_data.append(sentence)

    assert len(input_data) == len(pred_data)

    with open(output_file, 'w') as f:
        for i in range(len(input_data)):
            assert len(input_data[i]) == len(pred_data[i])
            line = []
            for j in range(len(input_data[i])):
                line.append(input_data[i][j]+'|'+pred_data[i][j][1])
            f.write(' '.join(line))
            f.write('\n')


if __name__ == '__main__':
    merge_train(os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_train.input'),
            os.path.join(os.path.dirname(__file__), '../ptb-sd/train_pro.conll'),
            os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_train_ws.input'))

    merge_pred(os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_dev.input'),
            os.path.join(os.path.dirname(__file__), '../../subroot/result/dev_predicate_95.94.pred'),
            os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_dev_ws.input'))

    merge_pred(os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_test.input'),
            os.path.join(os.path.dirname(__file__), '../../subroot/result/test_predicate_95.16.pred'),
            os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_test_ws.input'))