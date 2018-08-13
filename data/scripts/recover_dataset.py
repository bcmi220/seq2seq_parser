import os


def recover_data(file_name, pred_data, output_path):
    with open(file_name, 'r') as f:
        data = f.readlines()


    golden_data = []
    sentence = []

    for i in range(len(data)):
        if len(data[i].strip()) > 0:
            sentence.append(data[i].strip().split('\t'))
        else:
            golden_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        golden_data.append(sentence)

    with open(pred_data, 'r') as f:
        data = f.readlines()

    pred_data = [item.strip().split() for item in data if len(item.strip()) > 0]

    pred_index = 0
    for i in range(len(golden_data)):
        predicate_idx = 0
        for j in range(len(golden_data[i])):
            if golden_data[i][j][12] == 'Y':
                predicate_idx += 1
                for k in range(len(golden_data[i])):
                    golden_data[i][k][13 + predicate_idx] = pred_data[pred_index][k]
                pred_index += 1

    with open(output_path, 'w') as f:
        for sentence in golden_data:
            for line in sentence:
                f.write('\t'.join(line))
                f.write('\n')
            f.write('\n')

if __name__ == '__main__':
    recover_data(os.path.join(os.path.dirname(__file__), 'conll09-english/conll09_test.dataset'),
                 os.path.join(os.path.dirname(__file__), 'tgt_conll09_en_test.pred'),
                 os.path.join(os.path.dirname(__file__), 'conll09_en_test.dataset.pred'))
