import os
from collections import Counter

def load_data(path):
    with open(path, 'r') as f:
        data = f.readlines()

    data = [line.strip().split() for line in data if len(line.strip())>0]

    return data


def f1(target, predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0
    correct = 0
    assert len(target) == len(predict)
    for i in range(len(target)):
        assert len(target[i]) == len(predict[i])
        for j in range(len(target[i])):
            total += 1
            if target[i][j] == predict[i][j]:
                correct += 1
            assert predict[i][j] == '0' or predict[i][j] == '1'
            if target[i][j] == '1' and target[i][j] == predict[i][j]:
                TP += 1
            if target[i][j] == '0' and target[i][j] == predict[i][j]:
                TN += 1
            if target[i][j] == '0' and target[i][j] != predict[i][j]:
                FP += 1
            if target[i][j] == '1' and target[i][j] != predict[i][j]:
                FN += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

    print('eval Acc:{:.2f} P:{:.2f} R:{:.2f} F1:{:.2f}'.format(correct/total*100, P * 100, R * 100, F1 * 100))

if __name__ == '__main__':
    f1(load_data(os.path.join(os.path.dirname(__file__), '../input/subroot/tgt_ptb_sd_subroot_train.input')),
        load_data(os.path.join(os.path.dirname(__file__), '../results/subroot/tgt_ptb_sd_subroot_train.pred')))

    f1(load_data(os.path.join(os.path.dirname(__file__), '../input/subroot/tgt_ptb_sd_subroot_dev.input')),
        load_data(os.path.join(os.path.dirname(__file__), '../results/subroot/tgt_ptb_sd_subroot_dev.pred')))
    
    f1(load_data(os.path.join(os.path.dirname(__file__), '../input/subroot/tgt_ptb_sd_subroot_test.input')),
        load_data(os.path.join(os.path.dirname(__file__), '../results/subroot/tgt_ptb_sd_subroot_test.pred')))
