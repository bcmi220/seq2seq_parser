import math

def stat_f1(pred_file):
    with open(pred_file,'r') as f:
        data_lines = f.readlines()

    # split by sentence
    pred_data = []
    sentence_data = []
    for line in data_lines:
        if len(line.strip()) > 0:
            sentence_data.append(line.strip().split("\t"))
        else:
            pred_data.append(sentence_data)
            sentence_data = []
    if len(sentence_data)>0:
        pred_data.append(sentence_data)
        sentence_data = []

    tps = [0 for _ in range(7)]
    fps = [0 for _ in range(7)]
    fns = [0 for _ in range(7)]
    f1s = [0 for _ in range(7)]
    for sentence in pred_data:
        idx = math.ceil(len(sentence)/10)-1
        if idx >= 7:
            continue
        for line in sentence:
            if line[0]!='0' and line[1] == line[0]:
                tps[idx] += 1
            if line[0]!='0' and line[1] != line[0]:
                fps[idx] += 1
            if line[0]=='0' and line[1] != line[0]:
                fns[idx] += 1

    for i in range(7):
        p = tps[i] / (tps[i] + fps[i] + 1e-13)
        r = tps[i] / (tps[i] + fns[i] + 1e-13)
        f1s[i] = 2 * p * r / ( p + r + 1e-13)

    return f1s
    

if __name__ == '__main__':
    print('\ndev:')
    print(stat_f1('../result/dev_predicate_96.53.pred'))

    print('\ntest:')
    print(stat_f1('../result/test_predicate_95.45.pred'))