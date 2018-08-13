import os

def preprocess():
    raw_train_file = os.path.join(os.path.dirname(__file__),'./data/ptb-sd/train_pro.conll')
    raw_dev_file = os.path.join(os.path.dirname(__file__),'./data/ptb-sd/dev_pro.conll')
    raw_test_file = os.path.join(os.path.dirname(__file__),'./data/ptb-sd/test_pro.conll')

    predicate_train_file = os.path.join(os.path.dirname(__file__),'./data/subroot-train.txt')
    predicate_dev_file = os.path.join(os.path.dirname(__file__),'./data/subroot-dev.txt')
    predicate_test_file = os.path.join(os.path.dirname(__file__),'./data/subroot-test.txt')

    with open(raw_train_file, 'r') as f:
        with open(predicate_train_file, 'w') as fo:
            data = f.readlines()

            # read data
            sentence_data = []
            sentence = []
            for line in data:
                if len(line.strip()) > 0:
                    line = line.strip().split('\t')
                    sentence.append(line)
                else:
                    sentence_data.append(sentence)
                    sentence = []
            if len(sentence)>0:
                sentence_data.append(sentence)
                sentence = []

            # process data
            # copy the data by predicate num
            train_data = []
            for sentence in sentence_data:
                lines = []
                for i in range(len(sentence)):
                    is_subroot = '0'
                    if sentence[i][6] == '0':
                        is_subroot = '1'
                    word = sentence[i][1].lower()
                    # is_number = False
                    # for c in word:
                    #     if c.isdigit():
                    #         is_number = True
                    #         break
                    # if is_number:
                    #     word = 'number'
                    lines.append([word, sentence[i][4], is_subroot])
                train_data.append(lines)

            for sentence in train_data:
                fo.write('\n'.join(['\t'.join(line) for line in sentence]))
                fo.write('\n\n')

    with open(raw_dev_file, 'r') as f:
        with open(predicate_dev_file, 'w') as fo:
            data = f.readlines()

            # read data
            sentence_data = []
            sentence = []
            for line in data:
                if len(line.strip()) > 0:
                    line = line.strip().split('\t')
                    sentence.append(line)
                else:
                    sentence_data.append(sentence)
                    sentence = []
            if len(sentence)>0:
                sentence_data.append(sentence)
                sentence = []

            # process data
            # copy the data by predicate num
            dev_data = []
            for sentence in sentence_data:
                lines = []
                for i in range(len(sentence)):
                    is_subroot = '0'
                    if sentence[i][6] == '0':
                        is_subroot = '1'
                    word = sentence[i][1].lower()
                    # is_number = False
                    # for c in word:
                    #     if c.isdigit():
                    #         is_number = True
                    #         break
                    # if is_number:
                    #     word = 'number'
                    lines.append([word, sentence[i][4], is_subroot])
                dev_data.append(lines)

            for sentence in dev_data:
                fo.write('\n'.join(['\t'.join(line) for line in sentence]))
                fo.write('\n\n')

    with open(raw_test_file, 'r') as f:
        with open(predicate_test_file, 'w') as fo:
            data = f.readlines()

            # read data
            sentence_data = []
            sentence = []
            for line in data:
                if len(line.strip()) > 0:
                    line = line.strip().split('\t')
                    sentence.append(line)
                else:
                    sentence_data.append(sentence)
                    sentence = []
            if len(sentence)>0:
                sentence_data.append(sentence)
                sentence = []

            # process data
            # copy the data by predicate num
            test_data = []
            for sentence in sentence_data:
                lines = []
                for i in range(len(sentence)):
                    is_subroot = '0'
                    if sentence[i][6] == '0':
                        is_subroot = '1'
                    word = sentence[i][1].lower()
                    # is_number = False
                    # for c in word:
                    #     if c.isdigit():
                    #         is_number = True
                    #         break
                    # if is_number:
                    #     word = 'number'
                    lines.append([word, sentence[i][4], is_subroot])
                test_data.append(lines)

            for sentence in test_data:
                fo.write('\n'.join(['\t'.join(line) for line in sentence]))
                fo.write('\n\n')

                
if __name__ == '__main__':
    preprocess()