import os
import tqdm

def make_input(file_name, src_path, tgt_path):
    with open(file_name, 'r') as f:
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

    src_data = []
    tgt_data = []
    for sentence in origin_data:
        src_line = []
        tgt_line = []
        for line in sentence:
            dep_ind = int(line[0])
            head_ind = int(line[6])
            if head_ind == 0:
                tag = '1'
            else:
                tag = '0'
            # word = ''.join([c if not c.isdigit() else '0' for c in line[1].lower()])
            is_number = False
            word = line[1].lower()
            for c in word:
                if c.isdigit():
                    is_number = True
                    break
            if is_number:
                word = 'number'
            src_line.append([word, line[4]])
            tgt_line.append(tag)
        if len(src_line) > 1:
            src_data.append(src_line)
            tgt_data.append(tgt_line)

    with open(src_path, 'w') as f:
        for line in src_data:
            f.write(' '.join(['|'.join(item) for item in line]))
            f.write('\n')


    with open(tgt_path, 'w') as f:
        for line in tgt_data:
            f.write(' '.join(line))
            f.write('\n')

if __name__ == '__main__':
    train_file = os.path.join(os.path.dirname(__file__), '../ptb-sd/train_pro_wsd.conll')
    dev_file = os.path.join(os.path.dirname(__file__), '../ptb-sd/dev_pro.conll')
    test_file = os.path.join(os.path.dirname(__file__), '../ptb-sd/test_pro.conll')

    make_input(train_file, 
               os.path.join(os.path.dirname(__file__), '../input/subroot/src_ptb_sd_subroot_train.input'),
               os.path.join(os.path.dirname(__file__), '../input/subroot/tgt_ptb_sd_subroot_train.input'))
    
    make_input(dev_file, 
               os.path.join(os.path.dirname(__file__), '../input/subroot/src_ptb_sd_subroot_dev.input'),
               os.path.join(os.path.dirname(__file__), '../input/subroot/tgt_ptb_sd_subroot_dev.input'))
    
    make_input(test_file, 
               os.path.join(os.path.dirname(__file__), '../input/subroot/src_ptb_sd_subroot_test.input'),
               os.path.join(os.path.dirname(__file__), '../input/subroot/tgt_ptb_sd_subroot_test.input'))
