import os

def replace_unk(input_file):
    with open(input_file, 'r') as f:
        input_data = f.readlines()
    
    input_data = [line.split() for line in input_data if len(line.strip())>0]

    with open(input_file, 'w') as f:
        for i in range(len(input_data)):
            line = []
            for j in range(len(input_data[i])):
                if input_data[i][j] == '0' or input_data[i][j]=='1':
                    line.append(input_data[i][j])
                else:
                    line.append('0')
            f.write(' '.join(line))
            f.write('\n')


if __name__ == '__main__':
    replace_unk(os.path.join(os.path.dirname(__file__), '../results/subroot/tgt_ptb_sd_subroot_train.pred'))

    replace_unk(os.path.join(os.path.dirname(__file__), '../results/subroot/tgt_ptb_sd_subroot_dev.pred'))

    replace_unk(os.path.join(os.path.dirname(__file__), '../results/subroot/tgt_ptb_sd_subroot_test.pred'))