import os

def make_wsd_pred(pred_file, map_file, output_file):
    with open(pred_file, 'r') as f:
        pred_data = f.readlines()
    
    pred_data = [line.split() for line in pred_data if len(line.strip())>0]


    with open(map_file, 'r') as f:
        map_data = f.readlines()
    
    map_data = [line.strip() for line in map_data if len(line.strip())>0]

    output_data = []

    assert len(map_data) == len(pred_data)

    sent_len = len(map_data)
    sent_line = []
    for i in range(sent_len):
        if len(sent_line) == 0:
            sent_line = pred_data[i]
        else:
            if map_data[i] == map_data[i-1]:
                sent_line[-1] = '<SUBROOT>'
                sent_line += pred_data[i][1:]
            else:
                output_data.append(sent_line)
                sent_line = pred_data[i]
    
    if len(sent_line)>0:
        output_data.append(sent_line)
    
    with open(output_file, 'w') as f:
        for i in range(len(output_data)):
            for j in range(len(output_data[i])):
                if output_data[i][j] == '<SUBROOT>':
                    output_data[i][j] = 'L'+str(j+1)
            f.write(' '.join(output_data[i]))
            f.write('\n')


if __name__ == '__main__':
    # make_wsd_pred(os.path.join(os.path.dirname(__file__), '../results/dp/tgt_ptb_sd_dev_wsd_30.pred'),
    #                            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_dev_wsd_30_map.input'))

    make_wsd_pred(os.path.join(os.path.dirname(__file__), '../results/dp/tgt_ptb_sd_test_wsd_40.pred'),
                               os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_test_wsd_40_map.input'),
                               os.path.join(os.path.dirname(__file__), '../results/dp/tgt_ptb_sd_test_wsd_40_org.pred'))

