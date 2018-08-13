import os

def subroot_decomposition_g(input_file, target_file, origin_file, output_file, output_target, dec_map_file, threshold):
    # input data
    with open(input_file, 'r') as f:
        input_data = f.readlines()
    
    input_data = [line.split() for line in input_data if len(line.strip())>0]

    # target data
    with open(target_file,'r') as f:
        target_data = f.readlines()

    target_data = [line.split() for line in target_data if len(line.strip())>0]

    # origin data
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

    assert len(input_data) == len(target_data) == len(origin_data)

    with open(output_file, 'w') as f:
        with open(output_target, 'w') as fo:
            with open(dec_map_file, 'w') as fd:
                for i in range(len(input_data)):
                    assert len(input_data[i]) == len(target_data[i]) == len(origin_data[i])
                    if len(input_data[i]) < threshold:
                        output_line = []
                        target_line = []
                        for j in range(len(input_data[i])):
                            output_line.append(input_data[i][j]) #+'|'+subroot_data[i][j]
                            target_line.append(target_data[i][j])
                        f.write(' '.join(output_line))
                        f.write('\n')
                        fo.write(' '.join(target_line))
                        fo.write('\n')
                        fd.write(str(i))
                        fd.write('\n')
                    else:
                        output_line = []
                        target_line = []
                        for j in range(len(input_data[i])):
                            if origin_data[i][j][6] == '0':
                                if len(output_line)>0:
                                    output_line.append(input_data[i][j]) #+'|'+subroot_data[i][j]
                                    target_data[i][j] = '<SUBROOT>'
                                    target_line.append(target_data[i][j])
                                    f.write(' '.join(output_line))
                                    f.write('\n')
                                    fo.write(' '.join(target_line))
                                    fo.write('\n')
                                    fd.write(str(i))
                                    fd.write('\n')
                                    output_line = []
                                    target_line = []
                            output_line.append(input_data[i][j]) #+'|'+subroot_data[i][j]
                            target_line.append(target_data[i][j])
                        if len(output_line)>1:
                            f.write(' '.join(output_line))
                            f.write('\n')
                            fo.write(' '.join(target_line))
                            fo.write('\n')
                            fd.write(str(i))
                            fd.write('\n')

def subroot_decomposition(input_file, target_file, subroot_file, output_file, output_target, dec_map_file, threshold):
    # input data
    with open(input_file, 'r') as f:
        input_data = f.readlines()
    
    input_data = [line.split() for line in input_data if len(line.strip())>0]

    # target data
    with open(target_file,'r') as f:
        target_data = f.readlines()

    target_data = [line.split() for line in target_data if len(line.strip())>0]

    # subroot data
    with open(subroot_file, 'r') as f:
        data = f.readlines()
    
    subroot_data = []
    sentence = []

    for i in range(len(data)):
        if len(data[i].strip()) > 0:
            sentence.append(data[i].strip().split('\t'))
        else:
            subroot_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        subroot_data.append(sentence)

    assert len(input_data) == len(target_data) == len(subroot_data)

    with open(output_file, 'w') as f:
        with open(output_target, 'w') as fo:
            with open(dec_map_file, 'w') as fd:
                for i in range(len(input_data)):
                    assert len(input_data[i]) == len(target_data[i]) == len(subroot_data[i])
                    if len(input_data[i]) < threshold:
                        output_line = []
                        target_line = []
                        for j in range(len(input_data[i])):
                            output_line.append(input_data[i][j]) #+'|'+subroot_data[i][j]
                            target_line.append(target_data[i][j])
                        f.write(' '.join(output_line))
                        f.write('\n')
                        fo.write(' '.join(target_line))
                        fo.write('\n')
                        fd.write(str(i))
                        fd.write('\n')
                    else:
                        output_line = []
                        target_line = []
                        for j in range(len(input_data[i])):
                            if subroot_data[i][j][1] == '1':
                                if len(output_line)>0:
                                    output_line.append(input_data[i][j]) #+'|'+subroot_data[i][j]
                                    target_data[i][j] = '<SUBROOT>'
                                    target_line.append(target_data[i][j])
                                    f.write(' '.join(output_line))
                                    f.write('\n')
                                    fo.write(' '.join(target_line))
                                    fo.write('\n')
                                    fd.write(str(i))
                                    fd.write('\n')
                                    output_line = []
                                    target_line = []
                            output_line.append(input_data[i][j]) #+'|'+subroot_data[i][j]
                            target_line.append(target_data[i][j])
                        if len(output_line)>1:
                            f.write(' '.join(output_line))
                            f.write('\n')
                            fo.write(' '.join(target_line))
                            fo.write('\n')
                            fd.write(str(i))
                            fd.write('\n')
                        
if __name__ == '__main__':
    # train with golden subroot data
    subroot_decomposition_g(os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_train.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_train.input'),
            os.path.join(os.path.dirname(__file__), '../ptb-sd/train_pro_wsd.conll'), # use golden subroot data
            os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_train_wsd_40.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_train_wsd_40.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_train_wsd_40_map.input'),
            40)

    # dev with predict subroot data
    subroot_decomposition(os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_dev.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_dev.input'),
            os.path.join(os.path.dirname(__file__), '../../subroot/result/dev_predicate_96.35.pred'), # use predict subroot data
            os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_dev_wsd_40.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_dev_wsd_40.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_dev_wsd_40_map.input'),
            40)

    # test with predict subroot data
    subroot_decomposition(os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_test.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_test.input'),
            os.path.join(os.path.dirname(__file__), '../../subroot/result/test_predicate_95.53.pred'), # use predict subroot data
            os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_test_wsd_40.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_test_wsd_40.input'),
            os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_test_wsd_40_map.input'),
            40)