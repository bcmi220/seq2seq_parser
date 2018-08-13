import os
import tqdm

# def is_scientific_notation(s):
#     s = str(s)
#     if s.count(',')>=1:
#         sl = s.split(',')
#         for item in sl:
#             if not item.isdigit():
#                 return False
#         return True   
#     return False

# def is_float(s):
#     s = str(s)
#     if s.count('.')==1:
#         sl = s.split('.')
#         left = sl[0]
#         right = sl[1]
#         if left.startswith('-') and left.count('-')==1 and right.isdigit():
#             lleft = left.split('-')[1]
#             if lleft.isdigit() or is_scientific_notation(lleft):
#                 return True
#         elif (left.isdigit() or is_scientific_notation(left)) and right.isdigit():
#             return True
#     return False

# def is_fraction(s):
#     s = str(s)
#     if s.count('\/')==1:
#         sl = s.split('\/')
#         if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
#             return True  
#     if s.count('/')==1:
#         sl = s.split('/')
#         if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
#             return True    
#     if s[-1]=='%' and len(s)>1:
#         return True
#     return False

# def is_number(s):
#     s = str(s)
#     if s.isdigit() or is_float(s) or is_fraction(s) or is_scientific_notation(s):
#         return True
#     else:
#         return False

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
            if dep_ind > head_ind:
                tag = 'L' + str(abs(dep_ind - head_ind))
            else:
                tag = 'R' + str(abs(dep_ind - head_ind))
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
        if len(src_line) >= 1:
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

    make_input(train_file, os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_train.input'),
               os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_train.input'))
    make_input(dev_file, os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_dev.input'),
               os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_dev.input'))
    make_input(test_file, os.path.join(os.path.dirname(__file__), '../input/dp/src_ptb_sd_test.input'),
               os.path.join(os.path.dirname(__file__), '../input/dp/tgt_ptb_sd_test.input'))
