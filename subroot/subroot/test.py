import subprocess
import os


dnn_tagger_script = '../../dnn_pytorch/seq_labeling/tag.py'
model_dir = 'model/tag_scheme=iobes,zeros=True,char_dim=25,char_lstm_dim=25,char_conv_channel=25,word_dim=100,word_lstm_dim=100,feat_dim=5,feat_column=1,crf=True,dropout=0.5,lr_method=sgd-init_lr=.005-lr_decay_epoch=100,num_epochs=100,batch_size=20/best_model.pth.tar'

input_file = 'data/eng.testb.bio'
output_file = 'result/eng.test.output'

cmd = [
    'python3',
    dnn_tagger_script,
    '--model', model_dir,
    '--input', input_file,
    '--output', output_file,
    '--batch_size', '50',
    '--gpu', '0'
]

# set OMP threads to 1
os.environ.update({'OMP_NUM_THREADS': '1'})
# set which gpu to use if gpu option is turned on
gpu_device = '0'
os.environ.update({'CUDA_VISIBLE_DEVICES': gpu_device})

print(' '.join(cmd))
subprocess.call(cmd)