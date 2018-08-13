# DNN Pytorch

This repository includes a name tagger implemented with bidirectional LSTMs CRF network. It has an interface for external features. 


## Model
 
   ![Alt Text](https://blender04.cs.rpi.edu/~zhangb8/public_misc/ijcnlp17_model.png)

## Requirements

Python3, Pytorch

## Data Format

* Label format

    The name tagger follows *BIO* or *BIOES* scheme:
    
    ![Alt Text](https://blender04.cs.rpi.edu/~zhangb8/public_misc/bio_scheme_example.png)

* Sentence format
    
    Document is segmented into sentences. Each sentence is tokenized into multiple tokens. 
    
    In the training file, sentences are separated by an empty line. Tokens are separated by linebreak. For each token, label should be always at the end. Token and label are separated by space.
    
    CRF style features can be added between token and labels.
    
    Example:
    ```
    George B-PER
    W. I-PER
    Bush I-PER
    went O
    to O
    Germany B-GPE
    yesterday O
    . O
    
    New B-ORG
    York I-ORG
    Times I-ORG
    ```
    
    A real example of a bio file: `example/seq_labeling/data/eng.train.bio`
    
    A real example of a bio file with features: `example/seq_labeling/data/eng.train.feat.bio` 
    

## Usage

Training and testing examples are provided in `example/seq_labeling/`.

## Citation

[1] Boliang Zhang, Di Lu, Xiaoman Pan, Ying Lin, Halidanmu Abudukelimu, Heng Ji, Kevin Knight. [Embracing Non-Traditional Linguistic Resources for Low-resource Language Name Tagging](http://aclweb.org/anthology/I17-1037), Proc. IJCNLP, 2017

[2] Boliang Zhang, Xiaoman Pan, Tianlu Wang, Ashish Vaswani, Heng Ji, Kevin Knight, and Daniel Marcu. [Name Tagging for Low-Resource Incident Languages Based on Expectation-Driven Learning](http://nlp.cs.rpi.edu/paper/expectation2016.pdf), Proc. NAACL, 2016


