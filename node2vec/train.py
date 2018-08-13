import networkx as nx
from node2vec import Node2Vec

# FILES
EMBEDDING_FILENAME = './node2vec_en.emb'
EMBEDDING_MODEL_FILENAME = './node2vec_en.model'

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
graph = nx.Graph()

raw_train_file = '../data/ptb-sd/train_pro.conll'

with open(raw_train_file, 'r') as f:
    data = f.readlines()

    # read data
    train_data = []
    sentence = []
    for line in data:
        if len(line.strip()) > 0:
            line = line.strip().split('\t')
            sentence.append(line)
        else:
            train_data.append(sentence)
            sentence = []
    if len(sentence)>0:
        train_data.append(sentence)
        sentence = []

for sentence in train_data:
    for line in sentence:
        head_idx = int(line[6])-1
        if head_idx == -1:
            is_number = False
            word = line[1].lower()
            for c in word:
                if c.isdigit():
                    is_number = True
                    break
            if is_number:
                word = 'number'
            graph.add_edge('<ROOT>', word, weight=1)
        else:
            hw = sentence[head_idx][1].lower()
            is_number = False
            for c in hw:
                if c.isdigit():
                    is_number = True
                    break
            if is_number:
                hw = 'number'
            w = line[1].lower()
            is_number = False
            for c in w:
                if c.isdigit():
                    is_number = True
                    break
            if is_number:
                w = 'number'
            graph.add_edge(hw, w, weight=0.5)

# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=100, walk_length=100, num_walks=18, workers=1) 

# Embed
model = node2vec.fit(window=16, min_count=1, batch_words=64)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('<ROOT>')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)