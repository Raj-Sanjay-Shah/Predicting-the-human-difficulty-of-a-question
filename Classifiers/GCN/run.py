import os

os.system('python preprocess.py')
os.system('python GCN/prepare_data.py')
os.system('python GCN/remove_words.py mr')
os.system('python GCN/build_graph.py mr')
os.system('python GCN/gcn_train.py mr')
