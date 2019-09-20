# A GCN based on a seq2seq-model

## Requirements
* torch 0.4.1
* python 3.6

# Model
## GCN

We use the simple 1-layer linear GCN, which references the source code from https://github.com/tkipf/pygcn.git

Preprocessing: loc_set and adjacency matrix are generated in train.py.
Our adjacency matrix is a directed graph, the normalize function in train.py initializes the adjacency matrix A to D^-1A.

## Seq2seq

The encoder is set as Bi-LSTM, the decoder is set as 2-layer LSTM.

# Data
You can directly use two datasets. The original dataset can be found at [1] and [2], the processing method can be found at [3] and https://github.com/vonfeng/DeepMove.git.

# Results

The results and pretrained model will come soon.

# References
[1]:Yang, D.; Zhang, D.; Zheng, V. W.; Yu, Z. (2014): Modeling user activity preference by leveraging user spatial temporal characteristics in lbsns. IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 45, no. 1, pp. 129–142.

[2]:Cho, E.; Myers, S. A.; Leskovec, J. (2011): Friendship and mobility: user movement in location-based social networks,” Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, ACM, pp. 1082–1090.

[3]:Feng, J.; Li, Y.; Zhang, C.; Sun, F.; Meng, F. et al. (2018): Deepmove: Predicting human mobility with attentional recurrent networks, Proceedings of the 2018 World Wide Web Conference, International World Wide Web Conferences Steering Committee, 2018: 1459-1468.