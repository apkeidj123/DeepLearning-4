# DeepLearning-4
Recurrent Neural Network

1. Gated recurrent neuralnetwork
* Recurrent structures are useful while dealing with sequential data. Train a model to classify movie reviews in IMDb dataset into good reviews (label 1) and bad reviews (label 0). We treat each review as a sample in the dataset with 120 "time points" (words) and use them as inputs.
* Before feeding samples into the model, you have to further transform the words (now they are encoded as integer indices) to vector representations. We call this procedure Embedding.
* Gated models such as Gated Recurrent Unit (GRU) and Long Short-Term Memory (LSTM)
* Evaluate the performance by test accuracy. Also, you have to plot learning curve as in previous homeworks, receiver operating characteristic curve (ROC) and precision-recall curve (PRC) with their area-under-curve (AUROC and AUPRC).

RNN:

![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final1RNN/rnn/Accuracy.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final1RNN/rnn/Loss.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final1RNN/rnn/PRC.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final1RNN/rnn/ROC.jpg)

LSTM:

![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final2LSTM/lstm/Accuracy.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final2LSTM/rnn/Loss.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final2LSTM/rnn/PRC.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final2LSTM/rnn/ROC.jpg)

GRU:

![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final3GRU/rnn/Accuracy.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final3GRU/rnn/Loss.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final3GRU/rnn/PRC.jpg)
![image](https://github.com/apkeidj123/DeepLearning-4/blob/master/Final3GRU/rnn/ROC.jpg)


2. Sequence-to-sequence learning
* machine translation task (from English to French) using sequence-to-sequence learning. The contents of attached datasets (en.txt and fr.txt) are paired English/French sentences. We should use en.txt as input (source) and fr.txt as output (target) to train/test our model. Notice that each sentence in the dataset should be treated as a sample with variable sequence length (i.e. variable number of words).


