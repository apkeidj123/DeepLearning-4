# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:37:25 2019

@author: Finlab-Yi Hsien
"""

import numpy as np
import tensorflow as tf

MaxLen = 120

imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def onehot(array, C):# binary onehot
    return np.eye(C)[array.reshape(-1)]

#print(train_data[0])
#print(decode_review(train_data[0]))


train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=MaxLen)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=MaxLen)
#"""
np.save("train_data",train_data)
np.save("train_labels",train_labels)
np.save("test_data",test_data)
np.save("test_labels",test_labels)
#"""

#train_label = onehot(train_labels, 2)
#print(train_label.shape)