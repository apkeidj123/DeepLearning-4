import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#import hw4_1_preprocess as dp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import sklearn.metrics as metrics

tf.reset_default_graph()

#"""
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
#"""
  
###----------Hyper parameter------------

training_epochs = 50
vocab_size = 88588 # 88588
batch_size = 25
LR = 1e-4         # learning rate
embedding_size = 64
n_input = 120        ## word 
n_hidden_units = 128
n_examples = 25000
n_classes = 2
calc_iter = 500
cell_index = 1     # select which cell to use => 1: RNN, 2:LSTM, 3:GRU
alpha = 0.01

file = 'test'
if cell_index == 1:   ## RNN
    file = 'rnn'
elif cell_index == 2: ## LSTM
    file = 'lstm'
elif cell_index == 3: ## GRU
    file = 'gru'  

def get_batch(inputs, targets, n_examples, batch_size):          
    #indices = np.random.choice(n_examples, n_examples, replace = False) # random indices
    
    for batch_i in range(n_examples // batch_size): # 25000 // 8
        start = batch_i * batch_size
        end = start + batch_size       
        batch_xs = inputs[start:end]
        batch_ys = targets[start:end]

        yield batch_xs, batch_ys
        
t00 = time.time()

x_train = np.load("train_data.npy")
y_train = np.load("train_labels.npy")
x_test = np.load("test_data.npy")
y_test = np.load("test_labels.npy")
t01 = time.time()

print('Load Time: ', t01 - t00)

"""
with tf.device("/cpu:0"):
    embedding = tf.get_variable("embedding", [vocab_size, hidden_units_size])
    inputs = tf.nn.embedding_lookup(embedding, train_x)
"""

### Embedding Layer

inputs = tf.placeholder(tf.int32, [None, n_input])
embedding = tf.Variable(tf.random_normal(shape=(vocab_size, embedding_size)))
embedded_vector = tf.nn.embedding_lookup(embedding, inputs)

# tf Graph input
xs = tf.placeholder(tf.float32, [None, n_input, embedding_size])
#y = tf.placeholder(tf.float32, [None, n_classes])
ys = tf.placeholder(tf.int32, [None])
y_oh = tf.one_hot(indices = ys, depth = n_classes)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Define weights
weights = {
    # (28, 128)
    #'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 2)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
   # 'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (2, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    #'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(X, weights, biases):    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    X = tf.unstack(X, n_input, 1)
       
    ### cell
    if cell_index == 0:   ## RNN   # ~100 s
        cell = tf.contrib.rnn.BasicRNNCell(num_units = n_hidden_units)
    elif cell_index == 1:   ## RNN   # ~100 s
        cell = tf.contrib.rnn.BasicRNNCell(num_units = n_hidden_units)    
    elif cell_index == 2: ## LSTM  # ~250 s
        cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden_units)
    elif cell_index == 3: ## GRU   # ~260 s
        cell = tf.contrib.rnn.GRUCell(num_units = n_hidden_units)  
        
    # lstm cell is divided into two parts (c_state, h_state)
    #init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    #outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
      
    #init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    #cell_dr = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = 1.0, output_keep_prob = 0.9)
    
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, X, dtype = tf.float32 )   
    
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 2)

    return results

y_pred = RNN(xs, weights, biases)
prediction = tf.nn.softmax(y_pred)

## L2 norm
#tv = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
#regularization_cost = alpha * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数

### the error between prediction and real data
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_pred, labels = y_oh))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_oh))
#cross_entropy = tf.losses.softmax_cross_entropy(logits = y_pred, onehot_labels = y_oh)

#total_loss = cross_entropy + regularization_cost

#train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(LR).minimize(total_loss)

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_oh,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###--------------Start Training--------------------
train_loss = np.zeros([training_epochs,1])
train_accuracy = np.zeros([training_epochs,1])
test_loss = np.zeros([training_epochs,1])
test_accuracy = np.zeros([training_epochs,1])


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('Start Training')
    tStart3 = time.time()
    
    train_weightvec = sess.run(embedded_vector, feed_dict={inputs: x_train})
    test_weightvec = sess.run(embedded_vector, feed_dict={inputs: x_test})
    #print(train_weightvec.shape)
    #print(test_weightvec.shape)

    for epoch in range(training_epochs):    
        print('Epoch: ', epoch + 1)
    ###-----------Accuracy / Loss--------------
        accuracy_train = 0.0
        accuracy_test = 0.0
        loss_train = 0.0
        loss_test = 0.0 
        L2_loss1 = 0.0
        L2_loss2 = 0.0
            
        tStart = time.time()

        for batch_xs,batch_ys in get_batch(train_weightvec, y_train, y_train.shape[0], batch_size): # 25000
            sess.run(train_step,feed_dict={xs: batch_xs, ys: batch_ys})
        
        tEnd = time.time()
        
        print("Training Time = ", tEnd - tStart)
        
        tStart2 = time.time()
        
        """
        ### L2 norm
        for i in range(calc_iter):
            random_index_train = np.random.choice(y_train.shape[0], 2, replace = False)
            Loss, Acc, l2_loss1, l2_loss2 = sess.run([total_loss, accuracy, cross_entropy, regularization_cost],
                                 feed_dict={xs: train_weightvec[random_index_train],
                                            ys:y_train[random_index_train]})
            L2_loss1 += l2_loss1
            L2_loss2 += l2_loss2
            accuracy_train += Acc
            loss_train += Loss
            
            random_index_test = np.random.choice(y_test.shape[0], 2, replace = False)
            Loss_2, Acc_2 = sess.run([total_loss, accuracy], 
                                     feed_dict={xs: test_weightvec[random_index_test],
                                                ys:y_test[random_index_test]})
            accuracy_test += Acc_2
            loss_test += Loss_2
            

        tEnd2 = time.time()
        print("Calculate Time = ", tEnd2 - tStart2)
        
        print('Train Accuracy: ', accuracy_train/calc_iter)
        print('Train Loss: ', loss_train/calc_iter)
        print('L2_loss1: ', L2_loss1/calc_iter)
        print('L2_loss2: ', L2_loss2/calc_iter)
        print('Test Accuracy: ', accuracy_test/calc_iter)
        print('Test Loss: ', loss_test/calc_iter)
        """

        #"""
        for i in range(calc_iter):
            random_index_train = np.random.choice(y_train.shape[0], 2, replace = False)
            
            Loss, Acc = sess.run([cross_entropy, accuracy],
                                 feed_dict={xs: train_weightvec[random_index_train],
                                            ys: y_train[random_index_train]})
            accuracy_train += Acc
            loss_train += Loss
            
            random_index_test = np.random.choice(y_test.shape[0], 2, replace = False)
            Loss_2, Acc_2 = sess.run([cross_entropy, accuracy], 
                                     feed_dict={xs: test_weightvec[random_index_test],
                                                ys: y_test[random_index_test]})
            accuracy_test += Acc_2
            loss_test += Loss_2
                       
        tEnd2 = time.time()
        print("Calculate Time = ", tEnd2 - tStart2)
        
        print('Train Accuracy: ', accuracy_train/calc_iter)
        print('Train Loss: ', loss_train/calc_iter)
        print('Test Accuracy: ', accuracy_test/calc_iter)
        print('Test Loss: ', loss_test/calc_iter)
        #"""
        
        train_loss[epoch]= loss_train/calc_iter    
        train_accuracy[epoch] = accuracy_train/calc_iter
        test_loss[epoch] = loss_test/calc_iter
        test_accuracy[epoch] = accuracy_test/calc_iter
        
        
    tEnd3 = time.time()
    print("Total Training Time = ", tEnd3 - tStart3)
        
    
    # plot ROC
    
    #pred = sess.run(prediction, feed_dict={xs : train_weightvec}) 
    pred = sess.run(prediction, feed_dict={xs : test_weightvec}) 
    probs = pred[:, 1] ## select bigger one
    
    """
    preds = np.zeros([n_examples, 1])
    for i in range(pred.shape[0]):
        if pred[i][1] > 0.5:
            preds[i] = 1
    
    preds = preds.flatten() 
    """
   
    #fpr, tpr, threshold = metrics.roc_curve(y_train, probs)
    #auc = metrics.roc_auc_score(y_train, probs)
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc = metrics.roc_auc_score(y_test, probs)
    
    np.save("result/" + file + "/fpr", fpr)
    np.save("result/" + file + "/tpr", tpr)
    np.save("result/" + file + "/threshold", threshold)
    
    #auc = train_accuracy[training_epochs-1]
    
    plt.figure()
    
    plt.plot([0, 1], [0, 1], label = 'Random', linestyle='--')
    plt.plot(fpr, tpr, label = "AUC =" + str(auc), marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (' + file + ', test)')
    plt.legend(loc='best')   
    plt.savefig('output/' + file + '/ROC.jpg')
    plt.show()
    plt.clf()
    
    # plot PRC
    
    #precision, recall, thresholds = metrics.precision_recall_curve(y_train, probs)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, probs)
    aucs = metrics.auc(recall, precision)
    
    np.save("result/" + file + "/precision", precision)
    np.save("result/" + file + "/recall", recall)
    np.save("result/" + file + "/thresholds", thresholds)
    
    plt.figure()

    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.plot(recall, precision, label = "AUPRC =" + str(aucs), marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precission')
    plt.title('Precision-Recall Curve (' + file + ', test)')
    plt.legend(loc='best')   
    plt.savefig('output/' + file + '/PRC.jpg')
    plt.show()
    plt.clf()
    
    

#"""
###------3. Plot------
### Loss
plt.figure()
y_range = range(0, training_epochs)       

plt.plot(y_range, train_loss, color = 'blue', label = "training loss")   
plt.plot(y_range, test_loss, color = 'orange', label = "test loss")

plt.xlabel('epoch')
plt.ylabel('Cross entropy')
plt.legend(loc = 'best')       
plt.savefig('output/' + file + '/Loss.jpg')
plt.show()
plt.clf()

### Accuracy
plt.figure()

plt.plot(y_range, train_accuracy, color = 'blue', label = "training acc")   
plt.plot(y_range, test_accuracy, color = 'orange', label = "test acc")

plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(loc = 'best')       
plt.savefig('output/' + file + '/Accuracy.jpg')
plt.show()
plt.clf()

#"""




###----------Save result-------- -
 

np.save("result/" + file + "/Loss_train", train_loss)
np.save("result/" + file + "/Acc_train", train_accuracy)
np.save("result/" + file + "/Loss_test", test_loss)
np.save("result/" + file + "/Acc_test", test_accuracy)

