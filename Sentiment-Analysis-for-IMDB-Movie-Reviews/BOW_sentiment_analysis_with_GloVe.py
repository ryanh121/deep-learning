import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from BOW_model import BOW_model

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
LR = 0.001
batch_size = 200
no_of_epochs = [3,6,20]
no_of_hidden_units_list = [100,500,1000]

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]

    line = np.mean(glove_embeddings[line],axis=0)

    x_train.append(line)
x_train = np.asarray(x_train)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]
    
    line = np.mean(glove_embeddings[line],axis=0)

    x_test.append(line)
x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

for idx, no_of_hidden_units in enumerate(no_of_hidden_units_list):
    model = BOW_model(no_of_hidden_units)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    L_Y_train = len(y_train)
    L_Y_test = len(y_test)

    model.train()

    train_loss = []
    train_accu = []
    test_accu = []

    for epoch in range(no_of_epochs[idx]):

        # training
        model.train()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_train)

        for i in range(0, L_Y_train, batch_size):

            x_input = x_train[I_permutation[i:i+batch_size]]
            y_input = y_train[I_permutation[i:i+batch_size]]

            data = Variable(torch.FloatTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()

            optimizer.zero_grad()
            loss, pred = model(data,target)
            loss.backward()

            optimizer.step()   # update weights
            
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        train_loss.append(epoch_loss)
        train_accu.append(epoch_acc)

        print('no_of_hidden_units: {}'.format(no_of_hidden_units), epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

        # ## test
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):

            # x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
            # y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
            # target = Variable(torch.FloatTensor(y_input)).cuda()

            x_input = x_test[I_permutation[i:i+batch_size]]
            y_input = y_test[I_permutation[i:i+batch_size]]

            data = Variable(torch.FloatTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()

            with torch.no_grad():
                loss, pred = model(data,target)
            
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print('no_of_hidden_units: {}'.format(no_of_hidden_units), "  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

    torch.save(model,'BOW.model')
    data = [train_loss,train_accu,test_accu]
    print(train_loss)
    print(train_accu)
    print(test_accu)
    data = np.asarray(data)
    np.save('no_of_hidden_units_{}_data.npy'.format(no_of_hidden_units),data)
