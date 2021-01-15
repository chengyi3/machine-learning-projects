# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
from torch import nn
import random


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.insize = in_size
        self.outsize = out_size
        # self.lay1 = torch.nn.Linear(in_size, 80)
        # self.lay3 = torch.nn.Linear(80, 80)
        # self.lay4 = torch.nn.Linear(80, 15)
        #add one more layer for extra

        # self.lay2 = torch.nn.Linear(15, out_size)
        # loss_fn = torch.nn.cross_entropy()
        # self.optimizer =  torch.optim.SGD(self.get_parameters(), lrate)

        # self.optimizer =  torch.optim.Adam(self.parameters(), lrate)
        self.loss_fn = loss_fn
        # self.drop_out1 = torch.nn.Dropout(p = 0.8)
        # self.drop_out2 = torch.nn.Dropout(p = 0.5)
        # self.bn1 = torch.nn.BatchNorm1d(80)
        # self.bn2 = torch.nn.BatchNorm1d(80)
        # self.bn3 = torch.nn.BatchNorm1d(15)


        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.conv2d(1, 32, 5, 1, 2)
        #     torch.nn.relu()
        #     torch.nn.MaxPool2d(2,2)
        #
        # )
        # using algorithm in https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch
        self.conv1 = torch.nn.Conv2d(3, 30, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(30 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, out_size)
        self.optimizer =  torch.optim.SGD(self.parameters(), lrate)


    def set_parameters(self, params):
        """ Set the parameters of your network
        @param params: a list of tensors containing all parameters of the network
        """
        # self.lay1.weight = params[0]
        # self.lay1.bias = params[1]
        # self.lay2.weight = params[2]
        # self.lay2.bias = params[3]
        # self.lay3.weight = params[4]
        # self.lay3.bias = params[5]
        # self.lay4.weight = params[6]
        # self.lay4.bias = params[7]
        self.fc1.weight = params[0]
        self.fc1.bias = params[1]
        self.fc2.weight = params[2]
        self.fc2.bias = params[3]
        # self.lay3.weight = params[4]
        # self.lay3.bias = params[5]
        # self.lay4.weight = params[6]
        # self.lay4.bias = params[7]

        # self.w1 = torch.zeros(self.insize, 32)
        # self,bias1 = torch.zeros(32)
        # self.bias2 = torch.zeros(self.outsize)
        # self.w2 = torch.zeros(32, self.outsize)
        # params.append(self.w1, self.w2, self.bias1, self.bias2)
        # self.parameters = params


    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """

        params = []
        params.extend([self.fc1.weight, self.fc1.bias,self.fc2.weight, self.fc2.bias])

        return params


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        #using algorithm in https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch
        x = x.view(-1, 3, 32, 32)
        # print(x.size())
        # layer1 = self.lay1(x)
        # self.y2 = self.drop_out(torch.nn.functional.relu(self.bn1(layer1)))
        # # self.y2 = torch.nn.functional.relu(self.bn1(layer1))
        #
        #
        # layer3 = self.lay3(self.y2)
        # self.y3 = self.drop_out(torch.nn.functional.relu(self.bn2(layer3)))
        # # self.y3 = torch.nn.functional.relu(self.bn2(layer3))
        #
        # layer4 = self.lay4(self.y3)
        #
        # self.y4 = self.drop_out(torch.nn.functional.relu(self.bn3(layer4)))
        #
        # layer2 = self.lay2(self.y4)
        x = (torch.nn.functional.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = x.view(-1, 30 * 16 *16)
        # print(x.size())
        x = (torch.nn.functional.relu(self.fc1(x)))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())



        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        y1 = self(x)
        # y1 = self.forward(x)
        L = self.loss_fn(y1, y)
        # self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        return L.item()



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, out_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of epochs of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    # lrate = 0.03
    #using algorithm in https://discuss.pytorch.org/t/standardization-of-data/16965
    means = train_set.mean(dim=1, keepdim=True)
    stands = train_set.std(dim=1, keepdim=True)
    normalized_train_set = (train_set - means) / stands
    # normalized_train_set = train_set
    lrate = 0.2
    n_iter = 3500
    batch_size = 100
    insize = len(train_set[0])
    outsize = len(np.unique(train_labels))
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss()
    net = NeuralNet(lrate, loss_fn, insize, outsize)
    losses = []
    index_set = list(range(len(train_set)))
    num_batches = len(train_set) // batch_size
    # num_batches = insize // batch_size
    num = 0
    for n in range(n_iter):
        losse = 0.0
        if num < num_batches:
            start = num * batch_size
            end = (num + 1)* batch_size
            inputdata = normalized_train_set[index_set[start:end]]
            inputlabels = train_labels[index_set[start:end]]
            # inputdata = train_set[start : end]
            # inputlabels = train_labels[start:end]
            losse += net.step(inputdata, inputlabels)
            num += 1
        else:
            startr = num_batches * batch_size
            # print(startr)
            endr = len(train_set)
            # print(endr)
            if startr == endr:
                num = 0
                random.shuffle(index_set)
                losses.append(losse)
                # trainsetandlabel = zip(train_set, train_labels)
                # # trainsetandlabel = list(trainsetandlabel)
                # random.shuffle(trainsetandlabel)
                # train_set, train_labels = zip(*trainsetandlabel)
                # train_set = torch.tensor(train_set)
                # train_labels = torch.tensor(train_labels)
                continue
            inputdata = normalized_train_set[index_set[startr:endr]]
            inputlabels = train_labels[index_set[startr:endr]]
            # inputdata = train_set[startr:endr]
            # # print(inputdata)
            # inputlabels = train_labels[startr:endr]
            losse += net.step(inputdata, inputlabels)
            num = 0
            random.shuffle(index_set)
            # trainsetandlabel = zip(train_set, train_labels)
            # trainsetandlabel = list(trainsetandlabel)
            # random.shuffle(trainsetandlabel)
            # train_set, train_labels = zip(*trainsetandlabel)
            # train_set = list(train_set)
            # train_set = torch.FloatTensor((train_set))
            # train_labels = list(train_labels)
            # train_labels = torch.FloatTensor(torch.FloatTensor(train_labels))

            #reshuffle?
        # num += 1
        # print(losse)
        # print(losses)
        losses.append(losse)
        # print(len(losses))
        # print(n_iter)
    # print(len(losses))


    num_batch_dev_set = len(dev_set) // batch_size
    yhats = []
    for i in range(num_batch_dev_set):
        start = i * batch_size
        end = (i + 1)* batch_size
        inputdata = dev_set[start : end]


        outputs = net.forward(inputdata)
        # print(outputs)
        index = torch.argmax(outputs, dim = 1)
        # index = np.argmax(outputs)
        # np.append(yhats, index)
        yhats.extend(index)
    startr = num_batch_dev_set * batch_size
    endr = len(dev_set)
    if startr != endr:
        inputdata = dev_set[startr:endr]
        outputs = net.forward(inputdata)
        index = torch.argmax(outputs, dim = 1)
        # index = np.argmax(outputs)
        # np.append(yhats, index)
        yhats.extend(index)
    yhat = np.array(yhats)
    # print(len(yhat))

        # optimizer.zero_grad()
        # noutsize = forward(train_set)
        # loss =s

    # print(train_set)
    # loss_fn = torch.nn.cross_entropy()
    torch.save(net, "net.model")
    return losses, yhat, net

    # for e in range(n_iter):
    #     losse = 0.0
    #     for i in range(num_batches):
    #         start = i * batch_size
    #         end = (i+1) * batch_size
    #         inputdata = train_set[start:end]
    #         inputlabels = train_labels[start:end]
    #         losse += net.step(inputdata, inputlabels)
    #
    #     startr = num_batches * batch_size
    #     endr = len(train_set)
    #     inputdata = train_set[startr:endr]
    #     inputlabels = train_labels[startr:endr]
    #     losse += net.step(inputdata, inputlabels)
    #
    #
    #     print(losse)
    #     losses.append(losse)
    #batch_size?
