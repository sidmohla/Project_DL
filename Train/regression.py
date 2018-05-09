import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelBinarizer as lb
#lb = lb()

def get_batch(x, y, batch_size=0, batch_idx=0, size=0):
    if (batch_size==0):
        batch_size=x.shape[0]
    if(size==0):
        size=x.shape[0]
    oX = torch.from_numpy(x).float()
    oY = torch.from_numpy(y).float()
    if ((batch_idx+1)*batch_size > size) :
        if(batch_size!=1):
            batch_size = size - (batch_idx)*batch_size
    start = batch_idx*batch_size
    #if(y.shape[0] == 1):
            #return Variable(oX[start:start+batch_size,:]),Variable(oY[start:start+batch_size]).view(-1,1);
    return Variable(oX[start:start+batch_size,:]),Variable(oY[start:start+batch_size].view(-1,1));

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

def fun(i):
    picFile = i;
    N = 10000
    D_in = 100
    H1 = 5
    H2 = 0
    D_out = 1
    print(picFile)
    stream = open(picFile,'rb');
    X,y=pickle.load(stream)
    #y = lb.fit_transform(y)
    #print(y)
    #D_out = y[0].shape[0]
#X = np.asarray(X)
#print(X.shape)
#print(X[-1000:].shape)
#print(X[:-1000].shape)
    X_test = X[-1000:];
    y_test = y[-1000:];
    X,y = X[:-1000],y[:-1000]
    print(len(X),X[0].shape)

#print(y)

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
#x = Variable(torch.randn(N, D_in))
#y = Variable(1/(1+np.exp(torch.randn(N, D_out))), requires_grad=False)

# Use the nn package to define our model and loss function.
    class Net(nn.Module):
        def __init__(self, D_in, N1, D_out):
            super(Net, self).__init__()                    # Inherited from the parent class nn.Module
            self.fc1 = nn.Linear(D_in, N1)  # 1st Full-Connected Layer: 5 (input data) -> 5 (hidden node)
            self.relu = nn.LeakyReLU(0.01)                          # Non-Linear ReLU Layer: max(0,x)
            self.fc2 = nn.Linear(N1, D_out) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
            #self.relu = nn.LeakyReLU(0.01)                          # Non-Linear ReLU Layer: max(0,x)
            #self.fc3 = nn.Linear(N2, D_out) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        def forward(self, x):                              # Forward pass: stacking each layer together
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            #out = self.relu(out)
            #out = self.fc3(out)
            return out

    model = Net(D_in, H1, D_out)

    loss_fn = torch.nn.MSELoss(size_average=True)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    length = X.shape[0]
    num_epochs = 2000
    batch_size = length
    err = []
    for j in range(num_epochs):
        err.append(0)
    err_test = []

    miss = []
    for j in range(num_epochs):
        miss.append(0)
    miss_test = []

    netOut = []
    netOutTest = []

    trueOut = []
    trueOutTest = []

    X_test,y_test = get_batch(X_test,y_test)

    i = []
    for j in range(num_epochs):
        i.append(j)

    for t in range(num_epochs):
#Batchify
        perm = torch.randperm(length)
        X = X[perm]
        y = y[perm]
#Test Error
        #print(X_test.shape)
        y_test_pred = model(X_test)
        netOutTest.append(np.sum(np.around(y_test_pred.data.numpy(),decimals=0)) )
        trueOutTest.append(np.sum(np.around(y_test.data.numpy(),decimals=0)) )
        miss_test.append((np.count_nonzero(np.around(y_test_pred.data.numpy(),decimals=0) != y_test.data.numpy()))/y_test.shape[0])
        #print(y_test_pred,y_test)
        err_test.append(loss_fn(y_test_pred,y_test).data[0])
        netOut.append(0)
        trueOut.append(0)

        for batch_idx in range(int(length/batch_size)):
# Forward pass: compute predicted y by passing x to the model.
            batch_X,batch_y = get_batch(X,y, batch_size, batch_idx)
            y_pred = model(batch_X)
# Compute and print loss.
            loss = loss_fn(y_pred, batch_y)
            netOut[t] += np.sum(np.around(y_pred.data.numpy(),decimals=0) )
            trueOut[t] += np.sum(np.around(batch_y.data.numpy(),decimals=0) )

#print(t, loss.data[0],err_test[-1])
            err[t] += loss.data[0]
            miss[t] += (np.count_nonzero(np.around(y_pred.data.numpy(),decimals=0) != batch_y.data.numpy()))/batch_y.shape[0]

            print("# : ", t)
            print("Loss : ",loss.data[0],err_test[-1])
            print("Miss : ",miss[t], miss_test[-1])

# Before the backward pass, use the optimizer object to zero all of the
# gradients for the variables it will update (which are the learnable
# weights of the model). This is because by default, gradients are
# accumulated in buffers( i.e, not overwritten) whenever .backward()
# is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()


# Backward pass: compute gradient of the loss with respect to model
# parameters
            loss.backward()

# Calling the step function on an Optimizer makes an update to its
# parameters
            optimizer.step()

        err[t] = err[t]/int(length/batch_size)
        miss[t] = miss[t]/int(length/batch_size)


    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(211)
    plt.plot(err,'r+',label='Train Error');
    mt, bt = np.polyfit(i, err, 1)
    plt.plot(i, err_test,'g+',label='Test Error');
    plt.yscale('log')
    plt.legend();

    plt.subplot(212)
    plt.yscale('linear')
    plt.plot(i, miss_test,'b-',label='Test Miss');
    plt.plot(i, miss,'y-',label='Train Miss');
    plt.legend();

    plt.savefig(fname = (picFile + ','+str(H1)+','+str(H2)+'Train'+str(miss[-1])+'Test'+str(miss_test[-1])+'.svg'),format = 'svg');

    plt.show();

arr = ['../dataGen/clique.pickle'
,'../dataGen/components.pickle'
,'../dataGen/conEdges.pickle'
,'../dataGen/conVert.pickle'
,'../dataGen/diameter.pickle'
,'../dataGen/domination.pickle'
,'../dataGen/idp.pickle']
for i in arr:
    fun(i);
