#!/usr/local/bin/python3

import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pickle
import sys
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N,D_in, H1, H2, D_out =  10000, 100, 20, 5, 1
fileName = 'connected.pickle'# sys.argv[1]
stream = open(fileName,'rb');
X,y=pickle.load(stream)
X = np.asarray(X)
X_test = X[-1000:];
y_test = y[-1000:];
X,y = X[:-1000],y[:-1000]
print(len(X),X[0].shape)

x = Variable(torch.from_numpy(X));
x = x.type(torch.FloatTensor);
X_test = Variable(torch.from_numpy(X_test));
X_test = X_test.type(torch.FloatTensor);

y_test = Variable(torch.from_numpy(y_test));
y = Variable(torch.from_numpy(y));
y_test = y_test.type(torch.FloatTensor).view(-1,1);
y = y.type(torch.FloatTensor).view(-1,1);


# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
#x = Variable(torch.randn(N, D_in))
#y = Variable(1/(1+np.exp(torch.randn(N, D_out))), requires_grad=False)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.LeakyReLU(0.01),
    #torch.nn.Linear(H1, H2),
    #torch.nn.LeakyReLU(0.01),
    torch.nn.Linear(H1, H2),
    torch.nn.LeakyReLU(0.01),
    torch.nn.Linear(H2, D_out),
    torch.nn.Sigmoid(),
)
loss_fn = torch.nn.BCELoss(size_average=True)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
err = []
err_test = []
for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model.
    #Test Error
    y_test_pred = model(X_test)
    err_test.append(loss_fn(y_test_pred,y_test).data[0])

    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0],err_test[-1])
    err.append(loss.data[0])

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


import matplotlib.pyplot as plt
plt.plot(err[100:],'ro',label='Train Error');
plt.plot(err_test[100:],'g+',label='Test Error');
plt.yscale('log')
plt.legend();

plt.show();
