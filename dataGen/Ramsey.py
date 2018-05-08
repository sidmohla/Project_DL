import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import approximation

G = nx.complete_graph(10)
e = np.array(G.edges)
print(G.edges,len(e))
X,y = [],[]
count = 0;
for i in range(10000):
    sample = np.random.normal(12,2);
    if sample < 0 : sample = 2;
    elif sample>45: sample = 42;
    idx= np.random.permutation(len(e))[:int(sample)]
    #print(idx)
    edg = e[idx];
    G = nx.empty_graph(10)
    G.add_edges_from(edg)
    mat = np.zeros((10,10),dtype=int)
    for i in edg:
        mat[tuple(i)]=True;
        mat[tuple(i[::-1])]=True;
    X.append(mat.reshape(1,-1)[0])

    label = nx.ramsey_r2(G)
    y.append(int(label))
    count += label
    print(i,label)

y = np.asarray(y)
X = np.asarray(X)
import pickle
pickle_out = open('hamil.pickle','wb')
pickle.dump((X,y),pickle_out);
pickle_out.close();
pickle_in  = open('components.pickle','rb')
X,y = pickle.load(pickle_in);
print(type(X),type(X[0][0]))
print("X shape",len(X),X[0].shape)
print("y shape",len(y))

print("DONE")
print("# Ham cycles :",count)
