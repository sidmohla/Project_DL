import numpy as np
import planarity
import matplotlib.pyplot as plt
import networkx as nx

def isSafe(G, v, pos, path):
    if G[path[pos-1],v] == 0:
        return False

    for vertex in path:
        if vertex == v:
            return False

    return True

def hamCycleUtil(G,V, path, pos):

    if pos == V:
        if G[path[pos-1],path[0]] == 1:
            return True
        else:
            return False

    for v in range(1,V):
        if isSafe(G, v, pos, path) == True:
            path[pos] = v
            if hamCycleUtil(G,V,path, pos+1) == True:
                return True
            path[pos] = -1

    return False

def hamCycle(G):
    path = [-1] * G.number_of_nodes()
    path[0] = 0

    if hamCycleUtil(nx.adjacency_matrix(G,nodelist=sorted(G.nodes())).todense(),G.number_of_nodes() - 1,path,1) == False:
        return False

    return True

G = nx.complete_graph(10)
e = np.array(G.edges)
print(G.edges,len(e))
X,y = [],[]
count = 0;
for i in range(10000):
    idx= np.random.permutation(len(e))[:21]
    #print(idx)
    edg = e[idx];
    G = nx.empty_graph(10)
    G.add_edges_from(edg)
    mat = np.zeros((10,10),dtype=int)
    for i in edg:
        mat[tuple(i)]=True;
        mat[tuple(i[::-1])]=True;
    X.append(mat.reshape(1,-1)[0])

    ham = hamCycle(G)
    if(ham):
        label = 1
    else:
        label = 0

    y.append(int(label))
    count += label
    print(i,label)
    
y = np.asarray(y)
X = np.asarray(X)
import pickle
pickle_out = open('hamil.pickle','wb')
pickle.dump((X,y),pickle_out);
pickle_out.close();
pickle_in  = open('hamil.pickle','rb')
X,y = pickle.load(pickle_in);
print(type(X),type(X[0][0]))
print("X shape",len(X),X[0].shape)
print("y shape",len(y))

print("DONE")
print("# Ham cycles :",count)
