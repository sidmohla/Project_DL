
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def is_semieulerian(G):
    # Check if all non-zero degree vertices are connected
    if nx.is_connected(G) == False:
        return 0
    else:
        #Count vertices with odd degree
        odd = 0
        for i in range(G.number_of_nodes()):
            #print(len(G.adj[i]))
            if len(G.adj[i]) % 2 !=0:
                odd +=1

        '''If odd count is 2, then semi-eulerian.
        If odd count is 0, then eulerian
        If count is more than 2, then graph is not Eulerian
        Note that odd count can never be 1 for undirected graph'''
        if odd == 0:
            return 1
        elif odd == 2:
            return 1
        elif odd > 2:
            return 0

G = nx.complete_graph(10)
e = np.array(G.edges)
X,y = [],[]
count = 0;
while(len(y) < 10000):
    idx= np.random.choice(len(e),25);
    edg = e[idx];
    G = nx.empty_graph(10);
    G.add_edges_from(edg)
    
    #print(mat)

    label = int(is_semieulerian(G) > 0)
    if(label == 0):
        if(count > 4550):
            mat = np.zeros((10,10),dtype=int);
            for i in edg:
                mat[tuple(i)]=True;
                mat[tuple(i[::-1])]=True;
            X.append(mat.reshape(1,-1)[0]);
            y.append(label)
            print(i,label)
    else:
        mat = np.zeros((10,10),dtype=int);
        for i in edg:
            mat[tuple(i)]=True;
            mat[tuple(i[::-1])]=True;
        X.append(mat.reshape(1,-1)[0]);
        y.append(label)
        print(i,label)

    count += label
    #print(i,label)
    #nx.draw(G);
    #plt.show();

y = np.asarray(y)
X = np.asarray(X)
perm = np.random.permutation(len(y))
#print(perm)
X = X[perm]
y = y[perm]
#print(y)
import pickle
pickle_out = open('eulerian.pickle','wb')
pickle.dump((X,y),pickle_out);
pickle_out.close();
pickle_in  = open('eulerian.pickle','rb')
X,y = pickle.load(pickle_in);
print(type(X),type(X[0][0]))
print("X shape",len(X),X[0].shape)
print("y shape",len(y))

print("DONE")
print("# Connected :",count)
