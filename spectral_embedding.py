import numpy as np

#
#   A represents an adjacency matrix for an undirected, 8-node simple (i.e. no self loops) graph
#       undirected --> symmetric matrix
#
A = np.array([
    [0,1,0,1,0,0,0,0],
    [1,0,1,0,1,0,0,0],
    [0,1,0,1,0,0,0,0],
    [1,0,1,0,1,0,0,1],
    [0,1,0,1,0,1,1,0],
    [0,0,0,0,1,0,1,0],
    [0,0,0,0,1,1,0,1],
    [0,0,0,1,0,0,1,0]
])

def make_grid_adjacency_matrix(n):
    points = n*n
    g = np.zeros((points,points), dtype='float')

    for i in range(n):
        for j in range(n):
            p = i*n + j
            if i == 0:
                if j == 0:
                    g[p,p+1], g[p,p+n] = (1,1)
                elif j == n-1:
                    g[p,p-1], g[p,p+n] = (1,1)
                else:
                    g[p,p-1], g[p,p+1], g[p,p+n] = (1,1,1)
            elif i == n-1:
                if j == 0:
                    g[p,p+1], g[p,p-n] = (1,1)
                elif j == n-1:
                    g[p,p-1], g[p,p-n] = (1,1)
                else:
                    g[p,p-1], g[p,p+1], g[p,p-n] = (1,1,1)
            else:
                if j == 0:
                    g[p,p+1], g[p,p-n], g[p,p+n] = (1,1,1)
                elif j == n-1:
                    g[p,p-1], g[p,p-n], g[p,p+n] = (1,1,1)
                else:
                    g[p,p-1], g[p,p+1], g[p,p-n], g[p,p+n] = (1,1,1,1)
    return g

def compute_laplacian(A):
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    return L

def compute_eigen_system(L):
    values, vectors = np.linalg.eig(L)
    sorted_value_indexes = sorted(range(len(values)), key=lambda i: values[i])
    sorted_vectors = vectors[:, sorted_value_indexes]
    return sorted_vectors

def create_embedding(A, k):
    #
    #   assert that adjacency matrix is symmetric (i.e. undirected graph)
    #
    assert(np.all(A == A.T))
    #
    #   compute diagonal degree matrix but summing along an axis
    #
    D = np.diag(np.sum(A, axis=0))
    #
    #   compute the Laplacian for the graph
    #       - diagonal is node degree
    #       - edge = -1
    #       - no edge = 0
    L = D - A
    #
    #   compute eigen values/vectors
    #       - Note: eigenv ectors are columns in the returns matrix
    #
    values, vectors = np.linalg.eig(L)
    #
    #   sort the eigen values (ascending) and get the sorted indexes
    #
    sorted_value_indexes = sorted(range(len(values)), key=lambda i: values[i])
    #
    #   sort the eigen vectors according the ascending order of their associated values
    #
    sorted_vectors = vectors[:, sorted_value_indexes]
    #
    #   create the embedding by selecting the number of sorted eigenvectors (columns)
    #   correcsponding to the embedding dimension, ignoring eigenvector 0; e.g. for
    #   and embedding of dimension 4, select columns 1, 2, 3, and 4 from the sorted
    #   eigenvector matrix
    #
    embedding = vectors[:, sorted_value_indexes[1:k+1]]
    #
    #   the embedding for node j is embedding[j], i.e. the jth element from each
    #   eigenvector in the embedding matrix
    #
    return embedding

def read_edge_list(filename):
    edges = []
    with open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            n1,n2 = line[:-1].split(' ')
            edges.append((int(n1),int(n2)))
    print(f'{len(edges)} edges')
    
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    nodes = list(nodes)
    N = len(nodes)

    assert max(nodes) == N - 1

    A = np.zeros((N,N))
    A[[edge[0] for edge in edges], [edge[1] for edge in edges]] = 1

    return A


def is_simple_graph(A):
    return np.all(A == A.T) and np.all(np.diagonal(A) == 0)

def partition_graph(embedding):
    return np.array([int(embedding[:,0][i] > 0.0) for i in range(embedding.shape[0])])

def number_of_triangles(A):
    A_3 = np.dot(np.dot(A,A),A)
    return int(np.trace(A_3)/6)

if __name__ == '__main__':
    import sys

    A = read_edge_list('graphs/simple-8-node-graph-edge-list.txt')
    assert is_simple_graph(A)
    A_3 = np.dot(np.dot(A,A),A)
    print(A_3)
    sys.exit()
    
    D = np.diag(np.sum(A, axis=0))
    print(D)
    #
    #   compute the Laplacian for the graph
    #       - diagonal is node degree
    #       - edge = -1
    #       - no edge = 0
    L = D - A
    print(L)
    sys.exit()
    
    A_3 = np.dot(np.dot(A,A), A)
    print(A_3)
    print(f'number of triangles: {number_of_triangles(A)}')
    sys.exit()

    embedding = create_embedding(A,2)

    print(partition_graph(embedding))
    # print(group_B)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embedding)
    print()
    print(kmeans.labels_)
    
    print()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding)
    print(kmeans.labels_)

    sys.exit()

    import matplotlib.pyplot as plt
    plt.scatter(embedding[:,1], embedding[:,0])
    for i in range(A.shape[0]):
        plt.annotate(str(i), (embedding[i,1], embedding[i,0]))
    plt.show()

