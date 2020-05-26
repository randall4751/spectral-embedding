import numpy as np

def compute_laplacian(A):
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    return L

def create_embedding(A, k):
    #
    #   assert that adjacency matrix is symmetric (i.e. undirected graph) and
    #   no self-loops
    #
    assert np.all(A == A.T) and np.all(np.diagonal(A) == 0)
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
