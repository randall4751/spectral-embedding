import numpy as np

def create_embedding(A, k=None):
    #
    #   assert that adjacency matrix is symmetric (i.e. undirected graph) and
    #   no self-loops
    #
    assert np.all(A == A.T) and np.all(np.diagonal(A) == 0)
    #
    #   if no embedding dimension is given, assume a complete embedding
    #
    if k is None:
        k = A.shape[0]
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
    #
    #   reference embedding for simple-8-node-graph
    #
    embedding_true = np.array([
        [-0.37175704, -0.52629103, -0.52469872, -0.09364766,  0.35355339,  0.0293175,  -0.22333003],
        [ 0.1820782,  -0.18943571, -0.27139624,  0.18357784, -0.35355339,  0.20061966,  0.73031271],
        [ 0.53492517, -0.34130226,  0.40829325,  0.35631998,  0.35355339, -0.20775103, -0.10266466],
        [ 0.38474139, -0.04361675,  0.05271512, -0.59756022, -0.35355339,  0.32312183, -0.36851978],
        [ 0.00918132,  0.40298566, -0.3337428,   0.48875778, -0.35355339, -0.24921224, -0.41850101],
        [-0.57600091, -0.1699332,   0.55242392, -0.0747754,  -0.35355339, -0.27452924,  0.05670809],
        [-0.23858739,  0.41025503,  0.22882148,  0.18052378,  0.35355339,  0.66277222,  0.02348926],
        [ 0.07541926,  0.45733826, -0.112416,   -0.4431961,   0.35355339, -0.48433869,  0.30250543]
    ])

    A = read_edge_list('graphs/simple-8-node-graph-edge-list.txt')
    embedding = create_embedding(A)
    assert np.allclose(embedding, embedding_true, rtol=1e-8, atol=1e-8)
    
    print("PASSED unit tests")
