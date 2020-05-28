from scipy import linalg
import scipy as sp

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

    A = sp.zeros((N,N), dtype='int32')
    A[[edge[0] for edge in edges], [edge[1] for edge in edges]] = 1

    return A

def is_simple_graph(A):
    return sp.all(A == A.T) and sp.all(sp.diagonal(A) == 0)

def compute_laplacian(A, normalization=None):
    '''
        Computes the laplacian matrix representation from an adjacency matrix

        The laplacian matrix has node degrees along the diagonal, -1 elements
        to indicate node-to-node connections and 0's elsewhere.

        The laplacian may be un-normalized or normalized. There are two
        versions of normalization, symmetric and un-symmetric (sometimes
        referred to as "random walk" normalization)
    '''
    assert is_simple_graph(A)

    D = sp.diag(sp.sum(A, axis=0))
    if normalization is None:
        #
        #   un-normalized
        #
        L = D - A
    elif normalization == 'rw':
        #   
        #   nonsymmetric normalization, i.e. "random walk"
        #   L_rw = I = D^-1 x A
        #
        L = sp.eye(A.shape[0], dtype='float') - sp.dot(sp.linalg.inv(D), A)
    elif normalization == 'sym':
        #
        #   symmetric normalization
        #   L_sym = I - D^-0.5 x A x D^-0.5
        #
        D_inv_sqrt = linalg.inv(sp.linalg.sqrtm(D))
        L = sp.eye(A.shape[0]) - sp.dot(sp.dot(D_inv_sqrt, A), D_inv_sqrt)
    else:
        raise ValueError('normalization type must be one of [None, "rw", "sym"]')
    return L
    
def create_embedding(A, normalization=None, k=None):
    #
    #   assert that adjacency matrix is symmetric (i.e. undirected graph) and
    #   no self-loops
    #
    assert is_simple_graph(A)
    #
    #   if no embedding dimension is given, assume a complete embedding
    #
    if k is None:
        k = A.shape[0]-1
    #
    #   eigenvector[0] is not used in the embeddings so the largest 
    #   embedding dimension allowed is N-1
    #
    assert k <= A.shape[0]-1
    #
    #   compute the Laplacian
    #
    L = compute_laplacian(A, normalization)
    #
    #   compute eigen values/vectors
    #       - Note: eigenv ectors are columns in the returns matrix
    #
    values, vectors = linalg.eig(L)
    values = sp.absolute(values).astype('float')
    vectors = vectors.astype('float')
    #
    #   sort the eigenvalues (ascending) and get the sorted indexes
    #
    sorted_value_indexes = sorted(range(len(values)), key=lambda i: values[i])
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

def partition_graph(embedding):
    '''
        Returns the vector of node cluster labels [0,1] that partitions the graph
        into two roughly equal numbers of nodes using the fewest edge cuts.
    '''
    return sp.array([int(embedding[:,0][i] > 0.0) for i in range(embedding.shape[0])])

if __name__ == '__main__':
    #
    #   reference embeddings for simple-8-node-graph
    #
    EMBEDDING_8_UNNORM = sp.array([
        [-0.37175704, -0.52629103, -0.52469872, -0.09364766,  0.35355339,  0.0293175,  -0.22333003],
        [ 0.1820782,  -0.18943571, -0.27139624,  0.18357784, -0.35355339,  0.20061966,  0.73031271],
        [ 0.53492517, -0.34130226,  0.40829325,  0.35631998,  0.35355339, -0.20775103, -0.10266466],
        [ 0.38474139, -0.04361675,  0.05271512, -0.59756022, -0.35355339,  0.32312183, -0.36851978],
        [ 0.00918132,  0.40298566, -0.3337428,   0.48875778, -0.35355339, -0.24921224, -0.41850101],
        [-0.57600091, -0.1699332,   0.55242392, -0.0747754,  -0.35355339, -0.27452924,  0.05670809],
        [-0.23858739,  0.41025503,  0.22882148,  0.18052378,  0.35355339,  0.66277222,  0.02348926],
        [ 0.07541926,  0.45733826, -0.112416,   -0.4431961,   0.35355339, -0.48433869,  0.30250543]
    ])

    EMBEDDING_8_SYMNORM = sp.array([
        [ 2.12088374e-01,  5.80134093e-01,  1.22211369e-01, -4.10426310e-01, -2.68854257e-01, -9.88028596e-02,  5.11980344e-01],
        [-3.09219152e-01,  2.56536657e-01,  4.80840398e-01,  2.16383938e-02,  3.46202793e-01, -3.78554979e-01, -4.02390759e-01],
        [-4.82761439e-01,  1.25595600e-01, -2.11426985e-01,  5.23616520e-01, -5.82584342e-01,  3.80315559e-04,  4.44719832e-02],
        [-4.40076396e-01, -7.37222168e-02, -4.31038310e-01, -2.56723438e-01,  4.85934035e-01,  3.27283223e-01,  2.63793520e-01],
        [ 1.30370623e-01, -3.61815685e-01,  5.87321441e-01,  1.04185994e-01, -1.03488101e-01,  5.74522298e-01,  1.38356138e-01],
        [ 4.72567483e-01,  3.78454230e-01, -3.33106242e-01,  1.37007815e-01,  5.12710149e-02,  3.85406094e-01, -5.11513240e-01],
        [ 4.33166598e-01, -2.63210337e-01, -1.72709105e-01,  4.40397380e-01,  2.60126636e-01, -4.41426113e-01,  3.47019038e-01],
        [ 6.87485937e-02, -4.82707243e-01, -1.93976338e-01, -5.17131633e-01, -3.88999025e-01, -2.57582857e-01, -3.21220481e-01]
    ])

    EMBEDDING_8_RWNORM = sp.array([
        [ 2.34519991e-01,  6.40305898e-01,  1.47661427e-01,  4.53133118e-01,  3.03485928e-01, -1.18469222e-01,  5.67143932e-01],
        [-2.41776671e-01,  2.00213587e-01,  4.10810374e-01, -1.68927608e-02, -2.76335837e-01, -3.20959318e-01, -3.15190415e-01],
        [-5.33820907e-01,  1.38622440e-01, -2.55455860e-01, -5.78101306e-01,  6.57628232e-01,  4.56016037e-04,  4.92636400e-02],
        [-3.97324597e-01, -6.64372840e-02, -4.25231761e-01,  2.31425122e-01, -4.47871400e-01,  3.20415958e-01,  2.38593438e-01],
        [ 1.17705598e-01, -3.26062515e-01,  5.79409589e-01, -9.39191864e-02,  9.53820010e-02,  5.62467304e-01,  1.25139035e-01],
        [ 5.22548783e-01,  4.17707696e-01, -4.02474365e-01, -1.51264129e-01, -5.78753400e-02,  4.62119826e-01, -5.66626499e-01],
        [ 3.91086061e-01, -2.37200951e-01, -1.70382528e-01, -3.96999270e-01, -2.39751225e-01, -4.32163829e-01,  3.13868458e-01],
        [ 6.20699213e-02, -4.35008056e-01, -1.91363268e-01,  4.66171894e-01,  3.58529193e-01, -2.52178089e-01, -2.90534426e-01]
    ])

    A = read_edge_list('graphs/simple-8-node-graph-edge-list.txt')

    embedding_8_unnorm = create_embedding(A)
    assert sp.allclose(embedding_8_unnorm, EMBEDDING_8_UNNORM, rtol=1e-8, atol=1e-8)
    
    embedding_8_symnorm = create_embedding(A, normalization='sym')
    assert sp.allclose(embedding_8_symnorm, EMBEDDING_8_SYMNORM, rtol=1e-8, atol=1e-8)
    
    embedding_8_rwnorm = create_embedding(A, normalization='rw')
    assert sp.allclose(embedding_8_rwnorm, EMBEDDING_8_RWNORM, rtol=1e-8, atol=1e-8)
    
    try:
        e = create_embedding(A, normalization='illegal_type')
        assert False, 'create_embedding failed to throw ValueError for illegal normalization type'
    except ValueError:
        pass
    print("PASSED unit tests")
