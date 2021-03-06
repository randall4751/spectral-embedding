import numpy as np

def compute_number_of_triangles(A):
    return int(np.trace(np.dot(np.dot(A,A),A)))//6

def compute_local_clustering_coefficients(A):
    '''
        The local clustering coefficient for a graph node C[i] is
            c[i] = (number of closed triplets (i.e. triangles))/(number of all triplets (i.e. open and closed))

        The number of all triplets for C[i] is the number of its neighbors taken 2 at a time, or
            k(k-1)/2
        for undirected graphs. The number of neighbors, k, for each node is computed either as diagonal(A^2) or
        as the marginal sum (either axis) of the adjacency matrix, A.

        The number of triangles is taken as the diagnonal of the 3rd power of the adjacency matrix divided by 2.

        In vector form, the formula for C, the vector of Local Clustering Coefficients, is
            C = ((diagnonal(A^3))/2)/(k*(k-1)/2), or simply as 
            C = diagonal(A^3)/(k*(k-1))
    '''
    A_2 = np.dot(A,A)
    A_3 = np.dot(A_2,A)
    neighbors = np.diagonal(A_2)
    triangles = np.diagonal(A_3)
    return triangles/(neighbors*(neighbors-1))

def compute_global_clustering_coefficient(A):
    '''
        The global clustering coefficient for a graph is defined as
            C = 3*total_triangles/total_triplets
    '''
    n_triangles = compute_number_of_triangles(A)
    k = np.sum(A, axis=0)                           # number of neighbors for each node
    n_triplets = 0.5*np.sum(k*(k-1))                # total number of triplets
    return 3.0*n_triangles/n_triplets

def compute_average_clustering_coefficient(A):
    return np.average(compute_local_clustering_coefficients(A))

if __name__ == '__main__':
    LOCAL_CLUSTERING_COEFFICIENTS_TRUE = np.array([
        0.00000000, 0.16666667, 1.00000000, 0.33333333, 0.33333333, 0.00000000, 0.33333333, 0.33333333
    ])

    edges = []
    with open('graphs/simple-8-node-graph-edge-list.txt', encoding='utf-8') as f:
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

    number_of_triangles = compute_number_of_triangles(A)
    assert number_of_triangles == 2

    local_clustering_coefficients = compute_local_clustering_coefficients(A)
    assert np.allclose(local_clustering_coefficients, LOCAL_CLUSTERING_COEFFICIENTS_TRUE, rtol=1.0e-8, atol=1.0e-8)

    global_clustering_coefficient = compute_global_clustering_coefficient(A)
    assert global_clustering_coefficient == 0.2857142857142857
    
    average_clustering_coefficient = compute_average_clustering_coefficient(A)
    assert average_clustering_coefficient == 0.3125

    print("PASSED all unit tests")
