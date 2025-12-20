'''Import this module to use the matrix multiplication function'''
print(__name__)
def MatrixMultiplication(A, B):
    m, p, n=len(A), len(B), len(B[0])
    C=[[0 for j in range(1, n+1)] for i in range(1, m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            s=0
            for k in range(1, p+1):
                s=s+A[i-1][k-1]*B[k-1][j-1]
            C[i-1][j-1]=s
    return C


