def project_rank(kernel_matrix, dim):
    D, U = kernel_matrix.symeig(eigenvectors=True)
    D = D[-dim:].clamp(min=0)
    return U[:, -dim:].mm(D.diag()).mm(U[:, -dim:].transpose(0, 1))
