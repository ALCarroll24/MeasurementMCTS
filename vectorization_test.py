import numpy as np

# Original matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [0, 2]])

# Step 1: Flatten matrix B
B_flat = B.flatten()

# Step 2: Kronecker product of A and an identity matrix of the same size as B
A_kron = np.kron(A, np.eye(B.shape[0]))

print("A_kron:", A_kron)

# Step 3: Dot product
result_vector = np.dot(A_kron, B_flat)

print("C:", result_vector)

# Step 4: Reshape result to original C shape
C_vectorized = result_vector.reshape(A.shape[0], B.shape[1])

print("C reshaped:", C_vectorized)
