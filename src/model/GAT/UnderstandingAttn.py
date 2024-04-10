import numpy as np

A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 1, 1]
])

X = np.random.uniform(-1, 1, (4, 4))
W = np.random.uniform(-1, 1, (2, 4))
W_attn = np.random.uniform(-1, 1, (1, 4))

connections = np.where(A > 0) # -- For masking unconnected nodes

leaky_relu = lambda x : np.maximum(0.2 * x, x)
attn_score = W_attn @ np.concatenate([(X @ W.T)[connections[0]], (X @ W.T)[connections[1]]], axis = 1).T
e = leaky_relu(attn_score) # For calculate similarity

E = np.zeros(A.shape)
E[connections[0], connections[1]] = e[0]

def softmax2D(x, axis = 1) :
    e = np.exp(x - np.expand_dims(np.max(x, axis = axis), axis = axis))
    sum = np.expand_dims(np.sum(e, axis = axis), axis)
    return e / sum

W_alpha = softmax2D(E, 1)
H = A.T  @ W_alpha @ X @ W.T