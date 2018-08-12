import numpy as np

# Do array mirror retation by axis
square_state = np.random.random((3, 2, 3))
print(square_state)
print("======")
print(square_state[::-1, :, :])
print("======")
print(square_state[:, ::-1, :])
print("======")
print(square_state[:, :, ::-1])
