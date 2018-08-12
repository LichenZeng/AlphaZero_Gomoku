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

# List remove
availables = list(range(10))
print("show:", availables)
availables.remove(3)
print("show1:", availables)
availables.remove(6)
print("show2:", availables)
availables.remove(3)  # ValueError: list.remove(x): x not in list
print("show3:", availables)
