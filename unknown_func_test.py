import numpy as np
import random

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
# availables.remove(3)  # ValueError: list.remove(x): x not in list
# print("show3:", availables)


# format and rjust / ljust
for x in range(10):
    print("{0:8}".format(x), end='')
print()
for _ in range(10):
    print("       X".center(8), end="")
print()

print("zPlayer", 3, "with X".rjust(10, ","))
print("xwPlayer", 3, "with 0".ljust(9, "."))

# move = np.random.choice(
#     acts,
#     p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
# )
p = np.random.dirichlet(np.ones(10))
a = np.random.choice([1, 3, 5, 7, 9, 0, 2, 4, 6, 8], p=p)
print(p)
print(a)

# random sample
sam = random.sample([[1, 2, 3], [5, 3, 1], [6, 8, 9], [3, 4, 3], [5, 1, 1], [6, 9, 9]], 6)
print(sam)

x = np.arange(6).reshape(2, 3)
x = np.array([1, 3, 5, 7, 9, 2]).reshape(2, 3)
print(x)
y = np.ascontiguousarray(x, dtype=np.float32)
print(y)
# array([[0., 1., 2.],
#        [3., 4., 5.]], dtype=float32)
flag = x.flags['C_CONTIGUOUS']
print(flag)
