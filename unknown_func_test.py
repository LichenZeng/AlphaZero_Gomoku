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

# Big big bug at list(zip(xxx))
from operator import itemgetter
import copy


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board))
    print("act probs:", len(action_probs), action_probs)
    return zip(board, action_probs)


a = list(range(12))
print("list origin:", a)
a.remove(10)
print("list remove:", a)

action_probs = rollout_policy_fn(a)
print("action probs:", type(action_probs), action_probs)
act_probs = copy.deepcopy(action_probs)
# act_probs = action_probs
print("action probs:", list(act_probs))
print(type(action_probs))
# print("action probs:", list(action_probs))

max_action = max(action_probs, key=itemgetter(1))
# max action (5, 0.947674461950645) 5 0.947674461950645
print("max action", max_action, max_action[0], max_action[1])
"""
list origin: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
list remove: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
act probs: 11 [0.2683784  0.83308728 0.36062368 0.65347525 0.56862719 0.33665552
 0.21917294 0.20516729 0.79368059 0.51260938 0.11576489]
action probs: <class 'zip'> <zip object at 0x7ff6fb419948>
action probs: [(0, 0.2683783989971129), (1, 0.8330872809890206), (2, 0.36062368300716796), (3, 0.6534752476570113), (4, 0.5686271873920377), (5, 0.336655517905069), (6, 0.21917293578747066), (7, 0.20516729103062026), (8, 0.7936805943382976), (9, 0.5126093797909821), (11, 0.1157648903418973)]
<class 'zip'>
action probs: []
Traceback (most recent call last):
  File "/home/tensorflow01/workspace/python_study/AlphaZero_Gomoku/unknown_func_test.py", line 84, in <module>
    max_action = max(action_probs, key=itemgetter(0))[0]
ValueError: max() arg is an empty sequence
"""
