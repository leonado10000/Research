import numpy as np
def relu(x):
    if type(x) == int or type(x) == float or isinstance(x, np.int32) or isinstance(x, np.float32) or isinstance(x, np.int64) or isinstance(x, np.float64):
        return max(0, x)
    else:
        return [relu(i) for i in x]

# print(relu(np.array([[ -1, 2, -3], [4, -5, 6]]))) 