from src.core.tensor import Tensor

def oneHotEncoding(value, size):
    if type(value) in [int, float]:
        if size >= value:
            res = [0]*size
            res[value] = 1
            return Tensor(res, parents=[], op="oneHot")
        else:
            print("size issue error")
    else:
        print("type issue error")