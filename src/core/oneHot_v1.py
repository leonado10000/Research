from core.tensor_v1 import Tensor

def oneHotEncoding(value, size):
    if type(value) in [int, float]:
        if size >= value:
            res = [0]*size
            res[value] = 1
            return Tensor(res)
        else:
            print("size issue error")
    else:
        print("type issue error")