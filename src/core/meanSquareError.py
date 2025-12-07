import sys
sys.path.append("")
from src.core.tensor import Tensor

class MSE:
    def __init__(self):
        self.name = "MSE"

    def __call__(self, prediction: Tensor, target: Tensor):
        # prediction and target shapes must match
        diff = prediction - target   # elementwise Tensor
        sq = diff.square()           # squared elements
        n = 1
        for dim in sq.shape:
            n*=dim
        
        mse_tensor = sq * (1/n)
        # override backward on loss to insert factor 1 (sum already handles passing ones)
        # but we need to ensure gradient scaling flows correctly: diff.square._backward already handles 2*x
        return mse_tensor