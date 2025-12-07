import random
import sys
sys.path.append("")
from src.core.utility import is_scalar, shape_of, flatten, unflatten, elementwise_op, elementwise_map, zeros_like, add_inplace, sub_inplace, mul_inplace, scale, sum_all

class Tensor:
    def __init__(self, data=None, origin=None, parent_tensor=None):
        self.data = data
        # To track the operation history for backpropagation
        self.origin = origin if origin else None
        self.parent_tensor = parent_tensor if parent_tensor else self.origin
        # Shape
        self.shape = []
        _tensor = self.data
        while type(_tensor) == list and len(_tensor) > 0:
            self.shape.append(len(_tensor))
            _tensor = _tensor[0]
        self.name = f"tensor {self.shape}"

    def transpose(self):
        transposed_data = [[self.data[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return Tensor(transposed_data, origin=self)

    def dot(self, other):
        result = Tensor([[0]*other.shape[1] for _ in range(self.shape[0])], origin=self, parent_tensor=self)
        other = other.transpose()
        for i in range(self.shape[0]):
            for j in range(other.shape[0]):
                result.data[i][j] = sum([self.data[i][k] * other.data[j][k] for k in range(self.shape[1])])
        return result
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, origin={self.origin.name if self.origin else 'Input Tensor'}, parent={self.parent_tensor})"
    
    def _tensor_creator_zero(self, shape, value):
        if len(shape) == 0:
            return value
        elif len(shape) == 1:
            return [value]*shape[0]
        return [self._tensor_creator_zero(shape[1:], value) for _ in range(shape[0])]
        
    def _tensor_creator_random(self, shape):
        if len(shape) == 0:
            return random.uniform(-.5,.5)
        return [self._tensor_creator_random(shape[1:]) for _ in range(shape[0])]
    
    def create_tensor(self, shape, type=0, origin=None, parent=None):
        """
        type = "random" for random
        type = integer or float for all same values
        """
        if not isinstance(type, str):
            return Tensor(self._tensor_creator_zero(shape, type), origin=origin, parent_tensor=parent)
        else:
            return Tensor(self._tensor_creator_random(shape), origin=origin, parent_tensor=parent)
        
    def backward(self):
        """
        """
        # perform gradient for this tensor
        layer = self.parent_tensor.origin if self.parent_tensor else None
        if not layer or not hasattr(layer, "weights"):
            return
        else:
            input_tensor = self.parent_tensor.parent_tensor
            gradients = input_tensor.transpose().dot(self)
            layer.weights -= (gradients*layer.lr)

        # backward flow
        dx = self.dot(layer.weights.transpose())
        dx.parent_tensor = input_tensor
        dx.origin = layer
        dx.backward()
    
    def __mul__(self, num):
        """
        multiply a num to all values in tensor
        [[1,2,3],[4,5,6]].__mul__(2) = [[2,4,6],[8,10,12]]
        """
        if isinstance(num, Tensor) and self == num:
            def pow(a):
                if isinstance(a, list):
                    return [pow(v) for v in a]
                return a * a
            return Tensor(pow(self.data), origin=self, parent_tensor=self.parent_tensor)
        def mul(a):
            if isinstance(a, list):
                return [mul(v) for v in a]
            return a * num
        return Tensor(mul(self.data), origin=self, parent_tensor=self.parent_tensor)
        
    def __sub_helper(self, matrixA, matrixB):
        if type(matrixA) in [int, float] and type(matrixB) in [int, float]:
            return matrixA - matrixB
        return [self.__sub_helper(x,y) for x, y in zip(matrixA, matrixB)]
    
    def __sub__(self, matrixB):
        """
        [1,2,3]   [1,1,1] = [0,1,2]
        [4,5,6] - [1,1,1] = [3,4,5]
        [7,8,9]   [1,1,1] = [6,7,8]
        """
        subtracted_tensor = Tensor(
            [self.__sub_helper(x,y) for x, y in zip(self.data, matrixB.data)],
            parent_tensor=self
        )
        return subtracted_tensor
    
    def view(self, shape):
        def flatten(x):
            if isinstance(x, list):
                out = []
                for item in x:
                    out.extend(flatten(item))
                return out
            else:
                return [x]

        flat = flatten(self.data)

        total = 1
        for s in shape:
            total *= s
        if total != len(flat):
            raise ValueError(f"Cannot reshape tensor of size {len(flat)} into shape {shape}")

        def reshape(lst, shape):
            if len(shape) == 1:
                return lst[:shape[0]]
            size = shape[0]
            chunk = len(lst) // size
            return [reshape(lst[i*chunk:(i+1)*chunk], shape[1:]) for i in range(size)]

        _t1 = Tensor(reshape(flat, shape), origin=self.origin, parent_tensor=self.parent_tensor)
        return _t1
    
    def item(self):
        return self

            
m1 = Tensor([
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
    ])
m2 = Tensor([
        [1,1,1,1],
        [2,2,2,2],
        [3,3,3,3],
    ])

# print((m1 * 2).data)

# t = Tensor().create_tensor([1,2])
# print(t.data)

# t = m1.dot(m2)