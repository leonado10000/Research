import random
import sys
sys.path.append("")
from src.core.utility import is_scalar, shape_of, flatten, unflatten, elementwise_op, elementwise_map, zeros_like, add_inplace, sub_inplace, mul_inplace, scale, sum_all

class Tensor:
    def __init__(self, data, parents=None, op=None):
        # store data as nested lists or scalar
        self.data = data
        self.shape = shape_of(data)
        self.parents = parents if parents else []
        self.op = op  # string name for debugging
        self.grad = None  # same structure as data (nested lists) or scalar
        self._backward = lambda: None  # function to compute grads for parents

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self.op})"

    # --- creation helpers ---
    @staticmethod
    def from_shape(shape, random_init=False, value=0.0, scale=0.1):
        if len(shape) == 0:
            return Tensor((random.uniform(-scale,scale)) if random_init else value)
        if len(shape) == 1:
            return Tensor([((random.uniform(-scale,scale)) if random_init else value) for _ in range(shape[0])])
        return Tensor([Tensor.from_shape(shape[1:], random_init, value, scale).data for _ in range(shape[0])])

    # --- basic ops: add, sub, mul (elementwise), scalar mul ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(add_inplace(self.data, other.data), parents=[self, other], op="add")
        def _backward():
            # d(out)/dself = 1 * upstream
            if out.grad is None:
                return
            g = out.grad
            self.grad = add_inplace(self.grad if self.grad is not None else zeros_like(self.data), g)
            other.grad = add_inplace(other.grad if other.grad is not None else zeros_like(other.data), g)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(sub_inplace(self.data, other.data), parents=[self, other], op="sub")
        def _backward():
            if out.grad is None:
                return
            g = out.grad
            self.grad = add_inplace(self.grad if self.grad is not None else zeros_like(self.data), g)
            other.grad = add_inplace(other.grad if other.grad is not None else zeros_like(other.data), scale(g, -1.0))
        out._backward = _backward
        return out

    def __mul__(self, other):
        # support scalar or elementwise or self*self (square)
        if isinstance(other, Tensor):
            # elementwise
            out = Tensor(mul_inplace(self.data, other.data), parents=[self, other], op="mul")
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                # d/dself = other * g
                self.grad = add_inplace(self.grad if self.grad else zeros_like(self.data),
                                        mul_inplace(other.data, g))
                other.grad = add_inplace(other.grad if other.grad else zeros_like(other.data),
                                         mul_inplace(self.data, g))
            out._backward = _backward
            return out
        else:
            # scalar multiply
            out = Tensor(scale(self.data, float(other)), parents=[self], op=f"mul_{other}")
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                self.grad = add_inplace(self.grad if self.grad else zeros_like(self.data),
                                        scale(g, float(other)))
            out._backward = _backward
            return out

    # useful elementwise square
    def square(self):
        out = Tensor(elementwise_map(self.data, lambda a: a*a), parents=[self], op="square")
        def _backward():
            if out.grad is None:
                return
            g = out.grad
            # d(x^2)/dx = 2*x * g
            self.grad = add_inplace(self.grad if self.grad else zeros_like(self.data),
                                    mul_inplace(scale(self.data, 2.0), g))
        out._backward = _backward
        return out

    # sum -> returns scalar Tensor
    def sum(self):
        s = sum_all(self.data)
        out = Tensor(s, parents=[self], op="sum")
        def _backward():
            if out.grad is None:
                return
            # out.grad is scalar upstream (usually 1.0)
            g = out.grad
            # d(sum)/dself = 1 for every element
            ones = elementwise_map(self.data, lambda a: 1.0)
            self.grad = add_inplace(self.grad if self.grad else zeros_like(self.data),
                                    scale(ones, g if is_scalar(g) else sum_all(g)))
        out._backward = _backward
        return out

    # transpose for 2D tensors only
    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError("transpose supports 2D tensors only")
        rows, cols = self.shape
        trans = [[self.data[r][c] for r in range(rows)] for c in range(cols)]
        out = Tensor(trans, parents=[self], op="transpose")
        def _backward():
            if out.grad is None:
                return
            g = out.grad  # gradient in transposed space
            # transpose back
            back = [[g[c][r] for c in range(len(g))] for r in range(len(g[0]))]
            self.grad = add_inplace(self.grad if self.grad else zeros_like(self.data), back)
        out._backward = _backward
        return out

    # matrix multiplication (2D @ 2D) -> out shape (m x p) where self is (m x n), other is (n x p)
    def matmul(self, other):
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul supports 2D tensors only")
        m, n = self.shape
        n2, p = other.shape
        assert n == n2, "shapes incompatible for matmul"
        # compute product
        res = [[sum(self.data[i][k] * other.data[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
        out = Tensor(res, parents=[self, other], op="matmul")
        def _backward():
            if out.grad is None:
                return
            g = out.grad  # m x p
            # d/dself = g @ other.T  => (m x p) @ (p x n) = (m x n)
            other_T = [[other.data[r][c] for r in range(other.shape[0])] for c in range(other.shape[1])]
            # compute g @ other.T
            grad_self = [[sum(g[i][k] * other.data[j][k] for k in range(p)) for j in range(n)] for i in range(m)]
            # d/dother = self.T @ g => (n x m) @ (m x p) = (n x p)
            self_T = [[self.data[r][c] for r in range(self.shape[0])] for c in range(self.shape[1])]
            grad_other = [[sum(self.data[k][i] * g[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
            self.grad = add_inplace(self.grad if self.grad else zeros_like(self.data), grad_self)
            other.grad = add_inplace(other.grad if other.grad else zeros_like(other.data), grad_other)
        out._backward = _backward
        return out

    # convenience: dot alias (keeps your old naming)
    def dot(self, other):
        return self.matmul(other)

    # set scalar grad (starting point)
    def backward(self, retain_graph=False):
        # If tensor is not scalar but user calls backward, accept it as overall upstream grad of 1s matching shape
        # Set initial gradient to 1 (scalar) or ones of shape
        if self.grad is None:
            if is_scalar(self.data):
                self.grad = 1.0
            else:
                # gradient of loss wrt itself is ones
                self.grad = elementwise_map(self.data, lambda a: 1.0)

        # build topological order
        topo = []
        visited = set()
        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v.parents:
                build(p)
            topo.append(v)
        build(self)

        # traverse in reverse topological order to call backward functions
        for node in reversed(topo):
            node._backward()
            if not retain_graph:
                # optionally clear gradients on non-leaf nodes?
                pass

    # helpers for printing / item
    def item(self):
        return sum([el for el in flatten(self.data)])
    
    def relu(self):
        out_data = elementwise_map(self.data, lambda a: max(0, a))
        out = Tensor(out_data, parents=[self], op="relu")
        
        def _backward():
            if out.grad is None: 
                return
            
            # mask: gradient flows only where x > 0
            mask = elementwise_map(self.data, lambda a: 1.0 if a > 0 else 0.0)
            self.grad = add_inplace(
                self.grad if self.grad else zeros_like(self.data),
                mul_inplace(mask, out.grad)
            )
        
        out._backward = _backward
        return out
    
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

        _t1 = Tensor(reshape(flat, shape), parents=[self], op="view")
        return _t1