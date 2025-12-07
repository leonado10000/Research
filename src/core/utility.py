import random
import math
from typing import List, Callable, Any

# -----------------------
# Utility helpers (pure python, nested lists)
# -----------------------

def is_scalar(x):
    return isinstance(x, (int, float))

def shape_of(x):
    s = []
    while isinstance(x, list):
        s.append(len(x))
        if len(x) == 0:
            break
        x = x[0]
    return s

def flatten(x):
    if is_scalar(x):
        return [x]
    out = []
    for v in x:
        out.extend(flatten(v))
    return out

def unflatten(flat, shape):
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return flat[:shape[0]]
    size = shape[0]
    step = int(len(flat) / size)
    return [unflatten(flat[i*step:(i+1)*step], shape[1:]) for i in range(size)]

def elementwise_op(A, B, fn: Callable[[float,float], float]):
    # shapes assumed equal
    if is_scalar(A) and is_scalar(B):
        return fn(A, B)
    if is_scalar(A):
        return [elementwise_op(A, b, fn) for b in B]
    if is_scalar(B):
        return [elementwise_op(a, B, fn) for a in A]
    return [elementwise_op(a,b,fn) for a,b in zip(A,B)]

def elementwise_map(A, fn: Callable[[float], float]):
    if is_scalar(A):
        return fn(A)
    return [elementwise_map(a, fn) for a in A]

def zeros_like(x):
    if is_scalar(x):
        return 0.0
    return [zeros_like(v) for v in x]

def add_inplace(A, B):
    # return A + B (new)
    return elementwise_op(A, B, lambda a,b: a+b)

def sub_inplace(A, B):
    return elementwise_op(A, B, lambda a,b: a-b)

def mul_inplace(A, B):
    return elementwise_op(A, B, lambda a,b: a*b)

def scale(A, s):
    return elementwise_map(A, lambda a: a * s)

def sum_all(x):
    if is_scalar(x):
        return x
    return sum(sum_all(v) for v in x)