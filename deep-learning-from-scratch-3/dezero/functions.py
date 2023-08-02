import numpy as np
from dezero.core import Function

#############################################################


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy*cos(x)
        return gx


def sin(x):
    f = Sin()
    return f(x)

#############################################################


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy*-sin(x)
        return gx


def cos(x):
    f = Cos()
    return f(x)

#############################################################


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # tanh'(x) = 1 -y^2
        gx = gy * (1 - y*y)
        return gx


def tanh(x):
    f = Tanh()
    return f(x)

#############################################################


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    # f=Reshape(shape)
    # return f(x)
    return Reshape(shape)(x)

#############################################################


# class Transpose(Function):
#     def forward(self, x):
#         y = np.transpose(x)
#         return y

#     def backward(self, gy):
#         gx = transpose(gy)
#         return gx


# def transpose(x):
#     f = Transpose()
#     return f(x)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)
