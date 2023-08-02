import numpy as np
import weakref
import contextlib


class Config:  # 역전파 여부를 결정하는 코드
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data, name=None):

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

# @property 인스턴스 변수
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

# 특수 매서드

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+' '*9)
        return 'variable('+p+')'

    def __mul__(self, other):
        return mul(self, other)

    def set_creator(self, func):
        self.creator = func  # output이 어떤 함수로 생성된건지 저장해두는 변수
        self.generation = func.generation + 1  # func가 몇 세대의 함수인지 저장해두는 변수

    # 같은 변수를 이용하여 미분 계산을 두번 하게 될 경우 첫번째 grad값이 두번째 grad에 관여하게 되므로
    # grad를 외부에서 초기화 할 수 있는 함수
    def cleargrad(self):
        self.grad = None

    # 재귀 구조를 이용한 구현
    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()

    # 반복문을 이용한 구현(+입력값 여러개 가정)

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            # # np.ones_like(self.data): 모든 요소 1로 채워진 self.data와 같은 형상과 데이터 타입의 ndarray 인스턴스 생성
            self.grad = Variable(np.ones_like(self.data))
            # 역전파 인스턴스를 ndarray에서 Variable로 변경하여 고차 미분가능하게 변경

        # funcs = [self.creator]
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
                # x의 변수의 정보 중 x.generation을 이용하여 정렬

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # 약한 참조의 변수는 output이 아닌 output()처럼 ()필요
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)  # main backward
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    # 동일한 변수를 사용한 연산이 있을 때 grad 값을 합치기 위함

                    if x.creator is not None:
                        add_func(x.creator)
                        # funcs.append(x.creator)
                    # input의 이전 단계에서 사용된 함수 funcs에 추가
                    # 이전 단계에서 사용된 함수가 없을 경우 while문에서 정지

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # 약한 참조 y()

##################################################################
# funtion
##################################################################


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
# 0차원의 ndarray 형식의 데이터를 제곱을 하면 np.float64형식이 되기 때문에
# forward output이 스칼라로 바뀌는 것 방지


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    # #입출력 변수 하나인 경우
    # def __call__(self, input):
    #     x = input.data
    #     y = self.forward(x)
    #     output = Variable(as_array(y))
    #     output.set_creator(self)
    #     self.input = input
    #     self.output = output
    #     return output

    # 입출력 변수 리스트로 변경
    # *inputs: *을 앞에 사용하면 리스트대신 임의 개수의 인수를 사용가능 ex) f([1,2])==f(1,2)
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]  # inputs x들의 리스트 xs로 변환
        ys = self.forward(*xs)  # *xs:언팩 xs에 들어오는 리스트를 풀어 인수 여러개로 전달
        if not isinstance(ys, tuple):
            ys = (ys,)
        # xs의 forward 결과 ys를 y들의 리스트 outputs로 변환
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # Config.enable_backprop가 True일때만 역전파 코드 실행
            self.generation = max([x.generation for x in inputs])
            # 몇 세대까지의 함수가 존재하는지 단계를 파악하여 Funtion class의 generation 설정
            for output in outputs:
                output.set_creator(self)  # ouput 변수의 func과 generation 저장 및 설정
            self.inputs = inputs
            # 순환 참조를 막기위해 weakref를 사용
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, in_data):
        raise NotImplementedError()
    # matric의 함수를 호출하지 않고 그냥 Function을 호출시 에러 호출

    def backward(self, gy):
        raise NotImplementedError()

##################################################################
# matric
##################################################################


class Square(Function):
    def forward(self, x):
        output = x**2
        return output

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx


def square(x):
    f = Square()
    return f(x)
# 사용 편의성을 위해 Class를 받는 함수로 정의

#############################################################


class Exp(Function):
    def forward(self, x):
        output = np.exp(x)
        return output

    def backward(self, gy):
        x = self.inputs[0].data
        # np.exp에서 알아서 처리를 해주기 때문에 inputs[0] 할 필요 x
        gx = np.exp(x)*gy
        return gx


def exp(x):
    f = Exp()
    return f(x)
# 사용 편의성을 위해 Class를 받는 함수로 정의

#############################################################


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y  # 튜플로 반환

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)
# 사용 편의성을 위해 Class를 받는 함수로 정의


Variable.__add__ = add
Variable.__radd__ = add

#############################################################


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0


def mul(x0, x1):
    x1 = as_array(x1)
    f = Mul()
    return f(x0, x1)
# 사용 편의성을 위해 Class를 받는 함수로 정의


Variable.__rmul__ = mul

#############################################################


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    f = Neg()
    return f(x)
# 사용 편의성을 위해 Class를 받는 함수로 정의


Variable.__neg__ = neg

#############################################################


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x0, x1)
# 사용 편의성을 위해 Class를 받는 함수로 정의


def rsub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x1, x0)


Variable.__sub__ = sub
Variable.__rsub__ = rsub

#############################################################


class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x0, x1)
# 사용 편의성을 위해 Class를 받는 함수로 정의


def rdiv(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x1, x0)


Variable.__div__ = div
Variable.__rdiv__ = rdiv

#############################################################


class Pow(Function):
    def __init__(self, c):  # c는 순전파 메서드를 받으면 안되기 때문
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c*x**(c-1)*gy
        return gx


def pow(x, c):
    # f=Pow(c)
    # return f(x)
    return Pow(c)(x)  # c는 순전파 메서드를 받으면 안되기 때문


Variable.__pow__ = pow

#############################################################


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy*np.cos(x)
        return gx


def sin(x):
    f = Sin()
    return f(x)

#############################################################
