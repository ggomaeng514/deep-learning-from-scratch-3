from steps.framework_base import Variable


def centered_diff(f, x, eps=1e-4):  # 중앙차분
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    output = (y1.data-y0.data)/(2*eps)
    return output


def forward_diff(f, x, eps=1e-4):  # 전진차분
    x0 = Variable(x.data)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    output = (y1.data-y0.data)/eps
    return output
