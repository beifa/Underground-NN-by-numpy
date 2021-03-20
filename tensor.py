import functools
import numpy as np

class Tensor:
    def __init__(self, data):

        if type(data) != np.ndarray:
            # set_trace()
            print('error type data %r' % data)
            assert(False)
        self.data = data        
        self.grad = None
        self._ctx = None
    
    def __repr__(self):
        # tensor([1.]))
        return f'tensor({self.data})'

    def backward(self, fill = True):
        
        # print("running backward on", self)

        if self._ctx is None:
            return

        if self.grad is None and fill:
            # iniy first grad ones
            # print(self.data.size)
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        
        assert(self.grad is not None)  

        """
        _ctx.backward return from <class '__main__.Mul'> 
        and method Mul.backward return y * grad_out, x * grad_out
        
        """      
        grads = self._ctx.backward(self._ctx, self.grad)
        # print('Func name: ',type(self._ctx))
        # print('Grad: ', self.grad.shape)
        # print('Self: ', self.data.shape)
        # set_trace()
        if len(self._ctx.parents) == 1:
            grads = [grads]

        # print(self._ctx.parents, grads)

        for t, g in zip(self._ctx.parents, grads):
            # t  Tensor
            """
            print(g.shape,t.data.shape)

                    (1,) (1,)
                    (30, 10) (30, 10)
                    (30, 10) (30, 10)
                    (1, 30, 10) (30, 10) ??? why, fix squeeze
                    (30, 128) (30, 128)
                    (30, 128) (30, 128)
                    (30, 784) (30, 784)
                    (784, 128) (784, 128)
                    (128, 10) (128, 10)
                    (30, 10) (30, 10)
                    (1,) (1,)             
            """
               
            # print(g.shape, t.data.shape)
            if g.shape != t.data.shape :
                g = np.squeeze(g)              
           
            if g.shape != t.data.shape:
                print('grad shape must match tensor shape')                
                assert(False)
            t.grad = g           
            t.backward(False)          
      

class Function:

    def __init__(self, *tensor):
        self.parents = tensor  
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)    

    def apply(self, arg, *x):
        """
        почему мы здесь а потому что fn.apply и передаем аргумент fn
        a = Tensor(np.array([1]))
        b = Tensor(np.array([3]))
        
        a.mul(b)
        
        arg: <class '__main__.Mul'> , fn
        self.data -> Tensor.data->[1] a
        [3] = b        
        
        """        
        ctx = arg(self, *x)         
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret


def register(name, fn):
    """
    class A:
        print('hell')
    a = A()
    setattr(a, 'oo', lambda x: x *2)
    a.oo(2)
    >4

    we add mul to class Tensor    

    partialmethod(fumc, arg)
    """    
    # set_trace()
    setattr(Tensor, name, functools.partialmethod(fn.apply, fn))


class Mul(Function):
    """
    out = x.mul.y
    back
    out/dx, out/dy
    
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_out):       
        # set_trace()
        x, y = ctx.saved_tensors       
        return y * grad_out, x * grad_out

register('mul', Mul)

class Add(Function):
    """sum"""
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad_out):     
        x, y = ctx.saved_tensors       
        return grad_out, grad_out
register('add', Add)

class ReLU(Function):
    """relu"""
    @staticmethod
    def forward(ctx, in_val):
        ctx.save_for_backward(in_val)
        return np.maximum(in_val, 0)

    @staticmethod
    def backward(ctx, grad_out):     
        in_val = ctx.saved_tensors         
        grad_out[in_val[0] < 0] = 0   
        return grad_out

register('relu', ReLU)        

class Sum(Function):
    """sum array each dx is 1"""
    @staticmethod
    def forward(ctx, in_val):
       
        ctx.save_for_backward(in_val)
        return np.array([in_val.sum()])

    @staticmethod
    def backward(ctx, grad_out):
        # set_trace()     
        in_val = ctx.saved_tensors #!!!!!!!!!!!!!!!!!!!!!!!!!!            
        return grad_out * np.ones_like(in_val[0])


register('sum', Sum)

class Dot(Function):
    """
    a = [[1, 0], [0, 1]]
    b = [[4, 1], [2, 2]]
    np.dot(a, b)
    >> array([[4, 1],
             [2, 2]])  
    
    """
    @staticmethod
    def forward(ctx, in_val, w):
        ctx.save_for_backward(in_val, w)
        return in_val.dot(w)

    @staticmethod
    def backward(ctx, grad_out):     
        in_val, w = ctx.saved_tensors
        # set_trace()
        g_grad = grad_out.dot(w.T) # (10,10,1) and (128,10)  after 10, 128
        w_grad = grad_out.T.dot(in_val).T
        return g_grad, w_grad

register('dot', Dot)


class Log_softmax(Function):

    @staticmethod
    def forward(ctx, x):        
        mx = x.max(axis=1)
        stbl = mx + np.log(np.exp(x- mx.reshape((-1, 1))).sum(axis=1))        
        out = x - stbl.reshape(-1,1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):     
        x = ctx.saved_tensors
        # set_trace() 
        return  grad_out - np.exp(x)*grad_out.sum(axis=1).reshape((-1, 1))      
 
register('log_softmax', Log_softmax)