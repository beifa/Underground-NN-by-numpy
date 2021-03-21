import numpy as np

class SGD:
    def __init__(self, tensors, lr=0.001, momentum = 0.):
        """
        tensors: array 
        use:
            optim = SGD([model.l1, model.l2], lr = lr)
            ...
            optim.step() 

        momentum Nesterov implement use https://cs231n.github.io/neural-networks-3/
        but I'm not sure right 
        
        """
        self.lr = lr
        self.tensors = tensors
        self.v1 = 0.
        self.v2 = 0.
        self.mo = momentum
        if self.mo < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

    
    def nag(self, v, grad):
        # Nesterov’s Accelerated Momentum (NAG)
        v_prev = v 
        v = self.mo * v - self.lr * grad        
        return v, -self.mo * v_prev + (1 + self.mo) * v


    def step(self):
        # momentum
        if self.mo != 0.:
            for t, v in zip(self.tensors, [self.v1, self.v2]):
                v, upd =  self.nag(v, t.grad)
                t.data +=  upd 

        for t in self.tensors:
            t.data -= self.lr * t.grad     


class Adam:
    # https://arxiv.org/pdf/1412.6980.pdf
    def __init__(self,
                 tensors,
                 lr=1e-4,
                 b1 = 0.9,
                 b2 = 0.999,
                 eps = 1e-8):     

        self.lr = lr
        self.tensors = tensors
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.groups = {}

        # init param
        self.m = [np.zeros_like(t.data) for t in self.tensors]
        self.v = [np.zeros_like(t.data) for t in self.tensors]

        self.t = 0


    def step(self):
        self.t += 1
        for i, t in enumerate(self.tensors):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)

            m_hat = self.m[i] / (1. - self.b1 ** self.t)
            v_hat = self.v[i] / (1. - self.b2 ** self.t)

            t.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)      
