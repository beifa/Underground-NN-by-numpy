# Underground-NN-by-numpy


## Example
```python
from nntensor.tensor import Tensor

a = Tensor(np.array([1]))
b = Tensor(np.array([3]))
c = a.mul(b)
c.backward()
print('Numpy grad: ', c)
print('Numpy Grad: ', a.grad, b.grad)

```

## Example NN

```python
from nntensor.tensor import Tensor
import nntensor.optim as optim
import nntensor.utils as utils


class NN:
    def __init__(self, l1, l2):
        self.l1 = Tensor(l1)
        self.l2 = Tensor(l2)

    def forward(self, x):
        x = x.dot(self.l1)
        x = x.relu()
        x = x.dot(self.l2)
        return x.log_softmax()

model = NN(l1, l2)
optimizer = optim.SGD([model.l1, model.l2], momentum=0.7)  
bar = trange(500)
for i in bar:
    ...
    y_ = model.forward(X)
    ...
    loss.backward()
    optimizer.step()

```


## TODO

- Implement convolutions
- Add features
