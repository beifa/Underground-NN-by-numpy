import numpy as np
import matplotlib.pyplot as plt

def vis_same_error(x_test, y_test_t, pred)-> None:

    idx = np.argwhere(y_test_t.data != pred)
    bad = x_test[idx]  
    tmp = []
    for _ in range(12):
        tmp.append(np.random.randint(bad.shape[0]))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.concatenate(bad[tmp].reshape(4, 28*3, 28), axis = 1))
    print('')
    print('Predicted: ', pred[idx][tmp].ravel(), 'Target', y_test_t.data[idx][tmp].ravel())


def evaluate(model, x, y)-> None:
    
    y_= model.forward(x)
    # print(y_.data.shape)
    pred =  np.argmax(y_.data, axis =1)
    acc = (y.data == pred).mean()
    print('Accuracy model: ', acc)
    print('Bad count: ', int(y.data.shape[0] * (1 - acc)))
    return pred