from .layer_utils import *
import numpy as np
# 构建神经网络


class TwoLayerNet(object):
    # hidden_dim:隐层取100个神经元
    # num_classes:输出10个分类的概率值
    # weight_scale:用于扩大输出的概率值
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):

        self.params = {}    
        self.reg = reg

        # 初始化权重参数
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)     
        self.params['b1'] = np.zeros((1, hidden_dim))    
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)  
        self.params['b2'] = np.zeros((1, num_classes))

    def loss(self, X, y=None):    

        scores = None
        N = X.shape[0]

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # h1: Relu(w*x+b)
        # cache1: ((x,w,b), w*x+b)
        h1, cache1 = affine_relu_forward(X, W1, b1)
        # 输出层没有ReLu函数
        out, cache2 = affine_forward(h1, W2, b2)
        scores = out              # (N,C)
        # If y is None then we are in test mode so just return scores
        if y is None:   
            return scores

        loss, grads = 0, {}

        # def softmax_loss(x, y):
        # 归一化
        #     probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        #     probs /= np.sum(probs, axis=1, keepdims=True)
        #     N = x.shape[0]
        # 求概率
        #     loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        #     dx = probs.copy()
        # SoftMax的梯度值 = 属于真实类别的概率值 - 1
        #     dx[np.arange(N), y] -= 1
        #     dx /= N
        #
        #     return loss, dx
        data_loss, dscores = softmax_loss(scores, y)
        # L2正则化
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        loss = data_loss + reg_loss

        # Backward pass: compute gradients
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        # 先ReLu反向传播，在计算第一个隐含层的反向传播
        dX, dW1, db1 = affine_relu_backward(dh1, cache1)

        # Add the regularization gradient contribution
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads