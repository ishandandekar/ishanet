"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropogation
"""
from nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 1e-2) -> None:
        super().__init__()
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grams():
            param -= self.lr * grad
