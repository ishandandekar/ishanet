"""
Here's a function that can train a neural net
"""
from tqdm.auto import tqdm
from tensor import Tensor
from nn import NeuralNet
from loss import Loss, MSE
from optimizers import Optimizer, SGD
from data import DataIterator, BatchIterator


def train(
    net: NeuralNet,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 5000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD(),
) -> None:
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for batch in iterator(inputs=inputs, targets=targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted=predicted, actual=batch.targets)
            net.backward(grad=grad)
            optimizer.step(net=net)
