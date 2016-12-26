import numpy as np
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.alpha = learning_rate
        self.dropout = dropout_rate
        self.decay = decay_rate

    def forward(self, input):
        raise NotImplementedError

    def backward(self, dJ_dout):
        raise NotImplementedError

    def processInput(self, input, add_bias=True):
        if add_bias:
            input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
        if self.dropout > 0:
            input *= np.random.binomial(np.ones(input.shape, dtype=int), 1 - self.dropout) / (1 - self.dropout)
        return input

    def decayLearningRate(self):
        self.alpha *= self.decay

    def getClone(self, n_batch, dropout_rate):
        '''
        clones self for a larger batch - should set dropout rate to 0 when testing
        '''
        raise NotImplementedError

class ReLULayer(Layer):
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate, add_bias, calculate_dJ_din):
        super().__init__(n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate)
        self.input = None
        self.W = np.random.normal(0, np.sqrt(2.0 / (n_in + n_out + 1)), (n_out, n_in + 1))
        self.output = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dWin = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dW = np.zeros((n_out, n_in + 1), dtype=float)
        self.dJ_din = np.zeros((n_batch, n_in), dtype=float)
        self.add_bias = add_bias
        self.calculate_dJ_din = calculate_dJ_din

    def forward(self, input):
        self.input = self.processInput(input, self.add_bias)
        np.dot(self.input, self.W.T, out=self.output)
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, dJ_dout):
        self.dJ_dWin[self.output <= 0] = 0
        self.dJ_dWin[self.output > 0] = 1
        np.multiply(dJ_dout, self.dJ_dWin, out=self.dJ_dWin)
        np.dot(self.dJ_dWin.T, self.input, out=self.dJ_dW)
        if self.calculate_dJ_din:
            np.dot(self.dJ_dWin, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.alpha / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def getClone(self, n_batch, dropout_rate):
        layer = ReLULayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay, self.add_bias, self.calculate_dJ_din)
        layer.W = self.W
        return layer

class SoftmaxLayer(Layer):
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate):
        super().__init__(n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate)
        self.input = None
        self.W = np.random.normal(0, 1 / np.sqrt(n_in), (n_out, n_in + 1))
        self.output = np.zeros((n_batch, n_out), dtype=float)
        self.dout_dWin = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dW = np.zeros((n_out, n_in + 1), dtype=float)
        self.dJ_din = np.zeros((n_batch, n_in), dtype=float)

    def forward(self, input, predict_only=False, labels=None):
        self.input = self.processInput(input, True)
        np.dot(self.input, self.W.T, out=self.output)
        self.output -= np.amax(self.output, axis=1)[:, None]
        if predict_only:
            J = 0
            if labels is not None:  # compute loss as well
                batch_sums = np.sum(np.exp(self.output), axis=1)[:, None]
                J = -np.sum(inner1d(self.output - np.log(batch_sums), labels))
            return np.argmax(self.output, axis=1), J
        self.output = np.exp(self.output, out=self.output)
        batch_sums = np.sum(self.output, axis=1)[:, None]
        self.output /= batch_sums
        return self.output

    def backward(self, labels):
        dJ_ds = self.output - labels
        np.dot(dJ_ds.T, self.input, out=self.dJ_dW)
        np.dot(dJ_ds, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.alpha / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def getClone(self, n_batch, dropout_rate):
        layer = SoftmaxLayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay)
        layer.W = self.W
        return layer

class ConvLayer(Layer):
    def __init__(self, n_in, dim_filter, padding, stride, input_depth, output_depth, n_batch, learning_rate, dropout_rate, decay_rate, add_bias):
        self.dim_input = int(np.sqrt(n_in))
        assert(dim_input ** 2 == n_in)
        assert(dim_filter < dim_input)
        self.dim_output = (dim_input - dim_filter + 2 * padding) // stride + 1
        super().__init__(n_in, dim_output ** 2, n_batch, learning_rate, dropout_rate, decay_rate)
        self.n_filter = dim_filter ** 2
        self.input = None
        self.padding = padding
        self.stride = stride
        self.input_depth = input_depth
        self.output_depth = output_depth

        self.W = np.random.normal(0, np.sqrt(2.0 / (n_in + n_out + 1)), (n_batch, output_depth, input_depth, n_filter))
        self.output = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dWin = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dW = np.zeros((n_out, n_in + 1), dtype=float)
        self.dJ_din = np.zeros((n_batch, n_in), dtype=float)
        self.add_bias = add_bias
        self.calculate_dJ_din = calculate_dJ_din

    def forward(self, input):
        self.input = self.processInput(input, self.add_bias)
        np.dot(self.input, self.W.T, out=self.output)
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, dJ_dout):
        self.dJ_dWin[self.output <= 0] = 0
        self.dJ_dWin[self.output > 0] = 1
        np.multiply(dJ_dout, self.dJ_dWin, out=self.dJ_dWin)
        np.dot(self.dJ_dWin.T, self.input, out=self.dJ_dW)
        if self.calculate_dJ_din:
            np.dot(self.dJ_dWin, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.alpha / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def getClone(self, n_batch, dropout_rate):
        layer = ReLULayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay, self.add_bias, self.calculate_dJ_din)
        layer.W = self.W
        return layer
