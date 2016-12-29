import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, dJ_dout):
        raise NotImplementedError

    def decay_learning_rate(self):
        '''
        only FullyConnectedLayer implementations need to decay the rates
        '''
        pass

    def get_clone(self, n_batch):
        '''
        clones self for a prediction batch
        '''
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    '''
    Fully connects n_in inputs to n_out outputs. The weight matrix will be <n_out> * (<n_in> + 1) to account for the bias
    init_std: a function that takes in n_in and n_out and returns the standard deviation used to initialize the weights
    add_bias: by default assume that each input point has n_in values (so add_bias=True appends a 1 to the end of the inputs)
    calc_dJ_din: by default assume that we need to calculate gradient of loss w.r.t. input. Set to false if first layer
    '''
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate, init_std, add_bias=True, calc_dJ_din=True):
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.decay_rate = decay_rate
        self.init_std = init_std
        self.add_bias = add_bias
        self.calc_dJ_din = calc_dJ_din
        
        self.input = None
        self.output = np.zeros((self.n_batch, self.n_out), dtype=float)
        self.W = np.random.normal(0, self.init_std(self.n_in, self.n_out), (self.n_out, self.n_in + 1))
        self.dJ_dW = np.zeros((self.n_out, self.n_in + 1), dtype=float)
        self.dJ_din = np.zeros((self.n_batch, self.n_in), dtype=float)

    def forward(self, input):
        if self.add_bias:
            input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
        if self.dropout_rate > 0:
            input *= np.random.binomial(np.ones(input.shape, dtype=int), 1 - self.dropout_rate) / (1 - self.dropout_rate)
        self.input = input
        np.dot(self.input, self.W.T, out=self.output)
        return self.output

    def backward(self, dJ_dout):
        np.dot(dJ_dout.T, self.input, out=self.dJ_dW)
        if self.calc_dJ_din:
            np.dot(dJ_dout, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.learning_rate / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def decay_learning_rate(self):
        self.learning_rate *= self.decay_rate

    def get_clone(self, n_batch):
        '''
        dropout rate to 0 for prediction
        '''
        layer = FullyConnectedLayer(self.n_in, self.n_out, n_batch, self.learning_rate, 0, self.decay_rate, self.init_std, self.add_bias, self.calc_dJ_din)
        layer.W = self.W
        return layer

class ReLULayer(Layer):
    '''
    Apply the ReLU function. Input and output will be the same dimension (with n_batch as the first dim)
    shape: shape of input / output for ONE sample (not one batch)
    '''
    def __init__(self, shape, n_batch):
        self.shape = shape
        self.output = np.zeros((n_batch,) + self.shape, dtype=float)
        self.dJ_din = np.zeros((n_batch,) + self.shape, dtype=float)

    def forward(self, input):
        np.maximum(input, 0, out=self.output)
        return self.output

    def backward(self, dJ_dout):
        np.maximum(self.output, 0, out=self.dJ_din)
        self.dJ_din[self.output > 0] = 1
        np.multiply(dJ_dout, self.dJ_din, out=self.dJ_din)
        return self.dJ_din

    def get_clone(self, n_batch):
        return ReLULayer(self.shape, n_batch)

class SoftmaxLayer(Layer):
    '''
    Apply the softmax function. Input and output have dimensions <n_batch> * <n_in_out>
    '''
    def __init__(self, n_in_out, n_batch):
        self.n_in_out = n_in_out
        self.output = np.zeros((n_batch, self.n_in_out), dtype=float)
        self.dJ_din = np.zeros((n_batch, self.n_in_out), dtype=float)

    def forward(self, input, predict_only=False, labels=None):
        np.subtract(input, np.amax(input, axis=1, keepdims=True), out=self.output)
        if predict_only:
            J = 0
            if labels is not None:  # compute loss as well
                batch_sums = np.sum(np.exp(self.output), axis=1, keepdims=True)
                # np.einsum call performs row-wise dot product between the two arguments (returning a vector)
                J = -np.sum(np.einsum('ij,ij->i', self.output - np.log(batch_sums), labels))
            return np.argmax(self.output, axis=1), J
        self.output = np.exp(self.output, out=self.output)
        batch_sums = np.sum(self.output, axis=1, keepdims=True)
        self.output /= batch_sums
        return self.output

    def backward(self, labels):
        np.subtract(self.output, labels, out=self.dJ_din)
        return self.dJ_din

    def get_clone(self, n_batch):
        return SoftmaxLayer(self.n_in_out, n_batch)

class ReshapeLayer(Layer):
    '''
    Reshape the numpy input, but in each case <n_batch> is the first dimension
    '''
    def __init__(self, shape_input, shape_output, n_batch):
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.batch_input = (n_batch,) + shape_input
        self.batch_output = (n_batch,) + shape_output

    def forward(self, input):
        return input.reshape(self.shape_output)

    def backward(self, dJ_dout):
        return dJ_dout.reshape(self.shape_input)

    def get_clone(self, n_batch):
        return ReshapeLayer(self.shape_input, self.shape_output, n_batch)

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
        self.input = self.process_input(input, self.add_bias)
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

    def get_clone(self, n_batch, dropout_rate):
        layer = ReLULayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay, self.add_bias, self.calculate_dJ_din)
        layer.W = self.W
        return layer
