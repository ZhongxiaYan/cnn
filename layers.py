import numpy as np
from convolve import *
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
    Fully connects n_in inputs to n_out outputs. The weight matrix will be <n_out> * (<n_in> + 1) to account for the bias.
    Input should be dimension <n_batch> * <n_in>, or <n_batch> * (<n_in> + 1) if add_bias=False. Output has dimension <n_batch> * <n_out>
    '''
    def __init__(self, n_batch, n_in, n_out, learning_rate, dropout_rate, decay_rate, init_std, add_bias=True, calc_dJ_din=True):
        '''
        init_std: a function that takes in n_in and n_out and returns the standard deviation used to initialize the weights
        add_bias: by default assume that each input point has n_in values (so add_bias=True appends a 1 to the end of the inputs)
        calc_dJ_din: by default assume that we need to calculate gradient of loss w.r.t. input. Set to false if first layer
        '''
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
        layer = FullyConnectedLayer(n_batch, self.n_in, self.n_out, self.learning_rate, 0, self.decay_rate, self.init_std, self.add_bias, self.calc_dJ_din)
        layer.W = self.W
        return layer

class ReLULayer(Layer):
    '''
    Apply the ReLU function. Input and output will be the same dimension.
    Input / output should have shape <n_batch> * <shape>
    '''
    def __init__(self, n_batch, shape):
        '''
        shape: shape of input / output for ONE sample (not one batch)
        '''
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
        return ReLULayer(n_batch, self.shape)

class SoftmaxLayer(Layer):
    '''
    Apply the softmax function. Input and output have dimensions <n_batch> * <n_in_out>
    '''
    def __init__(self, n_batch, n_in_out):
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
        return SoftmaxLayer(n_batch, self.n_in_out)

class ReshapeLayer(Layer):
    '''
    Reshape the input with dim <n_batch> * <shape_input> into output with shape <n_batch> * <shape_output>
    '''
    def __init__(self, n_batch, shape_input, shape_output):
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.batch_input = (n_batch,) + shape_input
        self.batch_output = (n_batch,) + shape_output

    def forward(self, input):
        return input.reshape(self.shape_output)

    def backward(self, dJ_dout):
        return dJ_dout.reshape(self.shape_input)

    def get_clone(self, n_batch):
        return ReshapeLayer(n_batch, self.shape_input, self.shape_output)

class TransposeLayer(Layer):
    '''
    Reshape the input into output with the order (0, <i + 1 for i in transposition>)
    In other words don't pass in the batch dimension as a transposition argument
    '''
    def __init__(self, n_batch, transposition):
        '''
        transposition: per SAMPLE, should be 0 indexed
        '''
        self.transposition = tuple([0] + [x + 1 for x in transposition])
        reverse_transpostion = [0 for i in self.transposition]
        for i, x in enumerate(self.transposition):
            reverse_transpostion[x] = i
        self.reverse_transpostion = tuple(reverse_transpostion)

    def forward(self, input):
        return input.transpose(self.transposition)

    def backward(self, dJ_dout):
        return dJ_dout.reshape(self.reverse_transpostion)

    def get_clone(self, n_batch):
        return TransposeLayer(n_batch, self.transposition)

class ConvolutionLayer(Layer):
    '''
    Convolves input with W to get output. Assume square symmetry for input, W, and output
    Input should have dimensions <n_batch> * <input_depth> * <dim_input> * <dim_input>
    W has dimensions <output_depth> * <input_depth> * <dim_W> * <dim_W>
    Output has dimensions <n_batch> * <output_depth> * <dim_output> * <dim_output>
    '''
    def __init__(self, n_batch, dim_input, dim_W, input_depth, output_depth, padding, stride, learning_rate, decay_rate, calc_dJ_din=True):
        self.padding = padding
        self.stride = stride
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.calc_dJ_din = calc_dJ_din
        dim_output = (dim_input - dim_filter + 2 * padding) // stride + 1

        self.input = None
        self.W = np.random.normal(0, np.sqrt(2.0 / (n_in + n_out + 1)), (output_depth, input_depth, dim_W, dim_W))
        self.output = np.zeros((n_batch, output_depth, dim_output, dim_output), dtype=float)
        self.dJ_dW = np.zeros(W.shape, dtype=float)
        self.dJ_din = np.zeros((n_batch, input_depth, dim_input, dim_input), dtype=float)

    def forward(self, input):
        self.input = input
        conv_forward(self.input, self.W, self.padding, self.stride, self.output)
        return self.output

    def backward(self, dJ_dout):
        conv_backward_W(dJ_dout, self.input, self.padding, self.stride, self.dJ_dW)
        if self.calc_dJ_din:
            conv_backward_input(dJ_dout, self.W, self.padding, self.stride, self.dJ_din)
        self.dJ_dW *= self.learning_rate / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def decay_learning_rate(self):
        self.learning_rate *= self.decay_rate

    def get_clone(self, n_batch):
        _, input_depth, dim_input, _ = self.dJ_din.shape
        _, output_depth, dim_output, _ = self.output.shape
        dim_W = self.W.shape[2]
        layer = ConvolutionLayer(n_batch, dim_input, dim_W, input_depth, output_depth, self.padding, self.stride, self.learning_rate, self.decay_rate, self.calc_dJ_din)
        layer.W = self.W
        return layer

class MaxPoolLayer(Layer):
    '''
    Max over nonoverlapping <dim_pool> * <dim_pool> regions of the <dim_input> * <dim_input> input. Implicitly pad with 0's at the highest indices if necessary
    dim_output = (<dim_input - 1>) // <dim_pool> + 1
    '''
    def __init__(self, n_batch, dim_input, depth, dim_pool):
        self.dim_pool = dim_pool
        dim_output = (dim_input - 1) // self.dim_pool + 1

        self.input = None
        self.output = np.zeros((n_batch, depth, dim_output, dim_output), dtype=float)
        self.dJ_din = np.zeros((n_batch, depth, dim_input, dim_input), dtype=float)

    def forward(self, input):
        self.input = input
        pool_forward(self.input, self.dim_pool, self.output)
        return self.output

    def backward(self, dJ_dout):
        pool_backward(dJ_dout, self.input, self.output, self.dim_pool, self.dJ_din)
        return self.dJ_din

    def get_clone(self, n_batch):
        _, depth, dim_input, _ = self.input.shape
        return MaxPoolLayer(n_batch, dim_input, depth, self.dim_pool)