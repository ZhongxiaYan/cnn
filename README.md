# Convolutional Neural Network (a Python / Cython implementation)

A customizable, multilayered, and fully Numpy-parallelized/cythonized (convolutional) neural network classifier. Hyperparameters are tuned for the MNIST dataset (each sample input is dim 784, representing a 28 by 28 pixelmap; each sample output is preprocessed to be a one-hot encoding of the digits).

## Results:
### Single Hidden Layer (FC1 -> ReLU -> FC2 -> Softmax)
##### Cross validation accuracy: **0.9845**
#### General parameters
100,000 iterations with batch size of 20

#### FC1 parameters
`n_in`: `784`
`n_out`: `700`
`learning_rate`: `0.5`
`dropout_rate`: `0.3`
`decay_rate`: `0.9`
`init_std`: `2 / sqrt(n_in + n_out + 1)`

#### FC2 parameters
`n_in`: `700`
`n_out`: `10`
`learning_rate`: `0.1`
`dropout_rate`: `0.5`
`decay_rate`: `0.9`
`init_std`: `1 / sqrt(n_in)`

## Weighted layers:
### FullyConnectedLayer
Fully connects `n_in` neurons in the input layer to `n_out` neurons in the output layer. The weight matrix will be `n_out * (n_in + 1)` to account for the bias. Input should be dimension `n_batch * n_in`, or `n_batch * (n_in + 1)` with the last column all 1's if add_bias=False. Output has dimension `n_batch * n_out`.

`learning_rate`, `decay_rate`, and `dropout_rate` can be set. `learning_rate` is multiplied by `decay_rate` every epoch (N samples).

### ConvolutionLayer
Perform a "valid" 2D convolution with a weight matrix on the input. Assume square symmetry for input, W, and output. Input should have dimensions `n_batch * input_depth * dim_input * dim_input`. W has dimensions `output_depth * input_depth * dim_W * dim_W`. `input_depth` specifies the number of channels in the input, and `output_depth` specifies the number of channels in the output.

Output has dimensions `n_batch * output_depth * dim_output * dim_output`, where `dim_output = (dim_input - dim_W + 2 * padding) // stride + 1`. `padding` specifies the size of the 0 padding on each side of the input matrix. `stride` specifies how far to shift the weight matrix each time during convolution with the input.

## Weightless layers:
### ReLULayer
### SoftmaxLayer
### MaxPoolLayer
### ReshapeLayer
### TransposeLayer
