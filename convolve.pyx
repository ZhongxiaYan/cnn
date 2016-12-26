import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

# def convForward(np.ndarray[np.float_t, ndim=2] batch, np.ndarray[np.float_t, ndim=2] filt, int dim_image, int dim_filter, int num_filters):
#     '''
#     batch: <batch_size> * <dim_image ^ 2>
#     filt: <num_filters> * <dim_filter ^ 2>
#     output: 2d convolution of each row in batch with filt, for a total of <batch_size> rows
#     Assume stride = 1, filt_dim = 5, dim_image = 28
#     '''

#     cdef int zero_pad = 1 # padding on each side
#     cdef int stride = 1
#     cdef int output_dim = (dim_image - dim_filter + 2 * zero_pad) // stride + 1
#     cdef np.ndarray[np.float_t, ndim=2] result = np.zeros((batch.shape[0], output_dim * output_dim * num_filters), dtype=np.float)

def convolve2D(np.ndarray[np.float_t, ndim=1] a_1, np.ndarray[np.float_t, ndim=1] a_2, int dim_a_1, int dim_a_2, int padding, int stride, np.ndarray[np.float_t, ndim=1] output):
    """
    padding: padding on left, right, top, and bottom
    'valid' convolution: dim_a_2 <= dim_a_1
    """
    cdef int dim_output = (dim_a_1 - dim_a_2 + 2 * padding) // stride + 1
    cdef int x_out, y_out
    cdef int x_1, y_1, x_2, y_2, x_1_2, y_1_2
    cdef np.float_t output_xy
    for x_out in range(dim_output):
        x_1 = x_out * stride - padding
        for y_out in range(dim_output):
            y_1 = y_out * stride - padding
            output_xy = 0
            for x_2 in range(dim_a_2):
                for y_2 in range(dim_a_2):
                    y_1_2 = (y_1 + y_2)
                    x_1_2 = (x_1 + x_2)
                    if y_1_2 < 0 or x_1_2 < 0 or y_1_2 >= dim_a_1 or x_1_2 >= dim_a_1:
                        continue
                    output_xy += a_1[y_1_2 * dim_a_1 + x_1_2] * a_2[y_2 * dim_a_2 + x_2]
            output[y_out * dim_output + x_out] = output_xy
    