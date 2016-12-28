import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

# def convForward(np.ndarray[np.float_t, ndim=2] batch, np.ndarray[np.float_t, ndim=2] filt, int dim_input, int dim_W, int num_filters):
#     '''
#     batch: <batch_size> * <dim_input ^ 2>
#     filt: <num_filters> * <dim_W ^ 2>
#     output: 2d convolution of each row in batch with filt, for a total of <batch_size> rows
#     Assume stride = 1, filt_dim = 5, dim_input = 28
#     '''

#     cdef int zero_pad = 1 # padding on each side
#     cdef int stride = 1
#     cdef int output_dim = (dim_input - dim_W + 2 * zero_pad) // stride + 1
#     cdef np.ndarray[np.float_t, ndim=2] result = np.zeros((batch.shape[0], output_dim * output_dim * num_filters), dtype=np.float)

def conv_forward(np.ndarray[np.float_t, ndim=1] input, np.ndarray[np.float_t, ndim=1] W, int dim_input, int dim_W, int padding, int stride, np.ndarray[np.float_t, ndim=1] output):
    """
    padding: padding on left, right, top, and bottom
    'valid' convolution: dim_W <= dim_input
    """
    cdef int dim_output = (dim_input - dim_W + 2 * padding) // stride + 1
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef np.float_t output_mn
    for m in range(dim_output):
        i = m * stride - padding
        for n in range(dim_output):
            j = n * stride - padding
            output_mn = 0
            for a in range(dim_W):
                ia = (i + a)
                if ia < 0 or ia >= dim_input:
                    continue
                for b in range(dim_W):
                    jb = (j + b)
                    if jb < 0 or jb >= dim_input:
                        continue
                    output_mn += input[ia * dim_input + jb] * W[a * dim_W + b]
            output[m * dim_output + n] = output_mn

def conv_backward_W(np.ndarray[np.float_t, ndim=1] dJ_dout, np.ndarray[np.float_t, ndim=1] input, int dim_input, int dim_W, int padding, int stride, np.ndarray[np.float_t, ndim=1] dJ_dW):
    cdef int dim_output = (dim_input - dim_W + 2 * padding) // stride + 1
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef np.float_t dJ_dW_ab
    for a in range(dim_W):
        for b in range(dim_W):
            dJ_dW_ab = 0
            for m in range(dim_output):
                i = m * stride - padding
                ia = (i + a)
                if ia < 0 or ia >= dim_input:
                    continue
                for n in range(dim_output):
                    j = n * stride - padding
                    jb = (j + b)
                    if jb < 0 or jb >= dim_input:
                        continue
                    dJ_dW_ab += input[ia * dim_input + jb] * dJ_dout[m * dim_output + n]
            dJ_dW[a * dim_W + b] = dJ_dW_ab

def conv_backward_input(np.ndarray[np.float_t, ndim=1] dJ_dout, np.ndarray[np.float_t, ndim=1] W, int dim_input, int dim_W, int padding, int stride, np.ndarray[np.float_t, ndim=1] dJ_din):
    cdef int dim_output = (dim_input - dim_W + 2 * padding) // stride + 1
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int ip, jp # temp variables
    cdef int m_iter, n_iter, a_iter, b_iter # iteration variables
    cdef np.float_t dJ_dinput_ij
    for i in range(dim_input):
        ip = i + padding
        m = ip // stride
        a = ip % stride
        if m >= dim_output:
            a += stride * (m - dim_output + 1)
            m = dim_output - 1
        for j in range(dim_input):
            jp = j + padding
            n = jp // stride
            b = jp % stride
            dJ_dinput_ij = 0
            if n >= dim_output:
                b += stride * (n - dim_output + 1)
                n = dim_output - 1
            m_iter = m
            a_iter = a
            while m_iter >= 0 and a_iter < dim_W:
                n_iter = n
                b_iter = b
                while n_iter >= 0 and b_iter < dim_W:
                    dJ_dinput_ij += dJ_dout[m_iter * dim_output + n_iter] * W[a_iter * dim_W + b_iter]
                    n_iter -= 1
                    b_iter += stride
                m_iter -= 1
                a_iter += stride
            dJ_din[i * dim_input + j] = dJ_dinput_ij
