import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_forward(np.ndarray[np.float_t, ndim=4] input, np.ndarray[np.float_t, ndim=4] W, int padding, int stride, np.ndarray[np.float_t, ndim=4] output):
    '''
    input: <batch_size> * <input_depth> * <dim_input> * <dim_input>
    W: <output_depth> * <input_depth> * <dim_W> * <dim_W>
    padding: padding on left, right, top, and bottom
    output: <batch_size> * <output_depth> * <dim_output> * <dim_output>
    '''
    output.fill(0)
    cdef int batch_size = input.shape[0]
    cdef int output_depth = output.shape[1]
    cdef int input_depth = input.shape[1]
    cdef int dim_input = input.shape[2]
    cdef int dim_output = output.shape[2]
    cdef int dim_W = W.shape[2]
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int x, y, z # respectively, indices in batch_size, output_depth, input_depth
    # memory views for efficient accessing
    cdef np.float_t [:, :, :, :] input_buffer = input
    cdef np.float_t [:, :, :, :] output_buffer = output
    cdef np.float_t [:, :, :, :] W_buffer = W
    cdef np.float_t [:, :] input_xz, output_xy, W_yz

    for x in range(batch_size):
        for y in range(output_depth):
            output_xy = output_buffer[x, y]
            for z in range(input_depth):
                W_yz = W_buffer[y, z]
                input_xz = input_buffer[x, z]
                for m in range(dim_output):
                    i = m * stride - padding
                    for a in range(dim_W):
                        ia = (i + a)
                        if ia < 0 or ia >= dim_input:
                            continue
                        for n in range(dim_output):
                            j = n * stride - padding
                            for b in range(dim_W):
                                jb = (j + b)
                                if jb < 0 or jb >= dim_input:
                                    continue
                                output_xy[m, n] += input_xz[ia, jb] * W_yz[a, b]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_backward_W(np.ndarray[np.float_t, ndim=4] dJ_dout, np.ndarray[np.float_t, ndim=4] input, int padding, int stride, np.ndarray[np.float_t, ndim=4] dJ_dW):
    '''
    dJ_dout: <batch_size> * <output_depth> * <dim_output> * <dim_output>
    input: <batch_size> * <input_depth> * <dim_input> * <dim_input>
    dJ_dW: <output_depth> * <input_depth> * <dim_W> * <dim_W>
    '''
    dJ_dW.fill(0)
    cdef int batch_size = input.shape[0]
    cdef int output_depth = dJ_dout.shape[1]
    cdef int input_depth = input.shape[1]
    cdef int dim_input = input.shape[2]
    cdef int dim_output = dJ_dout.shape[2]
    cdef int dim_W = dJ_dW.shape[2]
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int x, y, z # respectively, indices in batch_size, output_depth, input_depth
    # memory views for efficient accessing
    cdef np.float_t [:, :, :, :] dJ_dout_buffer = dJ_dout
    cdef np.float_t [:, :, :, :] input_buffer = input
    cdef np.float_t [:, :, :, :] dJ_dW_buffer = dJ_dW
    cdef np.float_t [:, :] input_xz, dJ_dout_xy, dJ_dW_yz

    for x in range(batch_size):
        for y in range(output_depth):
            dJ_dout_xy = dJ_dout[x, y]
            for z in range(input_depth):
                dJ_dW_yz = dJ_dW_buffer[y, z]
                input_xz = input_buffer[x, z]
                for a in range(dim_W):
                    for m in range(dim_output):
                        i = m * stride - padding
                        ia = (i + a)
                        if ia < 0 or ia >= dim_input:
                            continue
                        for b in range(dim_W):
                            for n in range(dim_output):
                                j = n * stride - padding
                                jb = (j + b)
                                if jb < 0 or jb >= dim_input:
                                    continue
                                dJ_dW_yz[a, b] += input_xz[ia, jb] * dJ_dout_xy[m, n]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_backward_input(np.ndarray[np.float_t, ndim=4] dJ_dout, np.ndarray[np.float_t, ndim=4] W, int padding, int stride, np.ndarray[np.float_t, ndim=4] dJ_din):
    '''
    dJ_dout: <batch_size> * <output_depth> * <dim_output> * <dim_output>
    W: <output_depth> * <input_depth> * <dim_W> * <dim_W>
    dJ_din: <batch_size> * <input_depth> * <dim_input> * <dim_input>
    '''
    dJ_din.fill(0)
    cdef int batch_size = dJ_din.shape[0]
    cdef int output_depth = dJ_dout.shape[1]
    cdef int input_depth = dJ_din.shape[1]
    cdef int dim_input = dJ_din.shape[2]
    cdef int dim_output = dJ_dout.shape[2]
    cdef int dim_W = W.shape[2]
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int ip, jp # temp variables
    cdef int m_iter, n_iter, a_iter, b_iter # iteration variables
    cdef int x, y, z # respectively, indices in batch_size, output_depth, input_depth
    # memory views for efficient accessing
    cdef np.float_t [:, :, :, :] dJ_dout_buffer = dJ_dout
    cdef np.float_t [:, :, :, :] dJ_din_buffer = dJ_din
    cdef np.float_t [:, :, :, :] W_buffer = W
    cdef np.float_t [:, :] dJ_dout_xy, W_yz, dJ_din_xz

    for x in range(batch_size):
        for y in range(output_depth):
            dJ_dout_xy = dJ_dout_buffer[x, y]
            for z in range(input_depth):
                W_yz = W_buffer[y, z]
                dJ_din_xz = dJ_din_buffer[x, z]
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
                        if n >= dim_output:
                            b += stride * (n - dim_output + 1)
                            n = dim_output - 1
                        m_iter = m
                        a_iter = a
                        while m_iter >= 0 and a_iter < dim_W:
                            n_iter = n
                            b_iter = b
                            while n_iter >= 0 and b_iter < dim_W:
                                dJ_din_xz[i, j] += dJ_dout_xy[m_iter, n_iter] * W_yz[a_iter, b_iter]
                                n_iter -= 1
                                b_iter += stride
                            m_iter -= 1
                            a_iter += stride

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pool_forward(np.ndarray[np.float_t, ndim=4] input, int dim_pool, np.ndarray[np.float_t, ndim=4] output):
    output.fill(0)
    cdef int n_batch = input.shape[0]
    cdef int depth = input.shape[1]
    cdef int dim_input = input.shape[2]
    cdef int x, y # indices for batch and depth
    cdef int i, j # indices for one input slice
    cdef int m, n # indices for one output slice
    cdef np.float_t [:, :, :, :] input_buffer, output_buffer
    input_buffer = input
    output_buffer = output
    cdef np.float_t [:, :] input_xy, output_xy
    cdef np.float_t input_xyij
    for x in range(n_batch):
        for y in range(depth):
            input_xy = input_buffer[x, y]
            output_xy = output_buffer[x, y]
            for i in range(dim_input):
                m = i // dim_pool
                for j in range(dim_input):
                    n = j // dim_pool
                    input_xyij = input_xy[i, j]
                    if input_xyij > output_xy[m, n]:
                        output_xy[m, n] = input_xyij

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pool_backward(np.ndarray[np.float_t, ndim=4] dJ_dout, np.ndarray[np.float_t, ndim=4] input, np.ndarray[np.float_t, ndim=4] output, int dim_pool, np.ndarray[np.float_t, ndim=4] dJ_din):
    '''
    ALTERS dJ_dout as a byproduct!
    '''
    dJ_din.fill(0)
    cdef int n_batch = input.shape[0]
    cdef int depth = input.shape[1]
    cdef int dim_input = input.shape[2]
    cdef int dim_output = output.shape[2]
    cdef int x, y # indices for batch and depth
    cdef int i, j # indices for one input slice
    cdef int m, n # indices for one output slice
    cdef np.float_t [:, :, :, :] input_buffer, output_buffer, dJ_din_buffer, dJ_dout_buffer
    input_buffer = input
    output_buffer = output
    dJ_din_buffer = dJ_din
    dJ_dout_buffer = dJ_dout
    cdef np.float_t [:, :] input_xy, output_xy, dJ_din_xy, dJ_dout_xy
    cdef bool found
    for x in range(n_batch):
        for y in range(depth):
            input_xy = input_buffer[x, y]
            output_xy = output_buffer[x, y]
            dJ_din_xy = dJ_din_buffer[x, y]
            dJ_dout_xy = dJ_dout_buffer[x, y]
            for i in range(dim_input):
                m = i // dim_pool
                for j in range(dim_input):
                    n = j // dim_pool
                    if input_xy[i, j] == output_xy[m, n]:
                        dJ_din_xy[i, j] = dJ_dout_xy[m, n]
                        dJ_dout_xy[m, n] = 0
