# Copyright (c) 2017 Ioannis Athanasiadis(supernlogn)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
 Below is the frama code written
 in numpy and utilizing it for
 performance.
"""
__all__ = ['frama_perf']

import numpy as np

def frama_perf(InputPrice, batch):
    """
        frama with numpy for many datapoints
        InputPrice: The input time-series to estimate its frama per batch
        batch: The batch of datapoints, where the N1 and N2 are calculated
        See also: http://www.stockspotter.com/Files/frama.pdf 
    """
    Length = len(InputPrice)
    # calulcate maximums and minimums
    H = np.array([np.max(InputPrice[i:i+batch]) for i in range(0, Length-batch, 1)])
    L = np.array([np.min(InputPrice[i:i+batch]) for i in range(0, Length-batch, 1)])

    # set the N-variables
    b_inv = 1.0 / batch
    N12 = (H - L) * b_inv
    N1 = N12[0:-1]
    N2 = N12[1:]
    N3 = ( np.array([np.max(H[i:i+1]) for i in range(0, len(H) - 1)]) - np.array([np.min(L[i:i+1]) for i in range(0, len(H) - 1)]) ) / batch

    # calculate the fractal dimensions
    Dimen = np.zeros(N1.shape)
    Dimen_indices = np.bitwise_and(np.bitwise_and((N1 > 0), (N2 > 0)), (N3 > 0))
    lg2_inv = 1.0 / np.log2(2)
    d = (np.log2(N1 + N2) - np.log2(N3)) * lg2_inv
    Dimen[Dimen_indices] = d[Dimen_indices]

    # calculate the filter factor
    alpha = np.exp(-4.6 * (Dimen - 1))
    alpha = np.clip(alpha, 0.1, 1)


    Filt = np.array(InputPrice[0:len(alpha)])
    # Declare two variables to accelerate performance
    S = 1 - alpha
    A = alpha * InputPrice[0:len(alpha)]
    # This is the overheat of all the computation
    # where the filter applies to the input
    for i in range(0, Length-2*batch, 1):
        Filt[i+1] = A[i] + S[i] * Filt[i]
    return Filt

