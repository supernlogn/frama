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
in numpy following the example
by John Ehlers: http://www.stockspotter.com/Files/frama.pdf
This implementation is for educational purposes
"""

import numpy as np
from matplotlib import pyplot as plt

# Create input
Length = 10000
x = np.linspace(0,10 * np.pi, Length)
Price = 2 * np.sin(x)
Price[int(Length/2):Length] += 3
Noise = np.random.randn(Price.shape[0]) # white noise
InputPrice = Price + Noise
batch = 10

# Initialize output before the algorithm
Filt = np.array(InputPrice)

# sequencially calculate all variables and the output
for i in range(0, Length-batch, 1):
    # take 2 batches of the input
    v1 = InputPrice[i:i+batch]
    v2 = InputPrice[i+batch:i+2*batch]

    # for the 1st batch calculate N1
    H1 = np.max(v1)
    L1 = np.min(v1)
    N1 = (H1 - L1) / batch
    
    # for the 2nd batch calculate N2
    H2 = np.max(v2)
    L2 = np.min(v2)
    N2 = (H2 - L2) / batch
    
    # for both batches calculate N3    
    H = np.max([H1, H2])
    L = np.min([L1, L2])
    N3 = (H - L) / (2*batch)

    # calculate fractal dimension
    Dimen = 0
    if N1 > 0 and N2 > 0 and N3 > 0:
        Dimen = (np.log(N1 + N2) - np.log(N3)) / np.log(2)

    # calculate lowpass filter factor
    alpha = np.exp(-4.6*(Dimen - 1))
    alpha = np.max([alpha, 0.1])
    alpha = np.min([alpha, 1])    
    
    # filter the input data
    Filt[i+1] = alpha * InputPrice[i] + (1 - alpha) * Filt[i]
    # if currentBar < 2*batch + 1: <--- i dont get what these 2 lines do
        # Filt = InputPrice[i]

# plot the result to figure out the difference
# beween it (Filt) and the desired outcome (Price)
plt.plot(Price)
plt.plot(Filt)
plt.show()
