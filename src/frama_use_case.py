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
from frama_numpy_performance import frama_perf
import numpy as np
from matplotlib import pyplot as plt

# Create input, user can define its own input
Length = 10000
x = np.linspace(0,10 * np.pi, Length)
Price = 2 * np.sin(x)
Price[int(Length/2):Length] += 3
Noise = 0.2 * np.random.randn(Price.shape[0]) # white noise
InputPrice = Price + Noise
batch = 100

Filt = frama_perf(InputPrice, batch)

# plot the result to figure out the difference
# beween it (Filt) and the desired outcome (Price)

fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

ax1.plot(Price, label='real price', linewidth=3.0)
ax1.plot(Filt, label='estimated price', linewidth=1.0)
leg1 = ax1.legend()
ax2.plot(InputPrice, label='price + noise', linewidth=3.0)
ax2.plot(Filt, label='estimated price', linewidth=1.0) 
leg2 = ax2.legend()

fig.suptitle(('FRAMA under 10% noise and batch = 100'))
# plt.savefig('../images/estimation_example5.png')
plt.show()
