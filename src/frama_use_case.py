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
import numpy as np
import importlib

from frama_performance import frama_perf, frama_perf_torch


def create_input(length=10000, noise_scale=0.2, seed=123):
    """Create a synthetic signal with trend shift and additive white noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10 * np.pi, length)
    price = 2 * np.sin(x)
    price[int(length / 2):length] += 3
    noise = noise_scale * rng.standard_normal(price.shape[0])
    input_price = price + noise
    return price, input_price


def run_numpy_example(input_price, batch):
    """Usage example for fram_perf (NumPy implementation alias)."""
    return frama_perf(input_price, batch)


def run_torch_example(input_price, batch):
    """Usage example for frama_perf_torch (PyTorch implementation)."""
    try:
        filt_torch = frama_perf_torch(input_price, batch)
    except ImportError:
        return None
    return filt_torch.detach().cpu().numpy()


def plot_results(price, input_price, filt_numpy, filt_torch, noise_scale, batch, savePlotTo=None):
    """Plot original/observed signal and FRAMA outputs."""
    try:
        plt = importlib.import_module('matplotlib.pyplot')
    except ImportError:
        print('matplotlib is not installed. Skipping plot output.')
        return

    palette = {
        'figure_bg': '#f3efe7',
        'axes_bg': '#fff9ef',
        'title': '#2f3b52',
        'text': '#3a4253',
        'grid': '#d9cfbe',
        'price': '#264653',
        'input': '#6c757d',
        'numpy': '#e76f51',
        'torch': '#2a9d8f',
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
        figsize=(14, 5.5),
        dpi=130,
        facecolor=palette['figure_bg'],
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(palette['axes_bg'])
        ax.grid(True, color=palette['grid'], alpha=0.45, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(palette['grid'])
        ax.spines['bottom'].set_color(palette['grid'])
        ax.tick_params(colors=palette['text'])

    ax1.plot(price, label='real price', linewidth=2.8, color=palette['price'])
    ax1.plot(filt_numpy, label='estimated price (numpy)', linewidth=1.5, color=palette['numpy'])
    if filt_torch is not None:
        ax1.plot(
            filt_torch,
            label='estimated price (torch)',
            linewidth=1.9,
            linestyle='--',
            color=palette['torch'],
        )
    ax1.set_title('Target vs FRAMA', color=palette['title'], fontsize=11, fontweight='bold')
    ax1.legend(frameon=True, facecolor=palette['axes_bg'], edgecolor=palette['grid'])

    ax2.plot(input_price, label='price + noise', linewidth=2.6, color=palette['input'], alpha=0.9)
    ax2.plot(filt_numpy, label='estimated price (numpy)', linewidth=1.5, color=palette['numpy'])
    if filt_torch is not None:
        ax2.plot(
            filt_torch,
            label='estimated price (torch)',
            linewidth=1.9,
            linestyle='--',
            color=palette['torch'],
        )
    ax2.set_title('Noisy Signal vs FRAMA', color=palette['title'], fontsize=11, fontweight='bold')
    ax2.legend(frameon=True, facecolor=palette['axes_bg'], edgecolor=palette['grid'])

    fig.suptitle(
        'FRAMA under {:02d}% noise with batch = {}'.format(int(noise_scale * 100), batch),
        color=palette['title'],
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if savePlotTo is not None:
        plt.savefig(savePlotTo, facecolor=fig.get_facecolor(), bbox_inches='tight')
    else:
        plt.show()

def createPlots():
    batch = 20
    for noise_scale in [0.1, 0.2, 0.5]:
        price, input_price = create_input(length=10000, noise_scale=noise_scale)
        filt_numpy = run_numpy_example(input_price, batch)
        plot_results(price, input_price, filt_numpy, None, noise_scale, batch, savePlotTo=f'images/frama_plot_noise_{noise_scale}.png')
    
    noise_scale = 0.1
    for batch in [10, 50, 100]:
        price, input_price = create_input(length=10000, noise_scale=noise_scale)
        filt_numpy = run_numpy_example(input_price, batch)
        plot_results(price, input_price, filt_numpy, None, noise_scale, batch, savePlotTo=f'images/frama_plot_batch_{batch}.png')

def main():
    # User-configurable settings.
    batch = 100
    price, input_price = create_input(length=10000, noise_scale=0.2)

    # Usage example 1: NumPy FRAMA.
    filt_numpy = run_numpy_example(input_price, batch)

    # Usage example 2: PyTorch FRAMA (if torch is installed).
    filt_torch = run_torch_example(input_price, batch)
    if filt_torch is None:
        print('PyTorch is not installed. NumPy FRAMA example ran successfully.')

    plot_results(price, input_price, filt_numpy, filt_torch, 0.2, batch, savePlotTo='images/frama_example_plot.png')
    createPlots()

if __name__ == '__main__':
    main()
