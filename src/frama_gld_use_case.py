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
FRAMA comparison use case on GLD (Gold Index) data from yfinance.
"""
import numpy as np
import importlib
from time import perf_counter

from frama_performance import frama_perf_torch, frama_perf


def fetch_gld_data(period='1y'):
    """Fetch GLD (gold index) closing prices from yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            'frama_gld_use_case requires yfinance. Install it with pip install yfinance'
        ) from exc

    gld = yf.download('GLD', period=period, progress=False)
    prices = np.asarray(gld['Close'].to_numpy(dtype=float).reshape(-1), dtype=float)
    dates = gld.index.to_numpy()
    return prices, dates


def apply_frama_numpy(prices, batch=20):
    """Apply FRAMA using NumPy version."""
    filt_numpy = frama_perf(prices, batch)
    return np.asarray(filt_numpy, dtype=float)


def apply_frama_torch(prices, batch=20):
    """Apply FRAMA using PyTorch version."""
    try:
        # Copy to make sure we pass a writable NumPy array to torch.as_tensor.
        filt_torch = frama_perf_torch(np.array(prices, copy=True), batch)
    except ImportError:
        print('PyTorch is not installed. Cannot run frama_perf_torch.')
        return None
    return filt_torch.detach().cpu().numpy()


def compare_outputs(filt_numpy, filt_torch):
    """Compute quality metrics between NumPy and PyTorch FRAMA outputs."""
    min_len = min(len(filt_numpy), len(filt_torch))
    np_ref = np.asarray(filt_numpy[:min_len], dtype=float)
    torch_ref = np.asarray(filt_torch[:min_len], dtype=float)
    diff = torch_ref - np_ref
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    return {
        'numpy': np_ref,
        'torch': torch_ref,
        'diff': diff,
        'mae': mae,
        'rmse': rmse,
        'max_abs': max_abs,
    }


def plot_gld_results(prices, filt_numpy, filt_torch, dates, batch, stats, savePlotTo=None):
    """Plot GLD prices and NumPy/PyTorch FRAMA outputs."""
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
        'numpy': '#e76f51',
        'torch': '#2a9d8f',
        'diff': '#8d5a97',
    }

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        dpi=130,
        facecolor=palette['figure_bg'],
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(palette['axes_bg'])
        ax.grid(True, color=palette['grid'], alpha=0.45, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(palette['grid'])
        ax.spines['bottom'].set_color(palette['grid'])
        ax.tick_params(colors=palette['text'])

    min_len = min(len(stats['numpy']), len(stats['torch']))
    x_vals = np.arange(min_len)
    # FRAMA output indices start at the beginning of the input series.
    aligned_prices = prices[:min_len]
    aligned_dates = dates[:min_len]

    ax1.plot(x_vals, aligned_prices, label='GLD closing price', linewidth=2.4, color=palette['price'], alpha=0.7)
    ax1.plot(x_vals, stats['numpy'], label='FRAMA (NumPy)', linewidth=2.0, color=palette['numpy'])
    ax1.plot(x_vals, stats['torch'], label='FRAMA (PyTorch)', linewidth=2.0, linestyle='--', color=palette['torch'])
    ax1.set_ylabel('Price ($)', color=palette['text'])
    ax1.set_title(
        f'GLD FRAMA Comparison (batch={batch}) | MAE={stats["mae"]:.6f}, RMSE={stats["rmse"]:.6f}',
        color=palette['title'],
        fontsize=13,
        fontweight='bold',
    )
    ax1.legend(frameon=True, facecolor=palette['axes_bg'], edgecolor=palette['grid'], loc='best')

    ax2.plot(x_vals, stats['diff'], label='PyTorch - NumPy', linewidth=1.8, color=palette['diff'])
    ax2.axhline(0.0, color=palette['grid'], linewidth=1.0)
    ax2.set_ylabel('Diff', color=palette['text'])
    ax2.set_xlabel('Trading Days', color=palette['text'])
    ax2.legend(frameon=True, facecolor=palette['axes_bg'], edgecolor=palette['grid'], loc='best')

    # Show first/last date in x tick labels for context.
    if len(aligned_dates) > 1:
        tick_pos = np.linspace(0, len(aligned_dates) - 1, 6, dtype=int)
        tick_labels = [str(aligned_dates[i])[:10] for i in tick_pos]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, rotation=20, ha='right')

    fig.tight_layout()
    if savePlotTo is not None:
        plt.savefig(savePlotTo, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f'Plot saved to {savePlotTo}')
    else:
        plt.show()


def main():
    """Fetch GLD data, compare FRAMA implementations, and generate plot."""
    print('Fetching GLD data from yfinance...')
    prices, dates = fetch_gld_data(period='1y')
    print(f'Fetched {len(prices)} trading days of GLD data.')

    batch = 10
    print(f'Applying FRAMA (NumPy and PyTorch) with batch={batch}...')

    t0 = perf_counter()
    filt_numpy = apply_frama_numpy(prices, batch=batch)
    numpy_ms = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    filt_torch = apply_frama_torch(prices, batch=batch)
    torch_ms = (perf_counter() - t0) * 1000.0

    if filt_torch is not None:
        stats = compare_outputs(filt_numpy, filt_torch)
        speed_ratio = numpy_ms / torch_ms if torch_ms > 0 else float('inf')

        print('Comparison metrics:')
        print(f"  MAE      : {stats['mae']:.10f}")
        print(f"  RMSE     : {stats['rmse']:.10f}")
        print(f"  Max |diff|: {stats['max_abs']:.10f}")
        print('Runtime (single run):')
        print(f'  NumPy   : {numpy_ms:.3f} ms')
        print(f'  PyTorch : {torch_ms:.3f} ms')
        print(f'  Speed ratio (NumPy/PyTorch): {speed_ratio:.3f}x')

        print('Generating plot...')
        plot_gld_results(
            prices,
            filt_numpy,
            filt_torch,
            dates,
            batch,
            stats,
            savePlotTo='images/gld_frama_compare.png',
        )
        print('Done!')
    else:
        print('FRAMA filtering failed. Exiting.')


if __name__ == '__main__':
    main()
