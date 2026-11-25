"""Frequency-blocked Fourier POCS over two or four spatial dimensions.

The observed mask comes from trace presence, never from sample amplitudes.
This is binned Fourier sparsity reconstruction, not irregular NUFFT, MWNI or
wave-equation interpolation. Global periodic Fourier support is assumed.
"""
from math import prod
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import fft


def reconstruction_budget(shape, nt, nknown, frequency_block):
    count, nf = prod(shape), nt//2+1
    # Include input/output copies, temporal FFT, spatial work buffers and headers.
    return int(count*nt*12 + nknown*nf*24 + count*min(nf, frequency_block)*96 + count*384)


def reconstruct_survey(observed, indices, shape, *, iterations=80, final_threshold=.01,
                       frequency_block=8, memory_gb=2., work_dir=None, report=None):
    """Return all grid traces, retaining occupied bins exactly (float32).

    ``indices`` uniquely locate the observed rows in C-order flattened spatial
    ``shape``. Time is the last dimension. The full complex output spectrum is
    disk backed; only frequency blocks are Fourier transformed in memory.
    """
    observed = np.asarray(observed)
    indices = np.asarray(indices)
    shape = tuple(shape)
    if len(shape) not in (2, 4) or any(int(n) != n or n < 1 for n in shape):
        raise ValueError("use two or four positive integer spatial grid dimensions")
    shape = tuple(map(int, shape)); count = prod(shape)
    if (observed.ndim != 2 or not len(observed) or observed.shape[1] < 2 or
            not np.isfinite(observed).all() or indices.shape != (len(observed),) or
            not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("give finite observed traces and one integer bin index per trace")
    if indices.min() < 0 or indices.max() >= count or len(np.unique(indices)) != len(indices):
        raise ValueError("observed bin indices must be unique and lie inside the grid")
    if (int(iterations) != iterations or iterations < 2 or int(frequency_block) != frequency_block or
            frequency_block < 1 or not 0 < final_threshold < 1 or not np.isfinite(memory_gb) or memory_gb <= 0):
        raise ValueError("invalid iteration count, threshold, frequency block or memory budget")
    nt, nf = observed.shape[1], observed.shape[1]//2+1
    budget = reconstruction_budget(shape, nt, len(observed), frequency_block)
    if budget > memory_gb*1024**3:
        raise ValueError(f"survey reconstruction needs approximately {budget/1024**3:.3f} GiB working memory; "
                         "crop/coarsen the grid, reduce frequency block or increase the budget")
    destination = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    destination.mkdir(parents=True, exist_ok=True)
    disk_bytes = count*nf*8
    if shutil.disk_usage(destination).free < disk_bytes*1.1:
        raise ValueError(f"survey spectrum needs {disk_bytes/1024**3:.3f} GiB temporary disk space")
    if report:
        report(0.)
    observed = np.asarray(observed, np.float32)
    observed_spectrum = fft.rfft(observed, axis=-1, workers=1)
    axes = tuple(range(len(shape)))
    with tempfile.TemporaryDirectory(prefix="seisdesk-pocs-", dir=destination) as directory:
        storage = np.memmap(Path(directory)/"spectrum.bin", mode="w+", dtype=np.complex64, shape=(count, nf))
        try:
            for begin in range(0, nf, int(frequency_block)):
                end = min(nf, begin+int(frequency_block))
                current = np.zeros((*shape, end-begin), np.complex64)
                current.reshape(count, -1)[indices] = observed_spectrum[:, begin:end]
                peak = np.max(np.abs(fft.fftn(current, axes=axes, workers=1)), axis=axes)
                if np.any(peak):
                    for iteration in range(int(iterations)):
                        if report:
                            report(.9*(begin+(end-begin)*iteration/iterations)/nf)
                        spectrum = fft.fftn(current, axes=axes, workers=1)
                        threshold = peak*final_threshold**(iteration/(iterations-1))
                        spectrum[np.abs(spectrum) < threshold] = 0
                        current = fft.ifftn(spectrum, axes=axes, workers=1)
                        current.reshape(count, -1)[indices] = observed_spectrum[:, begin:end]
                storage[:, begin:end] = current.reshape(count, -1)
                if report:
                    report(.9*end/nf)
            output = np.empty((count, nt), np.float32)
            for begin in range(0, count, 256):
                end = min(count, begin+256)
                output[begin:end] = fft.irfft(storage[begin:end], n=nt, axis=-1, workers=1)
                if report:
                    report(.9+.1*end/count)
            output[indices] = observed
        finally:
            storage.flush()
            del storage
    return output
