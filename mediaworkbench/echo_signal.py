"""Identify a signed short acoustic path from an isolated reference passage."""
import numpy as np

TAPS = 256
CHUNK = 2048
# Playback references are frequently self-correlated (held notes, hum, a
# repeated phrase).  Fit every causal tap jointly and pick a small ridge by
# two-fold held-out error instead of trusting the single strongest lag.
RIDGES = (1e-9, 1e-7, 1e-5, 1e-3, 1e-1)
MIN_EXPLAINED = .5
MIN_DIRECTIONS = 4


def read(handle, first, count, total):
    result = np.zeros((count, 2), dtype=np.float64)
    a, b = max(0, first), min(total, first + count)
    if a < b:
        handle.seek(a * 8)
        raw = handle.read((b-a) * 8)
        if len(raw) != (b-a)*8:
            raise ValueError('truncated echo source')
        result[a-first:b-first] = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite echo source')
    return result


def reference_rows(handle, first, count, total):
    # Row n holds ref[n], ref[n-1], ..., ref[n-TAPS+1] at the already aligned
    # original reference coordinate.  This is the causal, signed short path.
    values = read(handle, first-TAPS+1, count+TAPS-1, total)[:, 0]
    return np.lib.stride_tricks.sliding_window_view(values, TAPS)[:, ::-1]


def _explained(power, cross, gram, kernel):
    residual = power-2*np.sum(kernel*cross, axis=0)+np.einsum(
        'ij,jk,ik->i', kernel.T, gram, kernel.T)
    return np.maximum(residual, 0)


def _factor(gram):
    values, vectors = np.linalg.eigh(gram)
    largest = float(values[-1])
    if largest <= 0 or not np.isfinite(largest):
        raise ValueError('calibration reference lacks independent excitation')
    return values, vectors, largest


def _solve(factor, cross, ridge):
    # One eigendecomposition per Gram serves every candidate regularization:
    # a ridge shifts the eigenvalues, not the tap-space directions.
    values, vectors, largest = factor
    return vectors @ ((vectors.T @ cross)/(values+ridge*largest)[:, None])


def excitation(factor):
    """Reject a trial that cannot distinguish the taps of a short path.

    Silence leaves no equations at all.  One steady tone (even with a
    constant offset) occupies only a couple of lag directions, so infinitely
    many signed short paths fit it equally well; the calibration must say so
    loudly rather than invent a path or a spectral notch.  Broadband
    material, including strongly correlated or coloured references, spans
    many directions and is accepted.
    """
    values, _, largest = factor
    if largest <= 0 or not np.isfinite(largest):
        raise ValueError('calibration reference lacks independent excitation')
    # A pure tone (or a constant offset) is not exactly rank deficient in a
    # finite window: it leaks a floor of low-energy directions.  Look for the
    # number of dominant directions before that floor, not for exact numerical
    # support.  One tone spans two directions and adding a constant offset
    # spans three; at least four independent directions are needed to fix a
    # short path.
    floor = float(values[0])
    span = largest-floor
    if span <= 0 or not np.isfinite(span):
        raise ValueError('calibration reference lacks independent excitation')
    rank = int(np.sum((values-floor) > span*1e-2))
    if rank < MIN_DIRECTIONS:
        raise ValueError('calibration reference lacks independent excitation')
    return rank


def identify(reference, microphone, first, stop, offset, ref_total, mic_total):
    if stop <= first:
        raise ValueError('calibration interval is empty')
    gram_fit = [np.zeros((TAPS, TAPS)), np.zeros((TAPS, TAPS))]
    cross_fit = [np.zeros((TAPS, 2)), np.zeros((TAPS, 2))]
    power_fit = [np.zeros(2), np.zeros(2)]
    # Alternate whole chunks so both folds cover the complete trial rather than
    # an arbitrary prefix and suffix of the material.
    for index, cursor in enumerate(range(first, stop, CHUNK)):
        count = min(CHUNK, stop-cursor)
        rows = reference_rows(reference, cursor+offset, count, ref_total)
        observed = read(microphone, cursor, count, mic_total)
        fold = index % 2
        gram_fit[fold] += rows.T @ rows
        cross_fit[fold] += rows.T @ observed
        power_fit[fold] += np.sum(observed*observed, axis=0)
    gram_all = gram_fit[0]+gram_fit[1]
    cross_all = cross_fit[0]+cross_fit[1]
    power_all = power_fit[0]+power_fit[1]
    if not np.isfinite(gram_all).all() or not np.isfinite(power_all).all():
        raise ValueError('calibration reference is unusable')
    trace = float(np.trace(gram_all))
    if trace < 1e-12*TAPS or float(np.max(power_all)) < 1e-12:
        raise ValueError('calibration reference lacks independent excitation')
    # Silence and a few steady tones cannot distinguish the taps: they leave
    # the normal equations rank deficient no matter how loud they are.
    factor_all = _factor(gram_all)
    excitation(factor_all)
    # Choose the regularization by cross-validation, then fit the complete
    # trial with it.  This explains every reflection of the passage instead of
    # letting one loud lag absorb the correlation of a self-similar reference.
    used = [fold for fold in range(2) if np.trace(gram_fit[fold]) > 0]
    if not used:
        raise ValueError('calibration does not describe an isolated stationary echo')
    factors = {fold: _factor(gram_fit[fold]) for fold in used}
    best = None
    for ridge in RIDGES:
        error = 0.0
        for fold in used:
            other = 1-fold
            kernel = _solve(factors[fold], cross_fit[fold], ridge)
            error += float(np.sum(_explained(
                power_fit[other], cross_fit[other], gram_fit[other], kernel)))
        if not np.isfinite(error):
            continue
        if best is None or error < best[0]:
            best = (error, ridge)
    if best is None:
        raise ValueError('calibration does not describe an isolated stationary echo')
    # Once regularization has selected the path structure, use the complete
    # trial to explain its amplitudes.  This keeps both signs of every
    # reflection and does not need a spectral notch that would also remove
    # simultaneous speech at the same frequencies.
    kernel = _solve(factor_all, cross_all, best[1])
    residual = _explained(power_all, cross_all, gram_all, kernel)
    # A microphone channel without playback stays at zero; every channel that
    # did record playback must be explained by the same aligned reference over
    # the whole trial.
    floor = max(float(np.max(power_all))*1e-4, 1e-20)
    explained = []
    for channel in range(2):
        if power_all[channel] <= floor:
            kernel[:, channel] = 0.0
            continue
        explained.append(1-float(residual[channel])/max(float(power_all[channel]), 1e-30))
    if not explained or min(explained) < MIN_EXPLAINED:
        raise ValueError('calibration does not describe an isolated stationary echo')
    return kernel.tolist()


def blocks(microphone, reference, clip, profile, first, stop, mic_total, ref_total, block_frames):
    kernel = np.asarray(profile['kernel'], dtype=np.float64)
    if kernel.shape != (TAPS, 2) or not np.isfinite(kernel).all():
        raise ValueError('damaged echo calibration')
    offset = profile['offset_samples']
    if (type(offset) is not int or type(first) is not int or type(stop) is not int
            or type(block_frames) is not int or block_frames < 1 or not 0 <= first < stop):
        raise ValueError('damaged echo calibration')
    gain = clip['gain']
    origin = clip['start_ms']*48
    cursor = first
    # Chunks are anchored to the original microphone sample clock, so a given
    # output sample is always computed from the same inputs regardless of the
    # cut, audition start, block size or repeated occurrence.
    base = (origin+first)//CHUNK*CHUNK
    pending = []
    filled = 0
    while cursor < stop:
        observed = read(microphone, base, CHUNK, mic_total)
        rows = reference_rows(reference, base+offset, CHUNK, ref_total)
        # The dry voice is untouched: subtract only the signed reference path
        # after its short room response, keeping both channels together.
        pcm = observed-rows@kernel
        left = origin+cursor-base
        right = min(stop, cursor+(base+CHUNK-origin-cursor))
        pending.append(pcm[left:left+(right-cursor)])
        filled += right-cursor
        cursor = right
        base += CHUNK
        if filled >= block_frames:
            joined = np.concatenate(pending)
            while filled >= block_frames:
                yield (joined[:block_frames]*gain).astype('<f4')
                joined = joined[block_frames:]
                filled -= block_frames
            pending = [joined] if len(joined) else []
    if filled:
        yield (np.concatenate(pending)*gain).astype('<f4')
