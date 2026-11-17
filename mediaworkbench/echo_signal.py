"""Identify a signed short acoustic path from an isolated reference passage."""
import numpy as np

TAPS = 256


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
    # The reference is mono. Native decoding duplicates it to stereo.
    values = read(handle, first-TAPS+1, count+TAPS-1, total)[:, 0]
    return np.lib.stride_tricks.sliding_window_view(values, TAPS)[:, ::-1]


def identify(reference, microphone, first, stop, offset, ref_total, mic_total):
    correlation = np.zeros((TAPS, 2))
    reference_power = np.zeros(TAPS)
    observed_power = np.zeros(2)
    for cursor in range(first, stop, 1024):
        count = min(1024, stop-cursor)
        rows = reference_rows(reference, cursor+offset, count, ref_total)
        observed = read(microphone, cursor, count, mic_total)
        correlation += rows.T @ observed
        reference_power += np.sum(rows*rows, axis=0)
        observed_power += np.sum(observed*observed, axis=0)
    if reference_power.max() < 1e-8 or observed_power.sum() < 1e-10:
        raise ValueError('calibration reference lacks independent excitation')
    denominator = np.sqrt(np.maximum(reference_power[:, None]*observed_power, 1e-30))
    coherence = np.abs(correlation)/denominator
    delay = int(np.argmax(np.sum(coherence, axis=1)))
    if float(np.max(coherence[delay])) < .1:
        raise ValueError('calibration does not describe an isolated stationary echo')
    kernel = np.zeros((TAPS, 2))
    kernel[delay] = np.abs(correlation[delay])/reference_power[delay]
    return kernel.tolist()


def blocks(microphone, reference, clip, profile, first, stop, mic_total, ref_total, block_frames):
    kernel = np.asarray(profile['kernel'], dtype=np.float64)
    for cursor in range(first, stop, block_frames):
        count = min(block_frames, stop-cursor)
        origin = clip['start_ms']*48 + cursor
        observed = read(microphone, origin, count, mic_total)
        rows = reference_rows(reference, origin+profile['offset_samples'], count, ref_total)
        yield ((observed - rows@kernel)*clip['gain']).astype('<f4')
