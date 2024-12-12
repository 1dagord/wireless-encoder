import numpy as np
import scipy.io.wavfile as wav

"""
    Decoder built for standard compliance
"""

SIZE = 25000
N = 500
K = 250

def sdec() -> np.array:
    Fs, Y = wav.read("encodedOutput.wav")
    Y = Y / np.iinfo(np.int32).max

    thresh = 0.1 * np.max(Y)
    start_index = np.where(np.abs(Y) > thresh)[0][0]
    end_index = (len(Y)) - np.where(Y[::-1] > thresh)[0][0]

    signal = Y[start_index:end_index]
    
    symbol = signal[:1752]
    symbol = symbol[K:]
    symbol = np.fft.fft(symbol, norm="ortho")
    symbol = np.abs(symbol)
    symbol = symbol[:len(symbol) // 2]
    
    out_bits = []
    
    shifted_signal = signal
    if len(signal) > 51 * 1752:
        shifted_signal = signal[:51 * 1752]
    elif len(signal) < 51 * 1752:
        shifted_signal = np.concatenate(
            (signal, np.zeros(np.abs(len(signal) - 51 * 1752)))
        )

    for chunk in np.reshape(shifted_signal, (51, 1752))[1:]:
        chunk = chunk[K:]
        chunk = np.fft.fft(chunk)
        chunk = np.abs(chunk)
        chunk = chunk[:len(chunk) // 2]
        chunk = chunk[:-1]
        chunk = chunk[:N]

        out_bits += [(chunk[i] > symbol[i] / 2) for i in range(len(chunk))]
        
    out_bits = np.array(out_bits)

    # ----- Testing -----
    tmp = (np.iinfo(np.int32).max * out_bits).astype(np.int32)
    wav.write("testOutput.wav", 44100, tmp)
    # -------------------

    return out_bits


# ----- Testing -----
sdec()
# -------------------
