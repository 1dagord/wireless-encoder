import numpy as np
import scipy.io.wavfile as wav

"""
    Encoder built for standard compliance
"""

SIZE = 25000
N = 500
K = 250


def senc(inBits: np.array) -> None:
    splitBits = np.array(np.split(inBits, SIZE / N))
    
    # creates random bit stream to be sent across channel (for testing)
    # analogous to white noise
    def cmplxEnc(k):
        return 0.0586 * np.exp(1j * (k % 13) * (k % 33))

    x1 = np.array([cmplxEnc(k) for k in range(N)] + [0] * K)
    x0 = np.concatenate(([0], x1, [0], np.flip(np.conj(x1))))
    x0 = np.fft.ifft(x0, norm="ortho")
    x0 = x0.real

    # prepend cyclic prefix
    xTrain = np.hstack([x0[len(x0) - K:], x0])

    xMat = [xTrain]
    for chunk in splitBits:
        # shape: (500,)
        chunk = np.array([chunk[k] * cmplxEnc(k) for k in range(len(chunk))])

        # shape: (750,)
        chunk = np.concatenate((chunk, np.zeros(K)))

        # shape: (1502,)
        chunk = np.concatenate(([0], chunk, [0], np.flip(np.conj(chunk))))
        chunk = np.fft.ifft(chunk, norm="ortho")
        chunk = chunk.real

        # shape: (1752,)
        chunk = np.hstack([chunk[len(chunk) - K:], chunk])

        xMat.append(chunk)
        
    xMat = np.array(xMat)
    xMat = xMat.flatten()
    
    tmp = (np.iinfo(np.int32).max * xMat).astype(np.int32)
    wav.write("encodedInput.wav", 44100, tmp)
