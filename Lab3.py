import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, sqrt, cos, sin

def signal(n, Wm, N, step):
    signal = np.zeros(N, dtype=np.float)
    t = np.arange(0, N, 1, dtype=np.float)
    for i in range(n):
        A = random.random()
        fi = random.random()*2*np.pi
        w = Wm - Wm*i/n
        signal += A*np.sin(w*t+fi)
    return t, signal

def turn_coef(k, N):
    if k%N == 0: return 1
    angle = -2j*np.pi*k/N
    return np.exp(angle)

def DFT(signal):
    N = len(signal)
    table = np.empty((N,N), dtype=np.complex)
    res = np.zeros(N, dtype=np.complex)

    angle = 2*np.pi / N

    for p in range(N):
        for k in range(N):
            table[p][k] = np.complex(np.cos(angle*p*k), -np.sin(angle*p*k))

    for p in range(N):
        res[p] = 0
        for k in range(N):
            res[p] += signal[k]*table[p][k]
    return res

def FFT(signal):
    N = len(signal)
    if N <= 1:
        return signal
    divider = int(N/2)

    even_vals = np.zeros(divider, dtype=complex)
    odd_vals = np.zeros(divider, dtype=complex)

    for i in range(divider):
        even_vals[i] = signal[2*i]
        odd_vals[i] = signal[2*i+1]
        
    even_vals = FFT(even_vals)
    odd_vals = FFT(odd_vals)

    res = np.zeros(N, dtype=complex)
    for i in range(divider):
        res[i] = even_vals[i] + turn_coef(i, N) * odd_vals[i]
        res[i + divider] = even_vals[i] - turn_coef(i, N) * odd_vals[i]

    return res

if __name__ == "__main__":
    n = 6
    Wm = 1500
    N = 1024
    step = 0.0001
    t, signal1 = signal(n, Wm, N, step)

    signal1_complex = np.array(signal1, dtype=np.complex)

    spectr1 = np.empty(N, dtype=np.complex)
    spectr2 = np.empty(N, dtype=np.complex)
    spectr1 = DFT(signal1_complex)
    spectr2 = FFT(signal1_complex)
    N = [i for i in range(N)]

    fig1, (axis1) = plt.subplots(1)
    fig1.suptitle("Signals")
    axis1.set_ylabel("signal")
    axis1.plot(t, signal1)
    axis1.grid(True)

    fig2, (axis1, axis2) = plt.subplots(2)
    fig2.suptitle("DFT")
    axis1.set_ylabel("DFT value")
    axis1.set_xlabel("N")
    axis1.plot(N, spectr1)
    axis1.grid(True)
    axis2.set_ylabel("FFT value")
    axis2.set_xlabel("N")
    axis2.plot(N, spectr2)
    axis2.grid(True)

    plt.show()
