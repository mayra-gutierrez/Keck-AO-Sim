import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import welch

class RandomTipTilt_Gaussian:
    def __init__(self, freq, dt, n_frames, seed=None, nm_rms=1):
        self.freq = freq      # cutoff frequency in Hz
        self.dt = dt          # time step in seconds
        self.n_frames = n_frames
        self.seed = seed
        self.nm_rms = nm_rms
        self._generate_sequence()

    def _generate_sequence(self):
        nyquist = 0.5 / self.dt
        b, a = butter(N=2, Wn=self.freq / nyquist)

        if self.seed is not None:
            np.random.seed(self.seed)

        # White noise
        x_white = np.random.randn(self.n_frames)
        y_white = np.random.randn(self.n_frames)

        # Apply low-pass filter
        x_filtered = filtfilt(b, a, x_white)
        y_filtered = filtfilt(b, a, y_white)

        # Normalize to RMS in arcsec
        x_filtered *= self.nm_rms / np.std(x_filtered)
        y_filtered *= self.nm_rms / np.std(y_filtered)

        self.x = x_filtered
        self.y = y_filtered

    def get_tilt(self, frame_idx):
        return self.x[frame_idx], self.y[frame_idx]


class SinusoidalTipTilt:
    def __init__(self, freq, dt, n_frames, nm_rms=1):
        self.freq = freq                  # frequency of the oscillation [Hz]
        self.dt = dt                      # time step [s]
        self.n_frames = n_frames
        self.nm_rms = nm_rms                # RMS amplitude in nm
        self._generate_signal()

    def _generate_signal(self):
        t = np.arange(self.n_frames) * self.dt

        amplitude = self.nm_rms 
        self.x = amplitude * np.sin(2 * np.pi * self.freq * t)
        self.y = amplitude * np.cos(2 * np.pi * self.freq * t)

    def get_tilt(self, frame_idx):
        return self.x[frame_idx], self.y[frame_idx]
    
#%% Example usage
if __name__ == "__main__":
    n_frames = 1000
    dt = 1 / 1000  # e.g., 1 ms time step
    cutoff_freq = 30  # Hz
    tt_rms = 0.1  # arcsec RMS

    tiptilt_gen = RandomTipTilt_Gaussian(freq=cutoff_freq, dt=dt, n_frames=n_frames, nm_rms=tt_rms) 
    tiptilt = SinusoidalTipTilt(freq=cutoff_freq, dt=dt, n_frames=n_frames, nm_rms=tt_rms)
    
    def compute_psd(signal, dt, label='X Tilt'):
        fs = 1 / dt  # Sampling frequency
        f, Pxx = welch(signal, fs=fs, window='hann', nperseg=1024, scaling='density')
    
        plt.figure(figsize=(8, 4))
        plt.semilogy(f, Pxx)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [arcsec²/Hz]')
        plt.xlim(0, fs//10)
        plt.title(f'Power Spectral Density - {label}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return f, Pxx
    
    compute_psd(tiptilt_gen.x, dt=tiptilt_gen.dt, label='X Tilt')
    compute_psd(tiptilt.x, dt=tiptilt.dt, label='X Tilt Sine')

    t = np.arange(0, 10, dt)
    sine_signal = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

    # Plot PSD
    f, Pxx = welch(sine_signal, fs=1/dt, nperseg=2048)
    plt.semilogy(f, Pxx)
    plt.title("PSD of 50 Hz Sine Wave (for comparison)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.show()
    #for i in range(10):  # Print first 10 frames
     #   x_tilt, y_tilt = tiptilt_gen.get_tilt(i)
      #  print(f"Frame {i}: X tilt = {x_tilt:.4f} arcsec, Y tilt = {y_tilt:.4f} arcsec")
# %%
