import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.signal
import wave

##Functions
def butter_lowpass(cutoff, fs, signal, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    w, h = scipy.signal.freqz(b, a, worN=8000)
    Time=np.linspace(0, signal.size, signal.size) * (1/(2*fs))
    window = np.hanning(signal.size)
    
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Signal Wave...')
    plt.plot(Time,signal)
    plt.xlabel('Seconds');
    plt.ylabel('Amplitude');
    
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0,500)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    signal_filtered = scipy.signal.filtfilt(b,a,signal)

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(Time, signal, 'b-', label='Raw Data')
    plt.plot(Time, signal_filtered, 'g-', linewidth=2, label='Filtered Data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    
    fft_orig = scipy.fft(signal*window)
    P2_orig = abs(fft_orig/int(fft_orig.size))
    P1_orig = P2_orig[1:int(fft_orig.size/2+1)]
    P1_orig[2:-2] = 2*P1_orig[2:-2];
    
    fft_filtered = scipy.fft(signal_filtered*window)
    P2_filtered = abs(fft_filtered/int(fft_filtered.size))
    P1_filtered = P2_filtered[1:int(fft_filtered.size/2+1)]
    P1_filtered[2:-2] = 2*P1_filtered[2:-2];
    
    f_full = np.fft.fftfreq(int(fft_filtered.size), 1/fs)
    f = f_full[0:int(f_full.size/2)]

    #Plot FFT of original signal
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(2*f[0:200000],P1_orig[0:200000],'b-', label='Raw Data')
    plt.plot(2*f[0:200000],P1_filtered[0:200000],'g-', label='Filtered Data')
    plt.ylabel('|P1(f)|')
    plt.xlabel('f (Hz)')
    plt.title('Single-Sided Spectrum of Low Pass Filtered Wave File')

    return signal_filtered, f_full

def butter_highpass(cutoff, fs, signal, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    w, h = scipy.signal.freqz(b, a, worN=8000)
    Time=np.linspace(0, signal.size, signal.size) * (1/(2*fs))
    
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Signal Wave...')
    plt.plot(Time,signal)
    plt.xlabel('Seconds');
    plt.ylabel('Amplitude');
    
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0,500)
    plt.title("Highpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    signal_filtered = scipy.signal.filtfilt(b,a,signal)

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(Time, signal, 'b-', label='data')
    plt.plot(Time, signal_filtered, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    
    fft_orig = scipy.fft(signal)
    P2_orig = abs(fft_orig/int(fft_orig.size))
    P1_orig = P2_orig[1:int(fft_orig.size/2+1)]
    P1_orig[2:-2] = 2*P1_orig[2:-2];
    
    fft_filtered = np.fft.fft(signal_filtered)
    P2_filtered = abs(fft_filtered/int(fft_filtered.size))
    P1_filtered = P2_filtered[1:int(fft_filtered.size/2+1)]
    P1_filtered[2:-2] = 2*P1_filtered[2:-2];
    
    f_full = np.fft.fftfreq(int(fft_filtered.size), 1/fs)
    f = f_full[0:int(f_full.size/2)]

    #Plot FFT of original signal
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(2*f[0:200000],P1_orig[0:200000],'b-', label='Raw Data')
    plt.plot(2*f[0:200000],P1_filtered[0:200000],'g-', label='Filtered Data')
    plt.ylabel('|P1(f)|')
    plt.xlabel('f (Hz)')
    plt.title('Single-Sided Spectrum of High Pass Filtered Wave File')
    return signal_filtered, f_full

wave_file = wave.open('Path_to_file.wav','r') #insert path to your *.wav file

#Extract Raw Audio from Wav File
temp = wave_file.readframes(-1)

#Convert raw audio to an array we can plot using numpy and trim it to a managable size
signal = np.fromstring(temp, 'Int16')

#Get frame rate to plot against time
fs = wave_file.getframerate()

#If Stereo
if wave_file.getnchannels() == 2:
    print('Just mono files')

#Calculate time vector
Time=np.linspace(0, signal.size, signal.size) * (1/(2*fs))

cutoff_HP = 400
signal_HP, f_full_HP, = butter_lowpass(cutoff_HP, fs, signal)
plt.show()

#Save out filtered signal into *.wav file
signal_HP2 = np.asarray(signal_HP, dtype=np.int16)
import scipy.io.wavfile
scipy.io.wavfile.write('Path_to_file.wav', 2*fs, signal_HP2) #insert path for your final *.wav file
