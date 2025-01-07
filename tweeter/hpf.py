import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.optimize import minimize
from scipy.signal import freqs

# define file path
file_path = './tweeter/ND25FW-4@0.frd'

# open file
data = open(file_path, 'r')

# read file
read = data.read()

# split into lines
lines = read.splitlines()

# define empty vars for the freq, dbSPL, and phase
freq_hz = []
dbl_sp = []
phase_deg = []

# iterate through lines and add each value to its corresponding column
for i in lines:
    values = i.split()
    freq_hz.append(float(values[0]))
    dbl_sp.append(float(values[1])-87) # bump it down to around 0
    phase_deg.append(float(values[2]))

# conver to np arrays
freq_hz = np.array(freq_hz)
dbl_sp = np.array(dbl_sp) # -9, uncomment -9 

# plot dbSPl vs f
plt.semilogx(freq_hz, dbl_sp)
plt.xlim(300, 20000)
plt.ylim(-40, 20)
plt.xlabel('frequency, hz')
plt.ylabel('SPL, dB')
plt.title('Frequency Response Graph of Woofer')
plt.grid(which='both')
plt.show()

def lr_4th(filter_order, corner_freq, filter_type, freqs_hz):
    crit_freq = corner_freq * 2 * np.pi  # Convert Hz to rad/s
    num, den = butter(filter_order, crit_freq, filter_type, analog=True, output='ba')
    num_lr = np.polymul(num, num)  # Combine two Butterworth filters
    den_lr = np.polymul(den, den)
    _, h = freqs(num_lr, den_lr, freqs_hz * 2 * np.pi)
    return h

# make da lr_4th instance for a test,
h_desired = lr_4th(2, 1800, 'highpass', freq_hz)
# make a transfer function
freq_rads = freq_hz * 2 * np.pi

# numerical solver, again, oh yay...
# convert db to linera magnitude
woofer_response = 10**(dbl_sp/ 20)
# define acoustic transfer function: uncorrected woofer * filter tf
def acoustic_tf(corner_freq):
    h_acoustic = lr_4th(2, corner_freq, 'highpass', freq_hz)
    combined_response = woofer_response * h_acoustic

    return combined_response

# objective transfer function: want to adjust filter corner frequency until the -3 dB point of the function occurs at 1.8kHz
def objective(corner_freq):
    combined_response = acoustic_tf(corner_freq) # returns this in FREQUENCY RESPONSE
    # calcualte the -3 db point, see if it is near 1800 hz, -3 dB like we want
    combined_response_dB = 20 * np.log10(np.abs(combined_response))
    index_1800 = np.argmin(np.abs(freq_hz - 1800))
    mag_at_1800 = combined_response_dB[index_1800]
    error_mag = np.abs(mag_at_1800 - 3)
    print(f"With {corner_freq}, error = {error_mag}")
    return error_mag # retrn the error from 1.8 kHz

# do da optimizing
initial_guess = 1800
result = minimize(objective, initial_guess, bounds=[(1700, 2200)])
optimal_corner_freq = result.x[0]

# lets see the sauce
print(f"Optimal filter corner frequency: {optimal_corner_freq:.2f} Hz")

# Plot the results
h_optimal = lr_4th(2, optimal_corner_freq, 'highpass', freq_hz)
combined_response = woofer_response * h_optimal

plt.semilogx(freq_hz, dbl_sp - 9, label="Tweeter Response")
plt.semilogx(freq_hz, 20 * np.log10(h_desired), label="Filter Response")
plt.semilogx(freq_hz, 20 * np.log10(h_optimal), label="Optimal filter repsonse", linestyle='--')
plt.semilogx(freq_hz,20 * np.log10(combined_response) - 9, label="Acoustic Transfer Function")
plt.axvline(1800, color='red', linestyle='--', label="Target Corner Frequency (1.8 kHz)")
plt.xlim(300, 20000)
plt.ylim(-40, 20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('SPL (dB)')
plt.legend()
plt.title('Optimization of Filter Corner Frequency')
plt.grid(which='both')
plt.show()

tweet_hpf = 20 * np.log10(combined_response)

np.save('hpf_tweet', tweet_hpf)
np.save('hpf_freq', freq_hz)