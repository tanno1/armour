import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.optimize import minimize
from scipy.signal import freqs

# load baffle shelf corrected array
dbl_spl = np.load('corrected_spl_db.npy')
freq_hz = np.load('frequency_array.npy')

# plt.semilogx(freq_hz, dbl_spl, label="Uncorrected Woofer")
# plt.xlim(20, 20000)
# plt.ylim(-20, 10)
# plt.xlabel('frequency, hz')
# plt.ylabel('SPL, dB')
# plt.legend()
# plt.title('Frequency Response of Woofer, and Corrected Woofer')
# plt.grid(which='both')
# plt.show()

# function to create a lr 4th order filter by combining two series 2nd order buttersworths
# we need to do this to optimize the corner frequency value and then design the filter in filter-pro

def lr_4th(filter_order, corner_freq, filter_type, freqs_hz):
    crit_freq = corner_freq * 2 * np.pi  # Convert Hz to rad/s
    num, den = butter(filter_order, crit_freq, filter_type, analog=True, output='ba')
    num_lr = np.polymul(num, num)  # Combine two Butterworth filters
    den_lr = np.polymul(den, den)
    _, h = freqs(num_lr, den_lr, freqs_hz * 2 * np.pi)
    return h

# make da lr_4th instance for a test,
h_desired = lr_4th(2, 1800, 'lowpass', freq_hz)
# make a transfer function
freq_rads = freq_hz * 2 * np.pi

# # print target and uncorrected woofer
# plt.semilogx(freq_hz, dbl_spl, label="Uncorrected Woofer")
# plt.semilogx(freq_hz, 20 * np.log10(np.abs(h_desired)), label="Linkwitz Riley 4th Order Filter")
# plt.semilogx(freq_hz, 20 * np.log10(np.abs(lr_4th(2, 2200, 'lowpass', freq_hz))), label="Linkwitz Riley 4th Order Filter")
# plt.semilogx(freq_hz, 20 * np.log10(np.abs(lr_4th(2, 1400, 'lowpass', freq_hz))), label="Linkwitz Riley 4th Order Filter")
# plt.axvline(x=1800, color='r', linestyle='--', label="1.8 kHz Corner Frequency")
# plt.xlim(20, 20000)
# plt.ylim(-20, 10)
# plt.xlabel('frequency, hz')
# plt.ylabel('SPL, dB')
# plt.legend()
# plt.title('Frequency Response of Woofer, and Corrected Woofer')
# plt.grid(which='both')
# plt.show()

# numerical solver, again, oh yay...
# convert db to linera magnitude
woofer_response = 10**(dbl_spl/ 20)
# define acoustic transfer function: uncorrected woofer * filter tf
def acoustic_tf(corner_freq):
    h_acoustic = lr_4th(2, corner_freq, 'lowpass', freq_hz)
    combined_response = woofer_response * h_acoustic
    # plt.semilogx(freq_hz, 20 * np.log10(combined_response), label=f"Combined Reponse")
    # plt.semilogx(freq_hz, 20 * np.log10(h_acoustic), label=f"Acoustic TF with {corner_freq}")
    # plt.semilogx(freq_hz, dbl_spl, label=f"Uncorrected Woofer")
    # plt.semilogx(freq_hz, 20 * np.log10(h_desired), label="Linkwitz Riley 4th Order Filter")
    # plt.grid(which='both')
    # plt.legend()
    # plt.xlim(20, 20000)
    # plt.ylim(-20, 10)
    # plt.show()

    return combined_response

# objective transfer function: want to adjust filter corner frequency until the -3 dB point of the function occurs at 1.8kHz
def objective(corner_freq):
    combined_response = acoustic_tf(corner_freq) # returns this in FREQUENCY RESPONSE
    # calcualte the -3 db point, see if it is near 1800 hz, -3 dB like we want
    combined_response_dB = 20 * np.log10(np.abs(combined_response))
    index_1800 = np.argmin(np.abs(freq_hz - 1800))
    mag_at_1800 = combined_response_dB[index_1800]
    error_mag = np.abs(mag_at_1800 + 3)
    print(f"With {corner_freq}, error = {error_mag}")
    return error_mag # retrn the error from 1.8 kHz

# do da optimizing
initial_guess = 400
result = minimize(objective, initial_guess, bounds=[(500, 2200)])
optimal_corner_freq = result.x[0]

# lets see the sauce
print(f"Optimal filter corner frequency: {optimal_corner_freq:.2f} Hz")

# Plot the results
h_optimal = lr_4th(2, optimal_corner_freq, 'lowpass', freq_hz)
combined_response = woofer_response * h_optimal

plt.semilogx(freq_hz, dbl_spl, label="Woofer Response")
plt.semilogx(freq_hz, 20 * np.log10(h_desired), label="Filter Response")
plt.semilogx(freq_hz, 20 * np.log10(h_optimal), label="Optimal filter repsonse")
plt.semilogx(freq_hz, (20 * np.log10(combined_response)), label=f"Acoustic Transfer Function with {optimal_corner_freq}")
plt.axvline(1800, color='red', linestyle='--', label="Target Corner Frequency (1.8 kHz)")
plt.xlim(20, 20000)
plt.ylim(-20, 10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('SPL (dB)')
plt.legend()
plt.title('Optimization of Filter Corner Frequency')
plt.grid(which='both')
plt.show()

combined_respose_db = 20 * np.log10(combined_response)

np.save('lpf_woof_db', combined_respose_db)
np.save('lpf_woof_freq', freq_hz)