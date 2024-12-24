# tanner, noah
# 12/17/24
# baffle step compensation calculator

# description: a numerical solver algorithim to compare woofer frequency response to a target response within a frequency ranmge, and determine passive component values to create the flattest response

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as opt

# define f vs db file location
file_path = './SDS-P830657_Extracted_Frequency_Response.txt'

# open file
data = open(file_path, 'r')

# read file
read = data.read()

# split into lines
lines = read.splitlines()

# grab header column
header = lines[0]

# define empty vars for the freq, dbSPL, and phase
freq_hz = []
dbl_sp = []
phase_deg = []

# skip header line
data_lines = lines[1:]

# iterate through lines and add each value to its corresponding column
for i in data_lines:
    values = i.split()
    freq_hz.append(float(values[0]))
    dbl_sp.append(float(values[1])-87) # bump it down to around 0
    phase_deg.append(float(values[2]))

# plot dbSPl vs f
plt.semilogx(freq_hz, dbl_sp)
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('SPL, dB')
plt.title('Frequency Response Graph of Woofer')
plt.grid(which='both')
plt.show()

# defined target for dB level will be 87
target = np.zeros_like(dbl_sp)

# print target and uncorrected woofer
plt.semilogx(freq_hz, dbl_sp, label="Uncorrected Woofer")
plt.semilogx(freq_hz, target, label="Target")
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('SPL, dB')
plt.legend()
plt.title('Frequency Response Graph of Woofer and Target')
plt.grid(which='both')
plt.show()

# now, design the shelving circuit to compensate that, starting with the shelving circuit transfer function
def tf(R1, R2, R3, C1):
    num = [R2*R3*C1, R3]
    den = [R1*R3*C1 + R1*R2*C1, R1]
    H = signal.TransferFunction(num, den)
    return H

# error function used to minimize
target_spl = 0
def error_function(params):
    R1, R2, R3, C1 = params
    H = tf(R1, R2, R3, C1)
    # convert frequency to rad/s for bode() fn
    w_rad_s = 2 * np.pi * np.array(freq_hz)
    _, mag, _ = signal.bode(H, w=w_rad_s)
    # compute error
    error = np.sum((mag - target_spl)**2)
    return error

test_tf = tf(10e3, 10e3, 10e3, 1e-9)
w_rad_s = 2 * np.pi * np.array(freq_hz)
_, test_mag, _ = signal.bode(test_tf, w=w_rad_s)

plt.semilogx(freq_hz, test_mag, label="Test Transfer Function")
plt.xlabel('frequency, Hz')
plt.ylabel('Magnitude (dB)')
plt.title('Test Transfer Function Response')
plt.grid(which='both')
plt.legend()
plt.show()

# initial guess for R1, R2, R3, C1
initial_guess = [10000, 10000, 10000, 100e-9]

ic_tf = tf(initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3])
w_rad_s = 2 * np.pi * np.array(freq_hz)
_, ic_mag, _ = signal.bode(ic_tf, w=w_rad_s)
plt.semilogx(freq_hz, ic_mag, label="Initial Conditions Function")
plt.xlabel('frequency, Hz')
plt.ylabel('Magnitude (dB)')
plt.title('Initial Conditions Function Response')
plt.grid(which='both')
plt.legend()
plt.show()


# run scopy optimization 
optimization = opt.minimize(error_function, initial_guess)

# get the component values
R1_opt, R2_opt, R3_opt, C1_opt = optimization.x

# create optimized tf
optimized_tf = tf(R1_opt, R2_opt, R3_opt, C1_opt)
# Convert to angular frequency
w_rad_s = 2 * np.pi * np.array(freq_hz)
# process bode plot
_, mag, _ = signal.bode(optimized_tf, w=w_rad_s)

# print target and uncorrected woofer
plt.semilogx(freq_hz, dbl_sp, label="Uncorrected Woofer")
plt.semilogx(freq_hz, target, label="Target")
plt.semilogx(freq_hz, mag, label="Optimization")
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('SPL, dB')
plt.legend()
plt.title('Frequency Response Graph of Woofer and Target')
plt.grid(which='both')
plt.show()

# print those beautiful component values!
print(f"R1: {R1_opt}, R2: {R2_opt}, R3: {R3_opt}, C1: {C1_opt}")

# combine the optimized filter with the uncorrecetd woofer
corrected_spl_db = [dbl_sp[i] + mag[i] for i in range(len(dbl_sp))]

# print target and uncorrected woofer
plt.semilogx(freq_hz, dbl_sp, label="Uncorrected Woofer")
plt.semilogx(freq_hz, mag, label="Optimization")
plt.semilogx(freq_hz, corrected_spl_db, label="Corrected Woofer")
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('SPL, dB')
plt.legend()
plt.title('Frequency Response of Woofer, and Corrected Woofer')
plt.grid(which='both')
plt.show()

# save results to use in other files
np.save('corrected_spl_db', corrected_spl_db)
np.save('frequency_array', freq_hz)