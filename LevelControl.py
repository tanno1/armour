import numpy as np
import matplotlib.pyplot as plt

def extract_data(file_path):
    data = open(file_path, 'r')
    read = data.read()
    lines = read.splitlines()

    # lists to return
    freq_hz = []
    db = []
    phase_deg = []

    for i in lines[1::]:
        values = i.split()
        freq_hz.append(float(values[0]))
        db.append(float(values[1])-87)
        phase_deg.append(float(values[2]))

    return freq_hz, db, phase_deg

woof_freq, woof_db, _ = extract_data('woofer\SDS-P830657_Extracted_Frequency_Response.txt')
tweet_freq, tweet_db, _ = extract_data('tweeter/ND25FW-4@0.frd')

# plot sound output
plt.semilogx(woof_freq, woof_db, label="woofer")
plt.semilogx(tweet_freq, tweet_db, label='tweeter')
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('Amplitude, dB')
plt.legend()
plt.title('Relative output of uncorrected Woofer and Tweeter')
plt.grid(which='both')
plt.show()

# plot relative output of baffle compensated woofer and tweeter
baff_corr_woof_db = np.load('woofer\corrected_spl_db.npy')
baff_corr_woof_freq = np.load('frequency_array.npy')
plt.semilogx(baff_corr_woof_freq, baff_corr_woof_db, label="woofer")
plt.semilogx(tweet_freq, tweet_db, label='tweeter')
plt.semilogx(tweet_freq, [i-11 for i in tweet_db], label='tweeter, 11dB attenuation')
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('Amplitude, dB')
plt.legend()
plt.title('Relative output of baffle corrected Woofer and Tweeter')
plt.grid(which='both')
plt.show()

# predicted system frequency response
final_woof = np.load('woofer\lpf_woof_db.npy')
woof_freq = np.load('woofer\lpf_woof_freq.npy')
final_tweet = np.load('tweeter\hpf_tweet.npy')
f_tweet_freq = np.load('tweeter\hpf_freq.npy')

plt.semilogx(woof_freq, final_woof, label="woofer")
plt.semilogx(f_tweet_freq, final_tweet, label='tweeter')
plt.xlim(20, 20000)
plt.ylim(-20, 15)
plt.xlabel('frequency, hz')
plt.ylabel('Amplitude, dB')
plt.legend()
plt.title('Predicted System Frequency Response')
plt.grid(which='both')
plt.show()