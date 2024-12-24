import numpy as np
import matplotlib.pyplot as plt

def extract_data(file_path):
    data = open(file_path, 'r')
    read = data.read()
    lines = read.splitlines()

    # lists to return
    freq_hz = []
    group_delay = []
    phase_deg = []

    for i in lines[1::]:
        values = i.split()
        freq_hz.append(float(values[0]))
        group_delay.append(float(values[1])*1000) # convert to micro seconds
        phase_deg.append(float(values[2]))

    return freq_hz, group_delay, phase_deg

woof_freq, woof_group_delay, _ = extract_data('VituixCAD_GroupDelay_SDS-P830657.txt')
tweet_freq, tweet_group_delay, _ = extract_data('VituixCAD_GroupDelay_ND25FW-4.txt')

# calculate the average value to determine how much more delay the tweeter has than the woofer in the xo region
woof_freq_array = np.array(woof_freq)
tweet_freq_array = np.array(tweet_freq)

fq_idx_strt = np.argmax(woof_freq_array > 1800)
fq_idx_end = np.argmax(woof_freq_array > 2200)

woof_avg = np.average(woof_group_delay[fq_idx_strt:fq_idx_end])
tweet_avg = np.average(tweet_group_delay[fq_idx_strt:fq_idx_end])

avg_diff = tweet_avg - woof_avg
print(avg_diff)

# plot group delay
plt.plot(woof_freq, woof_group_delay, label="woofer")
plt.plot(tweet_freq, tweet_group_delay, label='tweeter')
plt.xlim(1000, 2400)
plt.ylim(-500, 1000)
plt.axvspan(1800, 2200, color='blue', alpha=0.1, label="Crossover Region")
plt.hlines(woof_avg, 1800, 2200, color='black', linestyle='--', label='Average Woofer Delay in XO region')
plt.hlines(tweet_avg, 1800, 2200, color='green', linestyle='--', label='Average Tweeter Delay in XO region')
plt.xlabel('frequency, hz')
plt.ylabel('Group Delay, u"\u03bcs"')
plt.legend()
plt.title('Group Delay of Woofer and Tweeter')
plt.grid(which='both')
plt.show()