import numpy as np


##Sampling Rate and Duration
sampling_rate = 250 #Hz
duration = 10 #seconds

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

## Gamma, a (12,5) matrix to modify the five waves' amplitudes of 12 leads (P, Q, R, S, T)
gamma = np.array([[1, 0.1, 1, 1.2, 1],
                   [2, 0.2, 0.2, 0.2, 3],
                   [1, -0.1, -0.8, -1.1, 2.5],
                   [-1, -0.05, -0.8, -0.5, -1.2],
                   [0.05, 0.05, 1, 1, 1],
                   [1, -0.05, -0.1, -0.1, 3],
                   [-0.5, 0.05, 0.2, 0.5, 1],
                   [0.05, 0.05, 1.3, 2.5, 2],
                   [1, 0.05, 1, 2, 1],
                   [1.2, 0.05, 1, 2, 2],
                   [1.5, 0.1, 0.8, 1, 2],
                   [1.8, 0.05, 0.5, 0.1, 2]])

## Normal ECG Parameters
Anoise = 0.01

#heart rate
mu_hr_1 = 60          # mean of the heart rate
sigma_hr_1 = 7        # variance of the heart rate

#noise
min_noise_1 = 0.01      
max_noise_1 = 0.1

#For PQRST five spikes
# t, the starting position along the circle of each interval in radius
mu_t_1 = np.array((-70, -15, 0, 15, 100))  
sigma_t_1 = np.ones(5)*3

# a, the amplitude of each spike; b, the width of each spike
mu_a_1 = np.array((1.2, -5, 30, -7.5, 0.75))
mu_b_1 = np.array((0.25, 0.1, 0.1, 0.1, 0.4))
sigma_a_1 = np.abs(mu_a_1/5)
sigma_b_1 = np.abs(mu_b_1/5)

## Abnormal ECG Parameters
#heart rate
mu_hr_2 = 60          # mean of the heart rate
sigma_hr_2 = 7        # variance of the heart rate

#noise
min_noise_2 = 0.01      
max_noise_2 = 0.1

#t, a, b
mu_t_2 = np.array((-70, -15, 0, 15, 100))
mu_a_2 = np.array((1.2, -4, 25, -6.5, 0.75))
mu_b_2 = np.array((0.25, 0.1, 0.1, 0.1, 0.4))
sigma_t_2 = np.ones(5)*3
sigma_a_2 = np.abs(mu_a_1/5)
sigma_b_2 = np.abs(mu_b_1/5)

