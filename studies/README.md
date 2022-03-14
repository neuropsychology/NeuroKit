# Generating Synthetic ECG Data

Each of our ECG data consists of 12 leads, with a duration of 10 seconds and a frequency of 250 Hz. Therefore, each sample is a data matrix of 12 by 2500. In our case, we worked on 2 sets
of data, one was synthetic, one was real clinical data. Although we cannot share our real clinical data, the simulation code is accessible. 

***Simulation Code***

The simlation part includes two files: **'ecg_simulation.py'** and **'parameters. py'**.

**Instruction**:
 - Install package **neurokit2**
 - Open **sample_code_for_simulation.ipynb** which is a short sample notebook to show the way to output synthetic data.
 - Function **simulation** is all you need.
    1.  Input:\
    **normal_N**: the number of normal ECG data;\
   **abnormal_N**: the number of abnormal ECG data;\
   **save_params**: whether save parameters for each ecg sample, default value is *False*.
    2. Output:\
        **sim_ecg_data.npy**: output file, `shape = (normal_N + abnormal_N, 12, sampling_rate*duration)`;\
        **sim_ecg_labels.npy**: output file to save labels, `shape = (normal_N + abnormal_N, )`;\
        **sim_ecg_params.npy**: depend on **save_params**, file to save parameters for each ecg sample, `shape = (normal_N + abnormal_N, )`.
The saved data is already **shuffled**.
 - If you want more customized ECG data, please check **parameters. py** file. All the parameters' definitions are in the following table (_1 stands for normal, _2 stands for abnormal):
 
|                |Parameter                           |Meaning                          |
|----------------|-------------------------------|-----------------------------|
|1|sampling_rate          |sampling rate, default 250 Hz          |
|2          |duration|default 10s|
 |3         |gamma|a (12,5) matrix to modify each lead's five spikes' amplitudes|
 |4          |mu_hr_1          |the mean of heart rate        |
|5          |sigma_hr_1          |the variance of heart rate        |
|6         |min_noise_1, max_noise_1        |the max value and min value of noise        |
|7|t           |the starting position along the circle of each interval in radius         |
|8          |a          |the amplitude of each spike; b, the width of each spike         |
|9          |b|the width of each spike|

For a better understanding of the above parameters, please read the following.
[<img src="./3D.png" width="500"/>]
[<img src="./table.png" width="500"/>](table)

 We can see from the above table that each interval in the 3D trajectory can be fixed by 3 parameters: the starting position 
<img src="https://render.githubusercontent.com/render/math?math=\theta_i/t_i "> along the circle of each interval in radius, The amplitude of each spike a and the width of each wave b. By altering these 3 parameters we can change the shape of the 3D trajectory and thus change the waveform of the resulting ECG. 


**Prebuilt Synthetic Data**
Prebuilt synthetic data can be found at the following link 
https://drive.google.com/drive/folders/1iqyAlyHAvNWdOvjEGn8Y6C8kItsdZ_GC?usp=sharing


