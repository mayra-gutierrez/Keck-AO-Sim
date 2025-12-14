import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Folder where trial results are saved
results_folder = "/home/mcisse/keckAOSim/keckSim/zwfs_results/"
trial_files = sorted(glob.glob(os.path.join(results_folder, "trial_*.npz"))) #[f'{results_folder}trial_1.npz']#

for f in trial_files:
    aa = f.split('trial_')[1]
    trial_nb = aa.split('.npz')[0]
    data = np.load(f, allow_pickle=True)
    AO_output = data['arr_0'].item()  # AO_output dict
    ZWFS_param_loaded = data['params'].item() 
    
    SR = np.array(AO_output["SR"])
    M1_OPD = np.array(AO_output["M1_OPD"])
    
    # Plot SR evolution & M1_OPD evolution
    name_fig = f"Trials_{trial_nb}_ZWFS_total exposure_{ZWFS_param_loaded['avg_time']}s_fps_{ZWFS_param_loaded['freq']}Hz_Nrepeats_{ZWFS_param_loaded['n_average']}.png"
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    axs = axs.ravel()
    fig.suptitle(f"Trials {trial_nb} ZWFS: total exposure={ZWFS_param_loaded['avg_time']}s, fps = {ZWFS_param_loaded['freq']}Hz, Nrepeats = {ZWFS_param_loaded['n_average']}")   
    ax_sr = axs[0]
    ax_sr.plot(SR)
    ax_M1 = axs[1]
    ax_M1.plot(M1_OPD,'ko')

    ax_sr.set(xlabel='AO Loop Iteration', ylabel='Strehl Ratio')
    ax_sr.set_title("Strehl Evolution Across Trials")

    ax_M1.set(xlabel='ACS commands', ylabel='M1 OPD [nm] RMS')
    ax_M1.set_title("M1 Amplitude Across Trials")

    fig.tight_layout()
    fig.savefig(f"{results_folder}{name_fig}", dpi=300)
    plt.show(block=False)

