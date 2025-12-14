import optuna
import numpy as np
import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())

from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile
from simulations_codes.research_copy.initialize_AO_research_copy import initialize_AO_hardware
from simulations_codes.research_copy.close_loop_ZWFS import close_loop
from simulations_codes.read_ao_results_file import parse_text_file, get_best_value

import matplotlib
matplotlib.use("Agg")  # non-GUI backend, safe for multiprocessing


def objective(trial):
    """
    Optuna objective for tuning ZWFS parameters.
    """

    # --- Suggest values for the three ZWFS parameters ---
    avg_time  = trial.suggest_int("avg_time", 1, 30)       # [s], e.g. 2-30
    freq      = trial.suggest_int("freq", 1, 10)           # [Hz], e.g. 1-10
    n_average = trial.suggest_int("n_average", 1, 10)      # number of segment coefficients to average
    
    # --- Load AO system ---
    param = initializeParameterFile()
    KAO   = initialize_AO_hardware(param)
    
    system = f"sh{param['nSubaperture']}"   # could also be "xin", "sh28", etc.
    r0 = param['r0']*100            # cm
    magnitude = KAO.param['magnitude_guide']      # guide star magnitude
    KAO.param['Jitter'] = False
    #'''
    try:
        path = '/home/mcisse/keckAOSim/keckSim/data/'
        filename = f"{path}summary_results.txt"   # your text file
        data = parse_text_file(filename)
        gain_result = get_best_value(data, system, r0, magnitude, "gain")
        KAO.param['gainCL'] = gain_result["Best_Gain"]
        #freq_result = get_best_value(data, system, r0, magnitude, "frequency")
        #KAO.tel.samplingTime = 1/freq_result["Best_Frequency"]
        print(f"Updated closed-loop gain = {KAO.param['gainCL']}")# and frequency to {freq_result['Best_Frequency']} Hz from results file")
    except ValueError as e:
        print("Gain lookup failed:", e)
    #'''


    # --- Build ZWFS parameter dictionary ---
    ZWFS_param = {
        "activate": True,
        "max_CLiter": 5,
        "gain": 1,
        "n_average": n_average,
        "avg_time": avg_time,
        "freq": freq,
        "subGZ": 4,
        "maxZ": 1,
        "maxGlobalZ": 4
    }

    # --- Run AO loop (this is the heavy step) ---
    #KAO.param['nLoop'] = 70000
    AO_output = close_loop(KAO, ZWFS_param)
    print(f" Trial {trial.number} done")

    # --- Extract time series ---
    SR        = np.array(AO_output["SR"])        # Strehl ratio per iteration
    M1_OPD    = np.array(AO_output["M1_OPD"])    # nm RMS per iteration
    residual  = np.array(AO_output["residual"])  # WFE residual per iteration
    
    # save full vectors in a per-trial file
    os.makedirs("zwfs_results", exist_ok=True)
    np.savez(f"zwfs_results/trial_{trial.number}.npz",
        AO_output,params=ZWFS_param) 
    # --- Define optimization score ---
    # Example: maximize the *mean SR over the last 90%* of the simulation
    cutoff = int(0.1 * len(SR))
    score = np.mean(SR[cutoff:])
    score_OPD = -np.mean(M1_OPD) 

    # --- Store extra metrics for later inspection ---
    trial.set_user_attr("final_SR", float(score))
    trial.set_user_attr("final_M1_OPD", float(M1_OPD[-1]))
    trial.set_user_attr("mean_residual", float(np.mean(residual)))
    trial.set_user_attr("avg_time", avg_time)
    trial.set_user_attr("freq", freq)
    trial.set_user_attr("n_average", n_average)

    return score,score_OPD

if __name__ == "__main__": 
    """   
    study = optuna.create_study(
        study_name="zwfs_tuning",
        storage="sqlite:///optuna_zwfs.db",
        load_if_exists=True,
        directions=["maximize", "maximize"]
        )

    study.optimize(objective, n_trials=20, n_jobs=5)

    
    
    """
    # if it crashes
    study = optuna.load_study(
        study_name="zwfs_tuning",
        storage="sqlite:///optuna_zwfs.db"
        )
    study.optimize(objective, n_trials=10, n_jobs=5)

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)

    """
    from optuna.visualization import plot_optimization_history, plot_param_importances
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    """
    
    
    
    
    
    
    
    
    
