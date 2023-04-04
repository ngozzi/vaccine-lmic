import numpy as np 
from typing import List
import pyabc
import os
import uuid
import pickle as pkl
from datetime import timedelta
from typing import Callable, List


def calibration(epimodel : Callable, 
                prior : pyabc.Distribution, 
                params : dict, 
                distance : Callable,
                observations : List[float],
                basin_name : str,
                transition : pyabc.AggregatedTransition,
                max_walltime : timedelta = None,
                population_size : int = 1000,
                minimum_epsilon : float = 0.15, 
                max_nr_populations : int = 10, 
                filename : str = '', 
                run_id = None, 
                db = None):

    """
    Run ABC calibration on given model and prior 
    Parameters
    ----------
        @param epimodel (Callable): epidemic model 
        @param prior (pyabc.Distribution): prior distribution
        @param params (dict): dictionary of fixed parameters value
        @param distance (Callable): distance function to use 
        @param observations (List[float]): real observations 
        @param model_name (str): model name
        @param basin_name (str): name of the basin
        @param transition (pyabc.AggregatedTransition): next gen. perturbation transitions
        @param max_walltime (timedelta): maximum simulation time
        @param population_size (int): size of the population of a given generation
        @param minimum_epsilon (float): minimum tolerance (if reached calibration stops)
        @param max_nr_population (int): maximum number of generations
        @param filename (str): name of the files used to store ABC results
        @param runid: Id of previous run (needed to resume it)
        @param db: path to dd of previous run (needed to resume it)

    Returns
    -------
        @return: returns ABC history
    """
    
    def model(p): 
        return {'data': epimodel(**p, **params)['deaths']}

    if filename == '':
        filename = str(uuid.uuid4())

    abc = pyabc.ABCSMC(model, prior, distance, transitions=transition, population_size=population_size)
    if db == None:
        db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"{filename}.db")
        abc.new("sqlite:///" + db_path, {"data": observations})

    else:
        abc.load(db, run_id)
        
    history = abc.run(minimum_epsilon=minimum_epsilon, 
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)
    
    with open(os.path.join(f'./calibration_runs/{basin_name}/abc_history/', f"{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)

    history.get_distribution()[0].to_csv(f"./posteriors/posterior_{basin_name}.csv")
    np.savez_compressed(f"./posteriors/posterior_samples_{basin_name}.npz", np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))
    
    return history


def wmape_pyabc(sim_data : dict, 
                actual_data : dict) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE) to use for pyabc calibration
    Parameters
    ----------
        @param actual_data (dict): dictionary of actual data
        @param sim_data (dict): dictionary of simulated data 
    Return
    ------
        @return: returns wmape between actual and simulated data
    """
    return np.sum(np.abs(actual_data['data'] - sim_data['data'])) / np.sum(np.abs(actual_data['data']))


def import_projections(model_name, run_name, basin_name): 
    """
    This function imports the calibration data for a given model
    Parameters
    ----------
        @param model_name: model name
        @param run_name: run name
        @basin_name: name of the basin
    Return
    ------
        @return: returns pyabc calibration history 
    """
    with open(f'./calibration_runs/{basin_name}/{model_name}/abc_history/{run_name}.pkl', 'rb') as file: 
        data = pkl.load(file)
    return data


def import_parameters(model_name, run_name, basin_name): 
    """
    This function imports the parameters sampled during the calibration for a given model
    Parameters
    ----------
        @param model_name: model name
        @param run_name: run name
        @basin_name: name of the basin
    Return
    ------
        @return: returns sampled parameters
    """
    with open(f'./calibration_runs/{basin_name}/{model_name}/abc_history/{run_name}.pkl', 'rb') as file: 
        data = pkl.load(file)
    params = data.get_distribution()[0]
    return params
