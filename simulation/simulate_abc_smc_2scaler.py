import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import argparse 
import uuid
import sys 
sys.path.append("../models/")
from stochastic_SEIRD import simulate
from Basin import Basin
from functions import compute_contacts, get_epi_params, get_IFR_fixed
from deaths import compute_deaths
import pickle as pkl 
import os
import warnings
warnings.filterwarnings("ignore")

# n. of compartments and age groups
ncomp = 23
nage = 10

# simulation dates
start_date = datetime(2020, 10, 1)
end_date = datetime(2021, 10, 1)

def import_posterior(country):
    return pd.read_csv(f"../calibration_ABC-SMC/posteriors_2scaler/posterior_{country}.csv")


def simulation(basin_name, scenario, Nsim, sampled_params, vaccine, start_date, end_date=datetime(2021, 10, 1)):
    """
    This function runs the simulations for the calibrated model 
    :param basin_name: name of the basin
    :param scenario: vaccinations scenario
    :param Nsim: number of simulations 
    :param sampled_params: sampled parameters
    :param vaccine: vaccine strategy
    :param start_date: initial date of simulations
    :param end_date: last date of simulations
    :return: results of simulations
    """
    
    # create basin object and compute contacts 
    basin = Basin(basin_name, "../basins/")
    Cs, dates = compute_contacts(basin, start_date, end_date)
    
    deaths, deaths_scaled, incidence, incidence_VOC = [], [], [], []
      
    if scenario == 'data-driven':
        vaccinations = basin.vaccinations

    elif scenario == "us_rescale":
        vaccinations = basin.vaccinations_us_rescale
    elif scenario == "us_start":
        vaccinations = basin.vaccinations_us_start

    elif scenario == "eu_rescale":
        vaccinations = basin.vaccinations_eu_rescale
    elif scenario == "eu_start":
        vaccinations = basin.vaccinations_eu_start

    elif scenario == "isrl_rescale":
        vaccinations = basin.vaccinations_isrl_rescale
    elif scenario == "isrl_start":
        vaccinations = basin.vaccinations_isrl_start

    epi_params = get_epi_params()
    IFR = get_IFR_fixed("verity")
        
    for n in range(Nsim):
        
        params = sampled_params.iloc[n]
        date_VOC_intro = basin.start_date_VOC + timedelta(days=int(params.delta_days_intro_VOC))
        results = simulate(basin=basin, Cs=Cs, R0=params.R0, Delta=params.Delta, dates=dates, 
                           seasonality_min=params.seasonality_min, 
                           vaccine=vaccine, I_mult=params.I_mult, R_mult=params.R_mult, psi=params.psi, 
                           vaccinations=vaccinations, date_VOC_intro=date_VOC_intro)

        # compute deaths
        epi_params["Delta"] = int(params.Delta)
        epi_params["IFR"] = float(params.ifr_mult) * IFR
        epi_params["IFR_VOC"] = float(params.ifr_mult) * IFR

        results_deaths = compute_deaths(results["recovered"], results["recovered_VOC"], results["recovered_V1i"],
                                        results["recovered_V2i"], results["recovered_V1i_VOC"], results["recovered_V2i_VOC"], epi_params)
        
        # compute second scaler 
        scaler2 = params.scaler1 + params.delta_scaler2
        if scaler2 < 1.0:
            scaler2 = 1.0
        if scaler2 > 100:
            scaler2 = 99.0

        # rescale deaths and resample weekly
        deaths_temp = results_deaths["deaths_TOT"].sum(axis=0)
        l = len(deaths_temp)
        deaths_scl = np.concatenate((params.scaler1 / 100.0 * deaths_temp[:int(l/2)], scaler2 / 100.0 * deaths_temp[int(l/2):]))

        deaths.append(deaths_temp)
        deaths_scaled.append(deaths_scl)
        incidence.append(results["incidence"].sum(axis=0))
        incidence_VOC.append(results["incidence_VOC"].sum(axis=0))

    return {"incidence": incidence, 
            "incidence_VOC": incidence_VOC,
            "deaths": deaths,
            "deaths_scaled": deaths_scaled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--basin', type=str, help='name of the basin')
    args = parser.parse_args()

    sampled_params = import_posterior(args.basin)
    idxs = np.random.randint(0, len(sampled_params), size=100)
    unique_filename = str(uuid.uuid4())
    
    # simulate
    for keyword in ['data-driven', 'False', 'us_rescale', 'us_start', 'eu_rescale', 'isrl_rescale']:

        print(keyword)

        if keyword == 'False':
            scenario = 'data-driven'
            vaccine = 'False'
        else:
            scenario = keyword
            vaccine = 'age-order'

        results = simulation(args.basin, scenario, Nsim=100, sampled_params=sampled_params.iloc[idxs], vaccine=vaccine, start_date=start_date)
    
        # save
        if not os.path.exists("./projections_october_2scaler/projections_" + args.basin + "/"):
            os.system("mkdir ./projections_october_2scaler/projections_" + args.basin + "/")

        file_name = unique_filename + "_" + args.basin + "_scenario" + str(scenario) + "_vaccine" + str(vaccine)
        np.savez_compressed("./projections_october_2scaler/projections_" + args.basin + "/inc_" + file_name, results["incidence"])
        np.savez_compressed("./projections_october_2scaler/projections_" + args.basin + "/inc_VOC_" + file_name, results["incidence_VOC"])
        np.savez_compressed("./projections_october_2scaler/projections_" + args.basin + "/deaths_" + file_name, results["deaths"])
        np.savez_compressed("./projections_october_2scaler/projections_" + args.basin + "/deaths_scaled_" + file_name, results["deaths_scaled"])