import sys 
sys.path.append("../models/")
from stochastic_SEIRD import simulate
from functions import get_IFR_fixed, get_epi_params, compute_contacts
from deaths import compute_deaths
from Basin import Basin
from abc_smc import wmape_pyabc, calibration
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from pyabc import RV, Distribution
import pyabc
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

start_date = datetime(2020, 10, 1)
end_date = datetime(2021, 10, 1)


def create_folder(country):
    if os.path.exists(f"./calibration_runs/{country}") == False:
        os.system(f"mkdir ./calibration_runs/{country}")
        os.system(f"mkdir ./calibration_runs/{country}/abc_history/")
        os.system(f"mkdir ./calibration_runs/{country}/dbs/")



def run_fullmodel(basin, Cs, R0, Delta, dates, seasonality_min, vaccine, I_mult, R_mult, psi, delta_days_intro_VOC, ifr_mult, scaler, exclude_n=35):

    # simulate
    date_intro_VOC = basin.start_date_VOC + timedelta(days=int(delta_days_intro_VOC))
    #print(R0, Delta, seasonality_min, psi, I_mult, R_mult, ifr_mult, scaler1, scaler2, delta_days_intro_VOC)
    results = simulate(basin, Cs, R0, Delta, dates, seasonality_min, vaccine, I_mult, R_mult, psi, basin.vaccinations, date_intro_VOC)
    
    # compute deaths
    IFR = get_IFR_fixed('verity')
    epi_params = get_epi_params()
    epi_params["IFR"] = ifr_mult * IFR
    epi_params["IFR_VOC"] = ifr_mult * IFR
    epi_params["Delta"] = int(Delta)

    results_deaths = compute_deaths(results["recovered"], results["recovered_VOC"], results["recovered_V1i"],
                                    results["recovered_V2i"], results["recovered_V1i_VOC"], results["recovered_V2i_VOC"], epi_params)

    # rescale deaths and resample weekly
    deaths_temp = results_deaths["deaths_TOT"].sum(axis=0)
    deaths_scl = scaler / 100.0 * deaths_temp
    df_deaths = pd.DataFrame(data={"sim_deaths": deaths_scl[exclude_n:]}, index=dates[exclude_n:])
    df_deaths = df_deaths.resample("W").sum()

    # accept/reject                                    
    return {"deaths": df_deaths.sim_deaths.values, "results": results}


def run_calibration(country):

    # create Basin object 
    basin = Basin(country, "../basins/")
    Cs, dates = compute_contacts(basin, start_date, end_date)


    # get real deaths (first month excluded for the delay Delta)
    real_deaths = basin.epi_data_deaths.loc[(basin.epi_data_deaths["date"] >= start_date) &
                                            (basin.epi_data_deaths["date"] < end_date)].iloc[60:]

    real_deaths.index = real_deaths.date
    real_deaths = real_deaths.resample("W").sum()

    history = calibration(run_fullmodel, 
                        prior=Distribution(
                                    R0=RV("uniform", 0.6, 2.0 - 0.6), 
                                    Delta=RV('rv_discrete', values=(np.arange(10, 36), [1. / 26.] * 26)),
                                    seasonality_min=RV("uniform", 0.5, 1.0 - 0.5),
                                    psi=RV("uniform", 1.0, 3.0 - 1.0),
                                    I_mult=RV("uniform", 1.0, 1000.0 - 1.0),
                                    R_mult=RV("uniform", 1.0, 100.0 - 1.0), 
                                    ifr_mult=RV("uniform", 0.5, 2.0 - 0.5), 
                                    scaler=RV("uniform", 1.0, 100.0 - 1.0), 
                                    delta_days_intro_VOC=RV('rv_discrete', values=(np.arange(-45, 46), [1. / 91.] * 91))),
                        params={
                                'basin': basin, 
                                'Cs': Cs,
                                'dates': dates,
                                'vaccine': 'age-order'}, 
                        distance=wmape_pyabc,
                        basin_name=country,
                        observations=real_deaths.daily.values,
                        transition = pyabc.AggregatedTransition(
                                            mapping={
                                                'delta_days_intro_VOC': pyabc.DiscreteJumpTransition(
                                                    domain=np.arange(-45, 46), p_stay=0.7
                                                ),
                                                'Delta': pyabc.DiscreteJumpTransition(
                                                    domain=np.arange(10, 36), p_stay=0.7
                                                ),
                                                'R0': pyabc.MultivariateNormalTransition(),
                                                'seasonality_min': pyabc.MultivariateNormalTransition(),
                                                'psi': pyabc.MultivariateNormalTransition(),
                                                'I_mult': pyabc.MultivariateNormalTransition(),
                                                'R_mult': pyabc.MultivariateNormalTransition(),
                                                'ifr_mult': pyabc.MultivariateNormalTransition(),
                                                'scaler': pyabc.MultivariateNormalTransition()
                                            }
                                        ), 
                            max_nr_populations=20,
                            population_size=1000,
                            max_walltime=timedelta(hours=24), 
                            minimum_epsilon=0.15,
                            filename='calibration_single')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country")

    args = parser.parse_args()
    create_folder(str(args.country))
    run_calibration(str(args.country))