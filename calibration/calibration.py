# libraries
import sys
sys.path.append("../models/")
from stochastic_SEIRD import simulate
from functions import get_epi_params, get_IFR_fixed,  wmape, compute_contacts
from deaths import compute_deaths
from Basin import Basin
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import time 
import argparse
import warnings
warnings.filterwarnings("ignore")

# n. of compartments and age groups
ncomp = 23
nage = 10

# simulation dates
end_date = datetime(2021, 12, 1)



def calibration(basin_name, nsim, step_save, R0min, R0max, Imultmin, Imultmax, Rmultmin, 
                Rmultmax, VOCdeltadays, starting_month, vaccine='age-order', savefolder='posterior_samples'):
    """
    This function runs the calibration for a given basin
        :param basin_name: name of the basin
        :param nsim: number of total simulations
        :param step_save: saving frequency in steps
        :param R0min: minimum R0
        :param R0max: maximum R0
        :param Imultmin: minimum Imult
        :param Imultmax: maximum Imult
        :param Rmultmin: minimum Rmult
        :param Rmultmax: maximum Rmult
        :param VOCdeltadays: Delta days around date of VOC introduction
        :param starting month: 2020 starting month
        :param vaccine (default=age-order): vaccination strategy
        :return: 0
    """
    
    start_date = datetime(2020, int(starting_month), 1)

    # create Basin object
    basin = Basin(basin_name, "../basins/")
    
    # pre-compute contacts matrices over time
    Cs, dates = compute_contacts(basin, start_date, end_date)

    # get real deaths (first month excluded for the delay Delta)
    real_deaths = basin.epi_data_deaths.loc[(basin.epi_data_deaths["date"] >= start_date) &
                                            (basin.epi_data_deaths["date"] < end_date)]["daily"][33:].reset_index(drop=True)

    # run calibration
    all_params = []
    
    # epidemiological params 
    epi_params = get_epi_params()
    
    # start time of simulations
    start_time = time.time()
    
    for k in range(nsim):

        # sample parameters
        R0 = np.random.uniform(R0min, R0max)
        Delta = np.random.randint(10, 35)
        seasonality_min = np.random.uniform(0.5, 1.0)
        psi = np.random.uniform(1.0, 2.5)
        I_mult = np.random.uniform(Imultmin, Imultmax)
        R_mult = np.random.uniform(Rmultmin, Rmultmax)
        date_intro_VOC = basin.start_date_VOC + timedelta(days=np.random.randint(-VOCdeltadays, VOCdeltadays))
        
        # simulate
        results = simulate(basin, Cs, R0, Delta, dates, seasonality_min, vaccine, I_mult, R_mult, psi, basin.vaccinations, date_intro_VOC)
        
        # compute deaths
        IFR = get_IFR_fixed("verity")
        epi_params["Delta"] = Delta

        # sample a 100 IFR_multiplier/scaler sets
        grid = np.array([np.random.uniform(low=[0.5, 1.0], high=[2.0, 100.0]) for o in range(100)])
        for ifr_mult, scaler in grid:
            epi_params["IFR"] = ifr_mult * IFR
            epi_params["IFR_VOC"] = ifr_mult * IFR
            results_deaths = compute_deaths(results["recovered"], results["recovered_VOC"], results["recovered_V1i"],
                                            results["recovered_V2i"], results["recovered_V1i_VOC"], results["recovered_V2i_VOC"], epi_params)
                   
            df_deaths = pd.DataFrame(data={"real_deaths": real_deaths.values, 
                                           "sim_deaths": scaler / 100.0 * results_deaths["deaths_TOT"].sum(axis=0)[33:]}, index=dates[33:])
            df_deaths = df_deaths.resample("W").sum()

            # accept/reject
            err = wmape(df_deaths["real_deaths"].values, df_deaths["sim_deaths"].values)
            all_params.append([R0, Delta, seasonality_min, psi, I_mult, R_mult, ifr_mult, scaler / 100.0, date_intro_VOC, err])
            
        # save file every step_save steps 
        if k % step_save == 0 and k != 0:
            
            # create unique file name and save
            unique_filename = str(uuid.uuid4())
            filename = basin.name + "_vaccine" + str(vaccine) + "_" + unique_filename + ".npz"
            np.savez_compressed(f"./{savefolder}/posterior_samples_"+ basin.name + "/" + filename, all_params)
            all_params = []
            
    return 0


if __name__ == "__main__":
    
    # parse basin name
    parser = argparse.ArgumentParser(description='Calibration ABC')
    parser.add_argument('--basin', type=str, help='name of the basin')
    parser.add_argument('--nsim', type=int, help='number of simulations')
    parser.add_argument('--step', type=int, help='step save')
    parser.add_argument('--R0min', type=float, help='minimum R0')
    parser.add_argument('--R0max', type=float, help='maximum R0')
    parser.add_argument('--Imultmax', type=float, help='maximum Imult')
    parser.add_argument('--Imultmin', type=float, help='minimum Imult')
    parser.add_argument('--Rmultmax', type=float, help='maximum Rmult')
    parser.add_argument('--Rmultmin', type=float, help='minimum Rmult')
    parser.add_argument('--VOCdeltadays', type=int, help='Delta days around date of VOC introduction')
    parser.add_argument('--startmonth', type=int, help='starting month')
    args = parser.parse_args()

    calibration(basin_name=args.basin,
                nsim=args.nsim,
                step_save=args.step,
                R0min=args.R0min,
                R0max=args.R0max,
                Imultmin=args.Imultmin,
                Imultmax=args.Imultmax,
                Rmultmin=args.Rmultmin,
                Rmultmax=args.Rmultmax,
                VOCdeltadays=args.VOCdeltadays,
                starting_month=args.startmonth)

    