import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
import argparse 
import uuid
import sys 
sys.path.append("../models/")
from stochastic_SEIRD import simulate
from Basin import Basin
from functions import get_epi_params, get_IFR_fixed, compute_contacts
from deaths import compute_deaths
import warnings
warnings.filterwarnings("ignore")

# n. of compartments and age groups
ncomp = 23
nage = 10

# simulation dates
start_date = datetime(2020, 10, 1)
end_date = datetime(2021, 10, 1)


def import_posterior(country):
    return pd.read_csv(f"../calibration_ABC-SMC/posteriors/posterior_{country}.csv")


def simulation(basin_name, scenario, Nsim, sampled_params, vaccine, start_date, npis_rescale, weeks_rescale, end_date=datetime(2021, 12, 1)):
    """
    This function runs the simulations for the calibrated model 
    :param basin_name: name of the basin
    :param th: ABC tolerance
    :param scenario: vaccinations scenario
    :param Nsim: number of simulations 
    :param sampled_params: sampled parameters
    :param vaccine: vaccine strategy
    :return: results of simulations
    """

    # create basin object 
    basin = Basin(basin_name, "../basins/")

    # correct reductions 
    new_reds = []
    e = 0
    for index, row in basin.reductions.iterrows():
        if row["year_week"] > "2020-50":
            if e < weeks_rescale:
                new_reds.append(row["red"] - row["red"] * npis_rescale)
                e += 1
            else:
                new_reds.append(row["red"])
        else:
            new_reds.append(row["red"])
    basin.reductions = pd.DataFrame(data={"year_week": basin.reductions.year_week.values, "red": np.array(new_reds)})
    # compute contacts
    Cs, dates = compute_contacts(basin, start_date, end_date)
    
    deaths_scaled = []
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

    # get IFR 
    epi_params = get_epi_params()
    IFR = get_IFR_fixed("verity")
        
    for n in range(Nsim):
     
        params = sampled_params.iloc[n]
        date_VOC_intro = basin.start_date_VOC + timedelta(days=int(params.delta_days_intro_VOC))
        results = simulate(basin=basin, Cs=Cs, R0=params.R0, Delta=params.Delta, dates=dates, seasonality_min=params.seasonality_min, 
                           vaccine=vaccine, I_mult=params.I_mult, R_mult=params.R_mult, psi=params.psi, 
                           vaccinations=vaccinations, date_VOC_intro=date_VOC_intro)
        
        # compute deaths
        epi_params["Delta"] = int(params.Delta)
        epi_params["IFR"] = float(params.ifr_mult) * IFR
        epi_params["IFR_VOC"] = float(params.ifr_mult) * IFR

        results_deaths = compute_deaths(results["recovered"], results["recovered_VOC"], results["recovered_V1i"],
                                        results["recovered_V2i"], results["recovered_V1i_VOC"], results["recovered_V2i_VOC"], epi_params)

        # rescale deaths and resample weekly
        deaths_temp = results_deaths["deaths_TOT"].sum(axis=0)
        deaths_scl = params.scaler / 100.0 * deaths_temp
        deaths_scaled.append(deaths_scl)

    return {"deaths_scaled": deaths_scaled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--basin', type=str, help='name of the basin')
    parser.add_argument('--week', type=int, help='Week rescale batch')
    args = parser.parse_args()
    weeks = [args.week]
    sampled_params = import_posterior(args.basin)
    idxs = np.random.randint(0, 1000, size=100)
    unique_filename = str(uuid.uuid4())

    # simulate
    for scenario in ['data-driven', 'us_rescale']:
        print(scenario)
        results = simulation(args.basin, scenario, Nsim=100, sampled_params=sampled_params.iloc[idxs], vaccine='age-order', start_date=start_date, npis_rescale=0.0, weeks_rescale=10000)
        file_name = unique_filename + "_" + args.basin + "_scenario" + str(scenario) + "_vaccine" + str('age-order')
        np.savez_compressed("./projections_npis/projections_" + args.basin + "/deaths_scaled_" + file_name, results["deaths_scaled"])

    for npis_rescale in np.arange(0.05, 1.0, 0.05):
        print(npis_rescale)
        for weeks_rescale in weeks:
            results = simulation(args.basin, 'data-driven', Nsim=100, sampled_params=sampled_params.iloc[idxs], vaccine='age-order', start_date=start_date, npis_rescale=npis_rescale, weeks_rescale=weeks_rescale)
            file_name = unique_filename + "_" + args.basin + "_scenario" + str('data-driven') + "_vaccine" + str('age-order') + "_npis_rescale" + str(npis_rescale).replace(".", "_") + "_weeks_rescale" + str(weeks_rescale)
            np.savez_compressed("./projections_npis/projections_" + args.basin + "/deaths_scaled_" + file_name, results["deaths_scaled"])

