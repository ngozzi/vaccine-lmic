import pandas as pd
import numpy as np 
from datetime import datetime
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
end_date = datetime(2021, 10, 1)

def import_posterior(country):
    arr = np.load("../calibration/posteriors/posteriors_" + country + ".npz", allow_pickle=True)["arr_0"]
    np.random.shuffle(arr)
    return arr


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
    
    deaths, deaths_scaled, incidence, incidence_VOC = [], [], [], []
    if scenario == 'data-driven':
        vaccinations = basin.vaccinations
    elif scenario == "ita":
        vaccinations = basin.vaccinations_ita
    elif scenario == "ita_rescale":
        vaccinations = basin.vaccinations_ita_rescale
    elif scenario == "uk":
        vaccinations = basin.vaccinations_uk
    elif scenario == "uk_rescale":
        vaccinations = basin.vaccinations_uk_rescale
    elif scenario == "us":
        vaccinations = basin.vaccinations_us
    elif scenario == "us_rescale":
        vaccinations = basin.vaccinations_us_rescale
    elif scenario == "eu_start":
        vaccinations = basin.vaccinations_eu_start
    elif scenario == "eu_rescale":
        vaccinations = basin.vaccinations_eu_rescale
    elif scenario == "us_start":
        vaccinations = basin.vaccinations_us_start


    # get IFR 
    epi_params = get_epi_params()
    IFR = get_IFR_fixed("verity")
        
    for n in range(Nsim):
        # order of parameters: R0: 0, Delta: 1, seasonality_min: 2, psi: 3, I_mult: 4, R_mult: 5, 
        #                      ifr_mult: 6, scaler / 100.0: 7, date_intro_VOC: 8, err: 9
        params = sampled_params[n]
        results = simulate(basin=basin, Cs=Cs, R0=params[0], Delta=params[1], dates=dates, seasonality_min=params[2], 
                           vaccine=vaccine, I_mult=params[4], R_mult=params[5], psi=params[3], 
                           vaccinations=vaccinations, date_VOC_intro=params[8])
        
        # compute deaths
        epi_params["Delta"] = int(params[1])
        epi_params["IFR"] = float(params[6]) * IFR
        epi_params["IFR_VOC"] = float(params[6]) * IFR

        results_deaths = compute_deaths(results["recovered"], results["recovered_VOC"], results["recovered_V1i"],
                                        results["recovered_V2i"], results["recovered_V1i_VOC"], results["recovered_V2i_VOC"], epi_params)
      
        #incidence.append(results["incidence"].sum(axis=0))
        #incidence_VOC.append(results["incidence_VOC"].sum(axis=0))
        #deaths.append(results_deaths["deaths_TOT"].sum(axis=0))
        deaths_scaled.append(float(params[7]) * results_deaths["deaths_TOT"].sum(axis=0))

    return {#"incidence": incidence, 
            #"incidence_VOC": incidence_VOC,
            #"deaths": deaths, 
            "deaths_scaled": deaths_scaled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--basin', type=str, help='name of the basin')
    parser.add_argument('--week', type=int, help='Week rescale batch')
    args = parser.parse_args()

    if args.week == 1:
        weeks = [4, 8]
    elif args.week == 2:
        weeks = [12, 16]
    elif args.week == 3:
        weeks = [20, 24, 28]
    elif args.week == 4:
        weeks = [32, 36, 40]

    sampled_params = import_posterior(args.basin)
    idxs = np.random.randint(0, 1000, size=50)
    unique_filename = str(uuid.uuid4())

    # earlier start 
    if args.basin in ['Zimbabwe', 'Mali', 'Egypt']:
        start_date = datetime(2020, 10, 1)
    elif args.basin in ['Bolivia', 'Mozambique', 'El Salvador', 'Togo', 'Rwanda', 'Zambia', 'Senegal', 'Ghana']:    
        start_date = datetime(2020, 11, 1)
    else:
        start_date = datetime(2020, 12, 1)


    # simulate
    for scenario in ['data-driven', 'us_rescale']:
        results = simulation(args.basin, scenario, Nsim=50, sampled_params=sampled_params[idxs], vaccine='age-order', start_date=start_date, npis_rescale=0.0, weeks_rescale=10000)
        file_name = unique_filename + "_" + args.basin + "_scenario" + str(scenario) + "_vaccine" + str('age-order')
        np.savez_compressed("./projections_npis/projections_" + args.basin + "/deaths_scaled_" + file_name, results["deaths_scaled"])

    for npis_rescale in np.arange(0.05, 1.0, 0.05):
        for weeks_rescale in weeks:
            results = simulation(args.basin, 'data-driven', Nsim=50, sampled_params=sampled_params[idxs], vaccine='age-order', start_date=start_date, npis_rescale=npis_rescale, weeks_rescale=weeks_rescale)
            file_name = unique_filename + "_" + args.basin + "_scenario" + str('data-driven') + "_vaccine" + str('age-order') + "_npis_rescale" + str(npis_rescale).replace(".", "_") + "_weeks_rescale" + str(weeks_rescale)
            np.savez_compressed("./projections_npis/projections_" + args.basin + "/deaths_scaled_" + file_name, results["deaths_scaled"])

