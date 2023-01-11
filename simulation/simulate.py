import numpy as np 
from datetime import datetime
import argparse 
import uuid
import sys 
sys.path.append("../models/")
from stochastic_SEIRD import simulate
from Basin import Basin
from functions import compute_contacts, get_epi_params, get_IFR_fixed
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

def simulation(basin_name, scenario, Nsim, sampled_params, vaccine, start_date, end_date=datetime(2021, 12, 1)):
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
    #df_deaths = basin.epi_data_deaths
    #df_deaths = df_deaths.loc[(df_deaths.date >= dates[0]) & (df_deaths.date <= dates[-1])]
    
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
      
        #df_deaths["daily_sim"] = results_deaths["deaths_TOT"].sum(axis=0)
        #df_deaths["daily_sim_scaled"] = float(params[7]) * df_deaths["daily_sim"]

        deaths.append(results_deaths["deaths_TOT"].sum(axis=0))
        deaths_scaled.append(float(params[7]) * results_deaths["deaths_TOT"].sum(axis=0))


        incidence.append(results["incidence"].sum(axis=0))
        incidence_VOC.append(results["incidence_VOC"].sum(axis=0))
        #deaths.append(df_deaths["daily_sim"].values)
        #deaths_scaled.append(df_deaths["daily_sim_scaled"].values)

    return {"incidence": incidence, 
            "incidence_VOC": incidence_VOC,
            "deaths": deaths,
            "deaths_scaled": deaths_scaled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--basin', type=str, help='name of the basin')
    args = parser.parse_args()

    sampled_params = import_posterior(args.basin)
    #k = np.random.randint(0, 1000 - 50)
    idxs = np.random.randint(0, 1000, size=100)
    unique_filename = str(uuid.uuid4())
    
    # simulate
    for keyword in ['data-driven', 'False', 'eu_rescale', 'us_rescale', 'eu_start', 'us_start']:

        if keyword == 'False':
            scenario = 'data-driven'
            vaccine = 'False'
        else:
            scenario = keyword
            vaccine = 'age-order'
        # earlier start 
        if args.basin in ['Zimbabwe', 'Mali', 'Egypt']:
            start_date = datetime(2020, 10, 1)
        elif args.basin in ['Bolivia', 'Mozambique', 'El Salvador', 'Togo', 'Rwanda', 'Zambia', 'Senegal', 'Ghana']:    
            start_date = datetime(2020, 11, 1)
        else:
            start_date = datetime(2020, 12, 1)
        results = simulation(args.basin, scenario, Nsim=100, sampled_params=sampled_params[idxs], vaccine=vaccine, start_date=start_date)
        
        # sav
        file_name = unique_filename + "_" + args.basin + "_scenario" + str(scenario) + "_vaccine" + str(vaccine)
        np.savez_compressed("./projections_october/projections_" + args.basin + "/inc_" + file_name, results["incidence"])
        np.savez_compressed("./projections_october/projections_" + args.basin + "/inc_VOC_" + file_name, results["incidence_VOC"])
        np.savez_compressed("./projections_october/projections_" + args.basin + "/deaths_" + file_name, results["deaths"])
        np.savez_compressed("./projections_october/projections_" + args.basin + "/deaths_scaled_" + file_name, results["deaths_scaled"])