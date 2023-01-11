import numpy as np
import os

def import_posterior(country, path="./"):
    """
    This function import the posterior distribution for a given country
    """
    return np.load(path + "/posteriors_october/posteriors_" + country + ".npz", allow_pickle=True)["arr_0"]

def import_posterior_dict(country, path="./"):
    # order of parameters: R0: 0, Delta: 1, seasonality_min: 2, psi: 3, I_mult: 4, R_mult: 5, 
    #                      ifr_mult: 6, scaler / 100.0: 7, date_intro_VOC: 8, err: 9
    posterior = import_posterior(country, path=path)
    posterior_dict = {}
    posterior_dict["R0"] = posterior[:, 0]
    posterior_dict["Delta"] = posterior[:, 1]
    posterior_dict["seasonality_min"] = posterior[:, 2]
    posterior_dict["psi"] = posterior[:, 3]
    posterior_dict["I_mult"] = posterior[:, 4]
    posterior_dict["R_mult"] = posterior[:, 5]
    posterior_dict["ifr_mult"] = posterior[:, 6]
    posterior_dict["scaler"] = posterior[:, 7]
    posterior_dict["date_intro_VOC"] = posterior[:, 8]
    posterior_dict["err"] = posterior[:, 9]

    return posterior_dict


def import_projections_deaths(country, scenario, vaccine='age-order', scaled=True, path="./projections_october/"):
    """
    This function import the deaths projections for a given scenario and country
    """
    files = os.listdir(path + "projections_" + country + "/")
    files = [file for file in files if vaccine in file and "deaths" in file and scenario in file]

    if scaled:
        files = [file for file in files if "scaled" in file]
    else: 
        files = [file for file in files if "scaled" not in file]
        
    #print(len(files))
    data = []
    for file in np.sort(files):
        data.extend(np.load(path + "/projections_" + country + "/" + file, allow_pickle=True)["arr_0"])
    return np.array(data)


def get_averted_deaths(country, scenario, scaled, baseline, perc, path="./projections_october/"):

    """
    Compute the number of averted deaths
    :param country: country name
    :param scenario: scenario name
    :param scaled: if True averted deaths are multiplied by underreporting
    :param baseline: baseline scenario name
    :param perc: if True averted deaths are reported as percentage respect to baseline
    :return: averted deaths
    """

    # import baseline deaths
    if baseline == "False":
        deaths_baseline = import_projections_deaths(country=country, scenario="data-driven", vaccine="False", scaled=scaled, path=path)
    elif baseline == "data-driven":
        deaths_baseline = import_projections_deaths(country=country, scenario="data-driven", vaccine="age-order", scaled=scaled, path=path)
    else:
        print("Unknown baseline (supported: False, data-driven)")
        return 0

    # import scenario deaths
    deaths = import_projections_deaths(country=country, scenario=scenario, vaccine="age-order", scaled=scaled, path=path)

    if perc:
        return 100 * (np.sum(deaths_baseline, axis=1) - np.sum(deaths, axis=1)) / np.sum(deaths_baseline, axis=1)
    else:
        return np.sum(deaths_baseline, axis=1) - np.sum(deaths, axis=1)
