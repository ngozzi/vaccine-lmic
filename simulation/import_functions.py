import numpy as np
import os


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
