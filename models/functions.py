# libraries
import numpy as np
from datetime import datetime, timedelta
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

# n. of compartments and age groups
ncomp = 23
nage = 10


def get_IFR_fixed(source="salje"):
    """
    This function returns the COVID-19 IFR from different sources
    :param source: source of IFR (verity or salje)
    :return: array of IFR (10 age groups)
    """
    # salje et al IFR
    if source == "salje":
        return np.array([0.001 / 100,  # 0-9
                         0.001 / 100,  # 10-19
                         0.005 / 100,  # 20-24
                         0.005 / 100,  # 25-29
                         0.020 / 100,  # 30-39
                         0.050 / 100,  # 40-49
                         0.200 / 100,  # 50-59
                         0.700 / 100,  # 60-69
                         1.900 / 100,  # 70-79
                         8.300 / 100])  # 80+
    
    # verity et al IFR
    elif source == "verity":
        return np.array([0.00161 / 100,  # 0-9
                         0.00695 / 100,  # 10-19
                         0.0309  / 100,  # 20-24
                         0.0309  / 100,  # 25-29
                         0.0844  / 100,  # 30-39
                         0.161   / 100,  # 40-49
                         0.595   / 100,  # 50-59
                         1.93    / 100,  # 60-69
                         4.28    / 100,  # 70-79
                         7.80   / 100])  # 80+
    
    else:
        print("source not supported (salje or verity)")
        return np.zeros(nage)
        

def get_IFR(ifr_base, ifr_avg, Nk):
    """
    This function computes the IFR age stratified as reported in Verity et al
    :param IFR_avg: average IFR
    :param Nk: number of individuals in different compartments
    :return: new IFR
    """
    # get average IFR from base ifr
    ifr_avg_base = np.sum(np.array(ifr_base) * Nk) / np.sum(Nk)
    # compute the multiplying parameter and return new IFR
    gamma = ifr_avg / ifr_avg_base
    return gamma * np.array(ifr_base)


def update_contacts(basin, date):
    """
    This functions compute the contacts matrix for a specific date
        :param basin: Basin object
        :param date: date
        :return: contacts matrix for the given date
    """
    # get year-week
    if date.isocalendar()[1] < 10:
        year_week = str(date.isocalendar()[0]) + "-0" + str(date.isocalendar()[1])
    else:
        year_week = str(date.isocalendar()[0]) + "-" + str(date.isocalendar()[1])
    # red factor
    omega = basin.reductions.loc[basin.reductions.year_week == year_week]["red"].values[0]
    # contacts matrix with reductions
    C_hat = omega * basin.contacts_matrix 
    return C_hat


def compute_contacts(basin, start_date, end_date):
    """
    This function computes contacts matrices over a given time window
        :param basin: Basin object
        :param start_date: initial date
        :param end_date: last date
        :return: list of dates and dictionary of contacts matrices over time
    """
    # pre-compute contacts matrices over time
    Cs, date, dates = {}, start_date, [start_date]
    for i in range((end_date - start_date).days - 1):
        Cs[date] = update_contacts(basin, date)
        date += timedelta(days=1)
        dates.append(date)
    return Cs, dates


def compute_contacts_factor(basin, start_date, end_date, factors):
    """
    This function computes contacts matrices over a given time window
        :param basin: Basin object
        :param start_date: initial date
        :param end_date: last date
        :param factors: dictionary of reduction factors
        :return: list of dates and dictionary of contacts matrices over time
    """
    # pre-compute contacts matrices over time
    Cs, date, dates = {}, start_date, [start_date]
    for i in range((end_date - start_date).days - 1):
        month = date.month
        omega = factors[month]
        Cs[date] = omega * basin.contacts_matrix 
        date += timedelta(days=1)
        dates.append(date)
    return Cs, dates


def VEM(VE, VES):
    """
    This function returns VEM given VE and VES (VE = 1 - (1 - VES) * (1 - VEM))
    :param VE: overall vaccine efficacy
    :param VES: vaccine efficacy against infection
    :return: VEM (vaccine efficacy against disease)
    """
    return 1 - (1 - VE) / (1 - VES)


def get_epi_params():
    """
    This function return the epidemiological parameters
    :param increased_mortality: increased mortality of VOC (float, defual=1.0)
    :param reduced_efficacy: if True vaccines are less effective against VOC (default=False)
    :return dictionary of params
    """
    params = {}

    # epidemiological parameters
    params["mu"] = 1 / 2.5
    params["eps"] = 1 / 4.0
    params["eps_VOC"] = 1 / 3.0

    # vaccine delay
    params["Delta_V"] = 14.0

    # vaccine efficacy
    params["VE1"] = 0.8
    params["VES1"] = 0.7
    params["VEM1"] = VEM(params["VE1"], params["VES1"])
    params["VE2"] = 0.9
    params["VES2"] = 0.8
    params["VEM2"] = VEM(params["VE2"], params["VES2"])
    params["VEI"] = 0.4
    
    # vaccine efficacy VOC
    params["VE1_VOC"] = 0.7
    params["VES1_VOC"] = 0.3
    params["VEM1_VOC"] = VEM(params["VE1_VOC"], params["VES1_VOC"])
    params["VE2_VOC"] = 0.9
    params["VES2_VOC"] = 0.6
    params["VEM2_VOC"] = VEM(params["VE2_VOC"], params["VES2_VOC"])
    params["VEI_VOC"] = 0.4

    return params


def get_beta(R0, mu, C, Nk, date, seasonality_min, hemispheres):
    """
    This functions return beta for a SEIR model with age structure
        :param R0: basic reproductive number
        :param mu: recovery rate
        :param C: contacts matrix
        :param Nk: n. of individuals in different age groups
        :param date: current day
        :param seasonality_min: seasonality parameter
        :param hemispheres: hemispheres of basin (0: north, 1: tropical, 2: south)
        :return: returns the rate of infection beta for the given R0
    """
    # get seasonality adjustment
    seas_adj = apply_seasonality(date, seasonality_min, hemispheres)
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
    return R0 * mu / (seas_adj * np.max([e.real for e in np.linalg.eig(C_hat)[0]]))


def apply_seasonality(day, seasonality_min, hemispheres, seasonality_max=1):
    """
    This function computes the seasonality adjustment for transmissibility
        :param day: current day
        :param seasonality_min: seasonality parameter
        :param hemispheres: hemispheres of basins, 0: north, 1: tropical, 2; south (dataframe)
        :param seasonality_max: seasonality parameter
        :return: returns seasonality adjustment
    """
    
    s_r = seasonality_min / seasonality_max
    day_max_north = datetime(day.year, 1, 15)
    day_max_south = datetime(day.year, 7, 15)

    seasonal_adjustment = np.empty(shape=(3,), dtype=np.float64)

    # north hemisphere
    seasonal_adjustment[0] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_north).days + 0.5 * np.pi) + 1 + s_r)

    # tropical hemisphere
    seasonal_adjustment[1] = 1.0

    # south hemisphere
    seasonal_adjustment[2] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_south).days + 0.5 * np.pi) + 1 + s_r)

    num, den = 0.0, 0.0
    for index, row in hemispheres.iterrows():
        num += seasonal_adjustment[row['hemisphere']] * row['basin_population']
        den += row['basin_population']
        
    return float(num) / float(den)


def get_initial_conditions(basin, start_date, I_mult, R_mult, params, seed=None):
    """
    This function returns the initial conditions 
        :param basin: Basin object
        :param start_date: date of start of simulation
        :param I_mult: multiplying factor for infected
        :param R_mult: multiplying factor for recovered
        :param params: dictionary of params
        :param seed: random seed
        :return: initial conditions on start date
    """
    
    if seed != None:
        np.random.seed(seed)
        
    tot_R = basin.epi_data_cases.loc[basin.epi_data_cases.date == start_date]["cumulative"]
    tot_I = basin.epi_data_cases.loc[(basin.epi_data_cases.date < start_date) &
                                     (basin.epi_data_cases.date >= start_date - timedelta(days=7))]['daily'].sum()
    Nk = basin.Nk
    # initial conditions
    initial_conditions = np.zeros((ncomp, nage))
    for age in range(nage):
        
        # sample initial conditions
        Lstart = int(np.random.poisson(I_mult * tot_I * (Nk[age] / np.sum(Nk)) * (params["eps"] / (params["eps"] + params["mu"]))))
        Istart = int(np.random.poisson(I_mult * tot_I * (Nk[age] / np.sum(Nk)) * (params["mu"] / (params["eps"] + params["mu"]))))
        Rstart = int(np.random.poisson(R_mult * tot_R * (Nk[age] / np.sum(Nk))))

        # non-negative constraint
        total = Lstart + Istart + Rstart
        if total > Nk[age]:
            Lstart = int(Nk[age] * Lstart / total)
            Istart = int(Nk[age] * Istart / total)
            Rstart = int(Nk[age] * Rstart / total)
        # L 
        initial_conditions[1, age] = Lstart
        # I
        initial_conditions[2, age] = Istart
        # R 
        initial_conditions[3, age] = Rstart
        # S
        initial_conditions[0, age] = Nk[age] - Istart - Lstart - Rstart

        
    return initial_conditions


def plot_style(font_path):
    """
    This function defines the plot style
    :param font_path: path to font directory
    :return: list of colors
    """
    plt.rcParams['axes.linewidth'] = 0.3
    plt.rcParams['xtick.major.width'] = 0.3
    plt.rcParams['ytick.major.width'] = 0.3
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.minor.width'] = 0.2
    plt.rcParams['ytick.minor.width'] = 0.2
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.size'] = 1.5

    font_dirs = [font_path]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)
    plt.rcParams['font.family'] = 'Encode Sans Condensed'

    colors = ['#6CC2BD', '#5A809E', '#FFC1A6', '#F57D7C', '#7C79A2', '#FEE4C4']
    return colors


def wmape(arr1, arr2):
    """
    Weighted Mean Absolute Percentage Error (WMAPE)
    """
    return np.sum(np.abs(arr1 - arr2)) / np.sum(np.abs(arr1))
