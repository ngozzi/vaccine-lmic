# libraries
import numpy as np

def shift_deaths(arr, Delta):
    """
    This function shifts deaths ahead of Delta days
    :param arr: array of deaths
    :param Delta: delay in deaths
    :return: array of deaths shifted
    """
    arr = np.roll(arr, shift=int(Delta))
    arr[0:int(Delta)] = 0
    return arr


def compute_deaths(R, R_VOC, R_V1, R_V2, R_V1_VOC, R_V2_VOC, params, seed=None):
    """
    This function computes the number of daily deaths
    :param R: recovered
    :param R_VOC: recovered from VOC infection
    :param R_V1: recovered with one dose
    :param R_V2: recovered with two doses
    :param R_V1_VOC: recovered with one dose from VOC infection
    :param R_V2_VOC: recovered with two doses from VOC infection
    :param params: simulation parameters
    :param seed: random seed
    :return: total daily deaths and for different vaccination status
    """

    # set random seed
    if seed != None:
        np.random.seed(seed)

    # initialize deaths
    nage, T = R.shape[0], R.shape[1]
    deaths, deaths_vacc1dose, deaths_vacc2dose = np.zeros((nage, T)), np.zeros((nage, T)), np.zeros((nage, T))

    ifr = np.zeros((nage, T))
    tot_rec, tot_deaths = np.zeros(T), np.zeros(T)

    for age in range(nage):
        
        # compute deaths
        novacc = np.random.binomial(R[age].astype(int), params["IFR"][age])
        vacc1dose = np.random.binomial(R_V1[age].astype(int), (1. - params["VEM1"]) * params["IFR"][age])
        vacc2dose = np.random.binomial(R_V2[age].astype(int), (1. - params["VEM2"]) * params["IFR"][age])
        
        # compute deaths VOC
        novacc_VOC = np.random.binomial(R_VOC[age].astype(int), params["IFR_VOC"][age])
        vacc1dose_VOC = np.random.binomial(R_V1_VOC[age].astype(int), (1. - params["VEM1_VOC"]) * params["IFR_VOC"][age])
        vacc2dose_VOC = np.random.binomial(R_V2_VOC[age].astype(int), (1. - params["VEM2_VOC"]) * params["IFR_VOC"][age])
        
        deaths[age] = novacc + vacc1dose + vacc2dose + novacc_VOC + vacc1dose_VOC + vacc2dose_VOC
        deaths_vacc1dose[age] = vacc1dose + vacc1dose_VOC
        deaths_vacc2dose[age] = vacc2dose + vacc2dose_VOC
        
        tot_rec += (R[age] + R_V1[age] + R_V2[age] + R_VOC[age] + R_V1_VOC[age] + R_V2_VOC[age])
        tot_deaths += deaths[age]

        # update ifr
        ifr[age] = (deaths[age]) / (R[age] + R_V1[age] + R_V2[age] + R_VOC[age] + R_V1_VOC[age] + R_V2_VOC[age])

        # shift
        deaths[age] = shift_deaths(deaths[age], params["Delta"])
        deaths_vacc1dose[age] = shift_deaths(deaths_vacc1dose[age], params["Delta"])
        deaths_vacc2dose[age] = shift_deaths(deaths_vacc2dose[age], params["Delta"])
        ifr[age] = shift_deaths(ifr[age], params["Delta"]) 

    ifr_tot = tot_deaths / tot_rec
    ifr_tot = shift_deaths(ifr_tot, params["Delta"])

    return {"deaths_TOT": deaths,
            "deaths_vacc1dose": deaths_vacc1dose,
            "deaths_vacc2dose": deaths_vacc2dose,
            "ifr": ifr,
            "ifr_tot": ifr_tot}
