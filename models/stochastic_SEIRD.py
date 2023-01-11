# libraries
from functions import apply_seasonality, get_beta, get_IFR, get_initial_conditions, get_epi_params, get_IFR_fixed
from deaths import compute_deaths
from vaccinations import vaccinate
import numpy as np
from numpy.random import binomial

# n. of compartments and age groups
ncomp = 23
nage = 10


def get_VOC_introduction(I_today, perc_VOC, C, Nk, compartments_step, params):
    """
    This function computes the distribution of VOC cases at the introduction
    :param I_today: total incidence on introduction day 
    :param perc_VOC: percentage of VOC cases sequentiated on the introduction date
    :param C: contacts matrix on introduction date
    :param Nk: array of number of individuals in different age groups
    :param compartments_step: number of individuals in different compartments on the introduction date
    :param params: dictionary of parameters
    :return: returns updated compartments
    """
    
    # VOC cases
    I_VOC_today = perc_VOC * I_today
    L_VOC_today = I_VOC_today * (params["psi"] * params["beta"] * np.linalg.eigvals(C).real.max() / params["mu"])

    # distribute among age groups homogeneously
    for age in range(nage):
        # infected and latent in this group
        I_VOC_age = I_VOC_today * Nk[age] / Nk.sum()
        L_VOC_age = L_VOC_today * Nk[age] / Nk.sum()

        den_I = compartments_step[2, age] + compartments_step[7, age] + compartments_step[12, age]
        den_L = compartments_step[1, age] + compartments_step[6, age] + compartments_step[11, age]

        if den_I == 0:
            new_I_VOC, new_I_VOC_V1i, new_I_VOC_V2i = 0, 0, 0
        else:
            new_I_VOC = min(compartments_step[2, age], int(I_VOC_age * compartments_step[2, age] / den_I))
            new_I_VOC_V1i = min(compartments_step[7, age], int(I_VOC_age * compartments_step[7, age] / den_I))
            new_I_VOC_V2i = min(compartments_step[12, age], int(I_VOC_age * compartments_step[12, age] / den_I))
            
        if den_L == 0:
            new_L_VOC, new_L_VOC_V1i, new_L_VOC_V2i = 0, 0, 0
        else:
            new_L_VOC = min(compartments_step[1, age], int(L_VOC_age * compartments_step[1, age] / den_L))
            new_L_VOC_V1i = min(compartments_step[6, age], int(L_VOC_age * compartments_step[6, age] / den_L))
            new_L_VOC_V2i = min(compartments_step[11, age], int(L_VOC_age * compartments_step[11, age] / den_L))
        
        # update
        # I, L
        compartments_step[2, age] -= new_I_VOC
        compartments_step[15, age] += new_I_VOC
        compartments_step[1, age] -= new_L_VOC
        compartments_step[14, age] += new_L_VOC

        # I_V1i, L_V1i
        compartments_step[7, age] -= new_I_VOC_V1i
        compartments_step[18, age] += new_I_VOC_V1i
        compartments_step[6, age] -= new_L_VOC_V1i
        compartments_step[17, age] += new_L_VOC_V1i

        # I_V2i, L_V2i
        compartments_step[12, age] -= new_I_VOC_V2i
        compartments_step[21, age] += new_I_VOC_V2i
        compartments_step[11, age] -= new_L_VOC_V2i
        compartments_step[20, age] += new_L_VOC_V2i
        
    return compartments_step


def get_force_infection(age, params, compartments_step, C, Nk, seasonal_adjustment):
    """
    This function computes the force of infection of both wild type and VOC for a given age group
    :param age: age group
    :param params: dictionary of parameters
    :param compartments_step: individuals in different compartments at this step
    :param C: contacts matrix at this step
    :param Nk: number of people in different age groups
    :param seasonal_adjustment: seasonal adjustment
    :return: force of infection of both wild type and VOC
    """
    
    # compute forces of infection
    force_inf = np.sum(params["beta"] * seasonal_adjustment * C[age, :] * (compartments_step[2, :] +
                                                                           (1. - params["VEI"]) * compartments_step[7, :] +
                                                                           (1. - params["VEI"]) * compartments_step[12, :]) / Nk)
    
    force_inf_VOC = np.sum(params["beta"] * seasonal_adjustment * C[age, :] * (params["psi"] * compartments_step[15, :] +  
                                                                               params["psi"] * (1. - params["VEI_VOC"]) * compartments_step[18, :] +
                                                                               params["psi"] * (1. - params["VEI_VOC"]) * compartments_step[21, :]) / Nk)
    return force_inf, force_inf_VOC


def simulate(basin, Cs, R0, Delta, dates, seasonality_min, vaccine, I_mult, R_mult, psi, vaccinations, date_VOC_intro, seed=None):
    """
    This function runs the 2-strain SEIRD model
    :param basin: basin object
    :param Cs: dictionary of contacts matrices over time
    :param R0: basic reproductive number
    :param Delta: delay (in days) in deaths
    :param dates: array of dates
    :param seasonality_min: seasonality parameter
    :param vaccine: vaccination strategy
    :param I_mult: multiplying factor for initial infected
    :param R_mult: multiplying factor for initial recovered
    :param psi: increased transmissibility of VOC
    :param vaccinations: dictionary of vaccinations 
    :param date_VOC_intro: date of introduction of VOC
    :param seed: random seed
    :return: returns simulation results
    """
    
    # get epi params
    params = get_epi_params()
    
    # get initial conditions
    initial_conditions = get_initial_conditions(basin, dates[0], I_mult, R_mult, params, seed)
    
    # get beta from Rt w.r.t to initial_date
    beta = get_beta(R0, params["mu"], Cs[dates[0]], basin.Nk, dates[0], seasonality_min, basin.hemispheres)

    # add parameters
    params["beta"] = beta
    params["Delta"] = Delta
    params["seasonality_min"] = seasonality_min
    params["psi"] = psi 
        
    # simulate
    results = stochastic_seird(Cs, initial_conditions, vaccinations, basin.Nk, dates, vaccine, params, date_VOC_intro, 0.05, basin.hemispheres, seed)
    
    return results


def stochastic_seird(Cs, initial_conditions, vaccinations, Nk, dates, vaccine, params, start_date_VOC, perc_VOC, hemispheres, seed=None):
    """
    This function simulates a stochastic SEIR model with two strains and vaccinations
        :param Cs: dictionary of contact matrices
        :param initial_conditions: initial conditions for different compartment/age groups
        :param vaccinations: dictionary of vaccinations
        :param Nk: number of people in different age groups
        :param dates: array of dates
        :param vaccine: vaccination strategy
        :param params: dictionary of parameters
        :param hemispheres: hemispheres of country (dataframe)
        :param start_date_VOC: date of introduction of VOC
        :param perc_VOC: percentage of VOC cases sequentiated on the introduction date
        :param seed: random seed
        :return: returns evolution of n. of individuals in different compartments
    """

    # initial conditions
    T = len(dates)
    compartments = np.zeros((ncomp, nage, T))
    compartments[:, :, 0] = initial_conditions

    # recovered (to compute deaths)
    recovered, recovered_VOC, recovered_V1i, recovered_V2i, recovered_V1i_VOC, recovered_V2i_VOC = np.zeros((nage, T)), np.zeros((nage, T)), \
                                                                                                   np.zeros((nage, T)), np.zeros((nage, T)), \
                                                                                                   np.zeros((nage, T)), np.zeros((nage, T))

    # incidence
    incidence_TOT, incidence, incidence_VOC = np.zeros((nage, T)), np.zeros((nage, T)), np.zeros((nage, T))

    # vaccinations per step
    vaccines_per_step_1dose, vaccines_per_step_2dose = np.zeros((nage, T)), np.zeros((nage, T))

    # set seed
    if seed != None:
        np.random.seed(seed)

    # simulate
    for i in range(T - 1):

        # vaccinate
        compartments_step, vaccines_given_1dose, vaccines_given_2dose = vaccinate(vaccine, vaccinations, compartments[:, :, i], dates[i])
        compartments[:, :, i] = compartments_step
        vaccines_per_step_1dose[:, i] = vaccines_given_1dose
        vaccines_per_step_2dose[:, i] = vaccines_given_2dose
        
        # update contacts
        C = Cs[dates[i]]

        # seasonality adjustment
        seasonal_adjustment = apply_seasonality(dates[i], params["seasonality_min"], hemispheres)

        # next step solution
        next_step = np.zeros((ncomp, nage))

        # iterate over ages
        for age1 in range(nage):

            # get force of infection for this age group
            force_inf, force_inf_VOC = get_force_infection(age1, params, compartments[:, :, i], C, Nk, seasonal_adjustment)

            ### RANDOM SAMPLING ###
            # S->L, S->L_VOC
            if force_inf + force_inf_VOC == 0:
                new_L, new_L_VOC, new_L_fromV1i, new_L_fromV1i_VOC, new_L_fromV2i, new_L_fromV2i_VOC = 0, 0, 0, 0, 0, 0

            else:

                new_L_tot = binomial(compartments[0, age1, i], 1 - np.exp(-(force_inf + force_inf_VOC)))
                new_L = binomial(new_L_tot, force_inf / (force_inf + force_inf_VOC))
                new_L_VOC = new_L_tot - new_L

                # V1i -> L_V1i and V1i -> L_V1i_VOC
                new_L_fromV1i_TOT = binomial(compartments[5, age1, i], 1 - np.exp(-((1. - params["VES1"]) * force_inf + (1. - params["VES1_VOC"]) * force_inf_VOC)))
                new_L_fromV1i = binomial(new_L_fromV1i_TOT, (1. - params["VES1"]) * force_inf / ((1. - params["VES1"]) * force_inf +
                                                                                                           (1. - params["VES1_VOC"]) * force_inf_VOC))
                new_L_fromV1i_VOC = new_L_fromV1i_TOT - new_L_fromV1i

                # V2i -> L_V2i and V2i -> L_V2i_VOC
                new_L_fromV2i_TOT = binomial(compartments[10, age1, i], 1 - np.exp(-((1. - params["VES2"]) * force_inf + (1. - params["VES2_VOC"]) * force_inf_VOC)))
                new_L_fromV2i = binomial(new_L_fromV2i_TOT, (1. - params["VES2"]) * force_inf/ ((1. - params["VES2"]) * force_inf +
                                                                                                          (1. - params["VES2_VOC"]) * force_inf_VOC))
                new_L_fromV2i_VOC = new_L_fromV2i_TOT - new_L_fromV2i


            # V1r -> L_V1i, V1r -> L_V1i_VOC, and V1r -> V1i
            leaving_fromV1r = binomial(compartments[4, age1, i], 1 - np.exp(-(force_inf + force_inf_VOC + 1. / params["Delta_V"])))
            new_L_fromV1r = binomial(leaving_fromV1r, force_inf / (force_inf + force_inf_VOC + 1. / params["Delta_V"]))
            new_L_fromV1r_VOC = binomial(leaving_fromV1r - new_L_fromV1r, force_inf_VOC / (force_inf_VOC + 1. / params["Delta_V"]))
            new_V_fromV1r = leaving_fromV1r - new_L_fromV1r - new_L_fromV1r_VOC

            # V2r -> L_V2i, V2r -> L_V2i_VOC, and V2r -> V2i

            leaving_fromV2r = binomial(compartments[9, age1, i],
                                                 1 - np.exp(-((1. - params["VES1"]) * force_inf + (1. - params["VES1_VOC"]) * force_inf_VOC + 1. / params["Delta_V"])))
            new_L_fromV2r = binomial(leaving_fromV2r,
                                               (1. - params["VES1"]) * force_inf / ((1. - params["VES1"]) * force_inf + (1. - params["VES1_VOC"]) * force_inf_VOC + 1. / params["Delta_V"]))
            new_L_fromV2r_VOC = binomial(leaving_fromV2r - new_L_fromV2r,
                                                   (1. - params["VES1_VOC"]) * force_inf_VOC / ((1. - params["VES1_VOC"]) * force_inf_VOC + 1. / params["Delta_V"]))
            new_V_fromV2r = leaving_fromV2r - new_L_fromV2r - new_L_fromV2r_VOC

            # L -> I
            new_I = binomial(compartments[1, age1, i], params["eps"])
            # I -> R
            new_R = binomial(compartments[2, age1, i], params["mu"])
            # L_V1i -> I_V1i
            new_I_fromV1i = binomial(compartments[6, age1, i], params["eps"])
            # L_V2i -> I_V2i
            new_I_fromV2i = binomial(compartments[11, age1, i], params["eps"])
            # I_V1i -> R_V1i
            new_R_fromV1i = binomial(compartments[7, age1, i], params["mu"])
            # I_V2i -> R_V2i
            new_R_fromV2i = binomial(compartments[12, age1, i], params["mu"])
            # L_VOC -> I_VOC
            new_I_VOC = binomial(compartments[14, age1, i], params["eps_VOC"])
            # I_VOC -> R_VOC
            new_R_VOC = binomial(compartments[15, age1, i], params["mu"])
            # L_V1i_VOC -> I_V1i_VOC
            new_I_fromV1i_VOC = binomial(compartments[17, age1, i], params["eps_VOC"])
            # L_V2i_VOC -> I_V2i_VOC
            new_I_fromV2i_VOC = binomial(compartments[20, age1, i], params["eps_VOC"])
            # I_V1i_VOC -> R_V1i_VOC
            new_R_fromV1i_VOC = binomial(compartments[18, age1, i], params["mu"])
            # I_V2i_VOC -> R_V2i_VOC
            new_R_fromV2i_VOC = binomial(compartments[21, age1, i], params["mu"])

            # update next step solution
            # S
            next_step[0, age1] = compartments[0, age1, i] - new_L - new_L_VOC
            # L
            next_step[1, age1] = compartments[1, age1, i] + new_L - new_I
            # I
            next_step[2, age1] = compartments[2, age1, i] + new_I - new_R
            # R
            next_step[3, age1] = compartments[3, age1, i] + new_R
            # V1r
            next_step[4, age1] = compartments[4, age1, i] - new_L_fromV1r - new_L_fromV1r_VOC - new_V_fromV1r
            # V1i
            next_step[5, age1] = compartments[5, age1, i] - new_L_fromV1i - new_L_fromV1i_VOC + new_V_fromV1r
            # L_V1i
            next_step[6, age1] = compartments[6, age1, i] + new_L_fromV1i + new_L_fromV1r - new_I_fromV1i
            # I_V1i
            next_step[7, age1] = compartments[7, age1, i] + new_I_fromV1i - new_R_fromV1i
            # R_V1i
            next_step[8, age1] = compartments[8, age1, i] + new_R_fromV1i
            # V2r
            next_step[9, age1] = compartments[9, age1, i] - new_L_fromV2r - new_L_fromV2r_VOC - new_V_fromV2r
            # V2i
            next_step[10, age1] = compartments[10, age1, i] - new_L_fromV2i - new_L_fromV2i_VOC + new_V_fromV2r
            # L_V2i
            next_step[11, age1] = compartments[11, age1, i] + new_L_fromV2i + new_L_fromV2r - new_I_fromV2i
            # I_V2i
            next_step[12, age1] = compartments[12, age1, i] + new_I_fromV2i - new_R_fromV2i
            # R_V2i
            next_step[13, age1] = compartments[13, age1, i] + new_R_fromV2i
            # L_VOC
            next_step[14, age1] = compartments[14, age1, i] + new_L_VOC - new_I_VOC
            # I_VOC
            next_step[15, age1] = compartments[15, age1, i] + new_I_VOC - new_R_VOC
            # R_VOC
            next_step[16, age1] = compartments[16, age1, i] + new_R_VOC
            # L_V1i_VOC
            next_step[17, age1] = compartments[17, age1, i] + new_L_fromV1i_VOC + new_L_fromV1r_VOC - new_I_fromV1i_VOC
            # I_V1i_VOC
            next_step[18, age1] = compartments[18, age1, i] + new_I_fromV1i_VOC - new_R_fromV1i_VOC
            # R_V1i_VOC
            next_step[19, age1] = compartments[19, age1, i] + new_R_fromV1i_VOC
            # L_V2i_VOC
            next_step[20, age1] = compartments[20, age1, i] + new_L_fromV2i_VOC + new_L_fromV2r_VOC - new_I_fromV2i_VOC
            # I_V2i_VOC
            next_step[21, age1] = compartments[21, age1, i] + new_I_fromV2i_VOC - new_R_fromV2i_VOC
            # R_V2i_VOC
            next_step[22, age1] = compartments[22, age1, i] + new_R_fromV2i_VOC

            # update recovered
            recovered[age1, i + 1] += new_R
            recovered_VOC[age1, i + 1] += new_R_VOC
            recovered_V1i[age1, i + 1] += new_R_fromV1i
            recovered_V2i[age1, i + 1] += new_R_fromV2i
            recovered_V1i_VOC[age1, i + 1] += new_R_fromV1i_VOC
            recovered_V2i_VOC[age1, i + 1] += new_R_fromV2i_VOC

            # update incidence
            incidence_TOT[age1, i + 1] += new_I + new_I_fromV1i + new_I_fromV2i + new_I_VOC + new_I_fromV1i_VOC + new_I_fromV2i_VOC
            incidence[age1, i + 1] += new_I + new_I_fromV1i + new_I_fromV2i
            incidence_VOC[age1, i + 1] += new_I_VOC + new_I_fromV1i_VOC + new_I_fromV2i_VOC

        # update solution at the next step
        compartments[:, :, i + 1] = next_step

        # VOC introduction
        if dates[i] == start_date_VOC:
            compartments[:, :, i + 1] = get_VOC_introduction(I_today=incidence_TOT[:, i + 1].sum(), perc_VOC=perc_VOC, C=Cs[dates[i]], Nk=Nk,
                                                             compartments_step=compartments[:, :, i + 1], params=params)


    return {"compartments": compartments,
            "recovered": recovered,
            "recovered_V1i": recovered_V1i,
            "recovered_V2i": recovered_V2i,
            "recovered_VOC": recovered_VOC,
            "recovered_V1i_VOC": recovered_V1i_VOC,
            "recovered_V2i_VOC": recovered_V2i_VOC,
            "incidence_TOT": incidence_TOT,
            "incidence": incidence,
            "incidence_VOC": incidence_VOC,
            "vaccines_per_step_1dose": vaccines_per_step_1dose,
            "vaccines_per_step_2dose": vaccines_per_step_2dose}


