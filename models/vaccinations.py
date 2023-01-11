# libraries
import numpy as np

# n. of compartments and age groups
ncomp = 23
nage = 10


def get_vaccinated_2dose(vaccinations, compartments_step, date, seed=None):
    """
    This function compute the number of daily second doses in different age groups
    :param vaccinations: dictionary of vaccinations
    :param compartments_step: individuals in different compartments at this step
    :param date: current date
    :param seed: random seed
    :return: updated compartments and doses given by age group
    """

    # set random seed
    if seed != None:
        np.random.seed(seed)
    
    # doses given by age group
    vaccines_given_2dose = np.zeros(nage)
    
    # vaccinations for the day
    day_vaccine = vaccinations.loc[vaccinations.Day == date]
    
    # give second doses
    if day_vaccine.shape[0] == 0:
        tot_vaccine_2dose = 0       
    else:
        tot_vaccine_2dose = day_vaccine["two_dose_daily"].values[0] 
    left_vaccine_2dose = tot_vaccine_2dose
    
    # distribute 2 doses homogeneously to those that received the first dose
    den = 0
    for age_vacc in np.arange(1, nage):
        den += compartments_step[5, age_vacc] + compartments_step[6, age_vacc] + compartments_step[8, age_vacc] + compartments_step[17, age_vacc] + compartments_step[19, age_vacc] 
        
    # if there's still someone to vaccinate
    if den != 0:

        # iterate over the age groups 
        for age_vacc in np.arange(1, nage):

            # if there are still doses
            if left_vaccine_2dose > 0:

                # distribute among S / L / R
                v_from_V1i = int(min(compartments_step[5, age_vacc], left_vaccine_2dose * compartments_step[5, age_vacc] / den))
                v_from_LV1i = int(min(compartments_step[6, age_vacc], left_vaccine_2dose * compartments_step[6, age_vacc] / den))
                v_from_RV1i = int(min(compartments_step[8, age_vacc], left_vaccine_2dose * compartments_step[8, age_vacc] / den))
                
                v_from_LV1i_VOC = int(min(compartments_step[17, age_vacc], left_vaccine_2dose * compartments_step[17, age_vacc] / den))
                v_from_RV1i_VOC = int(min(compartments_step[19, age_vacc], left_vaccine_2dose * compartments_step[19, age_vacc] / den))

                # V1i
                compartments_step[5, age_vacc] -= v_from_V1i
                compartments_step[9, age_vacc] += v_from_V1i

                # L_V1i
                compartments_step[6, age_vacc] -= v_from_LV1i
                compartments_step[11, age_vacc] += v_from_LV1i

                # R_V1i
                compartments_step[8, age_vacc] -= v_from_RV1i
                compartments_step[13, age_vacc] += v_from_RV1i
                
                # L_V1i_VOC
                compartments_step[17, age_vacc] -= v_from_LV1i_VOC
                compartments_step[20, age_vacc] += v_from_LV1i_VOC
                
                # R_V1i_VOC 
                compartments_step[19, age_vacc] -= v_from_RV1i_VOC
                compartments_step[22, age_vacc] += v_from_RV1i_VOC

                # update doses given and left
                vaccines_given_2dose[age_vacc] += (v_from_V1i + v_from_LV1i + v_from_RV1i + v_from_LV1i_VOC + v_from_RV1i_VOC)
                left_vaccine_2dose -= (v_from_V1i + v_from_LV1i + v_from_RV1i + v_from_LV1i_VOC + v_from_RV1i_VOC)
                
    return compartments_step, vaccines_given_2dose
    


def get_vaccinated_1dose(vaccine, vaccinations, compartments_step, date, seed=None):
    """
    This function compute the number of daily first doses in different age groups for different allocation strategies
    :param vaccine: vaccine allocation strategy
    :param vaccinations: dictionary of vaccinations
    :param compartments_step: individuals in different compartments at this step
    :param date: current date
    :param seed: random seed
    :return: updated compartments and doses given by age group
    """

    # set random seed
    if seed != None:
        np.random.seed(seed)

    # doses given by age group
    vaccines_given_1dose = np.zeros(nage)
    
    # vaccinations for the day
    day_vaccine = vaccinations.loc[vaccinations.Day == date]

    if vaccine == "age-order":

        # total number of vaccines available for this step
        if day_vaccine.shape[0] == 0:
            tot_vaccine = 0
        else:
            tot_vaccine = day_vaccine["one_dose_daily"].values[0]
            
        left_vaccine = tot_vaccine

        # 9: 80+, 8: 70-79, 7: 60-69, 6: 50-59, 5: 40-49, 4: 30-39, 3: 25-29, 2: 20-24, 1: 10-19, 0: 0-9
        # distribute in decreasing order of age up to 50+
        for age_vacc in np.arange(nage - 1, 5, -1):
            # total number of people that can be vaccinated
            den = compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[3, age_vacc] + compartments_step[14, age_vacc] + compartments_step[16, age_vacc]

            # if there's still someone to vaccinate
            if den != 0 and left_vaccine > 0:

                # distribute among S / L / R
                v_to_S = int(min(compartments_step[0, age_vacc], left_vaccine * compartments_step[0, age_vacc] / den))
                v_to_L = int(min(compartments_step[1, age_vacc], left_vaccine * compartments_step[1, age_vacc] / den))
                v_to_R = int(min(compartments_step[3, age_vacc], left_vaccine * compartments_step[3, age_vacc] / den))
                v_to_L_VOC = int(min(compartments_step[14, age_vacc], left_vaccine * compartments_step[14, age_vacc] / den))
                v_to_R_VOC = int(min(compartments_step[16, age_vacc], left_vaccine * compartments_step[16, age_vacc] / den))

                # S
                compartments_step[0, age_vacc] -= v_to_S
                compartments_step[4, age_vacc] += v_to_S

                # L
                compartments_step[1, age_vacc] -= v_to_L
                compartments_step[6, age_vacc] += v_to_L

                # R
                compartments_step[3, age_vacc] -= v_to_R
                compartments_step[8, age_vacc] += v_to_R
                
                # L_VOC
                compartments_step[14, age_vacc] -= v_to_L_VOC
                compartments_step[17, age_vacc] += v_to_L_VOC
                
                # R_VOC
                compartments_step[16, age_vacc] -= v_to_R_VOC
                compartments_step[19, age_vacc] += v_to_R_VOC

                # update doses given and left
                vaccines_given_1dose[age_vacc] += (v_to_S + v_to_L + v_to_R + v_to_L_VOC + v_to_R_VOC)
                left_vaccine -= (v_to_S + v_to_L + v_to_R + v_to_L_VOC + v_to_R_VOC)

        # give the remaining homogeneously in the under 50 (but not under 10)
        den = 0
        for age_vacc in np.arange(1, 6):
            den += compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[3, age_vacc] + compartments_step[14, age_vacc] + compartments_step[16, age_vacc]

        # if there's still someone to vaccinate
        if den != 0:

            # iterate over the remaining age groups (10-50)
            for age_vacc in np.arange(1, 6):

                # if there are still doses
                if left_vaccine > 0:

                    # distribute among S / L / R
                    v_to_S = int(min(compartments_step[0, age_vacc], left_vaccine * compartments_step[0, age_vacc] / den))
                    v_to_L = int(min(compartments_step[1, age_vacc], left_vaccine * compartments_step[1, age_vacc] / den))
                    v_to_R = int(min(compartments_step[3, age_vacc], left_vaccine * compartments_step[3, age_vacc] / den))
                    v_to_L_VOC = int(min(compartments_step[14, age_vacc], left_vaccine * compartments_step[14, age_vacc] / den))
                    v_to_R_VOC = int(min(compartments_step[16, age_vacc], left_vaccine * compartments_step[16, age_vacc] / den))

                    # S
                    compartments_step[0, age_vacc] -= v_to_S
                    compartments_step[4, age_vacc] += v_to_S

                    # L
                    compartments_step[1, age_vacc] -= v_to_L
                    compartments_step[6, age_vacc] += v_to_L

                    # R
                    compartments_step[3, age_vacc] -= v_to_R
                    compartments_step[8, age_vacc] += v_to_R
                    
                    # L_VOC
                    compartments_step[14, age_vacc] -= v_to_L_VOC
                    compartments_step[17, age_vacc] += v_to_L_VOC
                
                    # R_VOC
                    compartments_step[16, age_vacc] -= v_to_R_VOC
                    compartments_step[19, age_vacc] += v_to_R_VOC

                    # update doses given and left
                    vaccines_given_1dose[age_vacc] += (v_to_S + v_to_L + v_to_R + v_to_L_VOC + v_to_R_VOC)
                    left_vaccine -= (v_to_S + v_to_L + v_to_R + v_to_L_VOC + v_to_R_VOC)         

    return compartments_step, vaccines_given_1dose


def vaccinate(vaccine, vaccinations, compartments_step, date, seed=None):
    """
    This function gives 1st and 2nd doses
    :param vaccine: vaccine allocation strategy
    :param vaccinations: dictionary of vaccinations
    :param compartments_step: individuals in different compartments at this step
    :param date: current date
    :param seed: random seed
    :return: updated compartments and doses given by age group
    """
    compartments_step, vaccines_given_1dose = get_vaccinated_1dose(vaccine, vaccinations, compartments_step, date, seed)
    compartments_step, vaccines_given_2dose = get_vaccinated_2dose(vaccinations, compartments_step, date, seed)
    return compartments_step, vaccines_given_1dose, vaccines_given_2dose
