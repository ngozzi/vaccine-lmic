# libraries
import numpy as np
import pandas as pd
from datetime import datetime
import os


class Basin:
    """
    This class create a basin object
    """

    def __init__(self, name, path_to_data):

        # name
        self.name = name
        
        # hemisphere
        self.hemispheres = pd.read_csv(os.path.join(path_to_data, name, "hemisphere.csv"))
        
        # VOC introduction 
        df_voc_introduction = pd.read_csv(os.path.join(path_to_data, name, "variants/delta_prevalence.csv"))
        date = pd.to_datetime(df_voc_introduction.loc[df_voc_introduction.prevalence_rolling > 0.05]['date'].values[0])
        self.start_date_VOC = datetime(date.year, date.month, date.day)
        self.perc_VOC = 5.0 / 100.0

        # contacts matrix
        self.contacts_matrix = np.load(os.path.join(path_to_data, name, "contacts-matrix/contacts_matrix_all_10x10.npz"))["arr_0"]
        self.contacts_home = np.load(os.path.join(path_to_data, name, "contacts-matrix/household_layer_10x10.npz"))["arr_0"]
        self.contacts_work = np.load(os.path.join(path_to_data, name, "contacts-matrix/workplace_layer_10x10.npz"))["arr_0"]
        self.contacts_school = np.load(os.path.join(path_to_data, name, "contacts-matrix/school_layer_10x10.npz"))["arr_0"]
        self.contacts_community = np.load(os.path.join(path_to_data, name, "contacts-matrix/community_layer_10x10.npz"))["arr_0"]
        
        # demographic
        self.Nk = pd.read_csv(os.path.join(path_to_data, name, "demographic/Nk_10.csv")).value.values

        # epidemiological data
        self.epi_data_cases = pd.read_csv(os.path.join(path_to_data, name, "epidemic-data/cases.csv"))
        self.epi_data_cases.date = pd.to_datetime(self.epi_data_cases.date)
        self.epi_data_deaths = pd.read_csv(os.path.join(path_to_data, name, "epidemic-data/deaths.csv"))
        self.epi_data_deaths.date = pd.to_datetime(self.epi_data_deaths.date)
        
        # IHME data 
        #self.ihme_data = pd.read_csv(os.path.join(path_to_data, name, "epidemic-data/ihme_data.csv"))
        #self.ihme_data.date = pd.to_datetime(self.ihme_data.date)

        # contacts reductions
        self.reductions = pd.read_csv(os.path.join(path_to_data, name, "restrictions/reductions.csv"))
        
        # vaccinations
        self.vaccinations = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations.csv"))
        self.vaccinations.Day = pd.to_datetime(self.vaccinations.Day)
        
        self.vaccinations_eu_start = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_eu_start.csv"))
        self.vaccinations_eu_start.Day = pd.to_datetime(self.vaccinations_eu_start.Day)

        self.vaccinations_us_start = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_us_start.csv"))
        self.vaccinations_us_start.Day = pd.to_datetime(self.vaccinations_us_start.Day)
        
        self.vaccinations_ita = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_ita.csv"))
        self.vaccinations_ita.Day = pd.to_datetime(self.vaccinations_ita.Day)
        
        self.vaccinations_ita_rescale = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_ita_rescale.csv"))
        self.vaccinations_ita_rescale.Day = pd.to_datetime(self.vaccinations_ita_rescale.Day)
    
        self.vaccinations_uk = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_uk.csv"))
        self.vaccinations_uk.Day = pd.to_datetime(self.vaccinations_uk.Day)
        
        self.vaccinations_uk_rescale = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_uk_rescale.csv"))
        self.vaccinations_uk_rescale.Day = pd.to_datetime(self.vaccinations_uk_rescale.Day)
        
        self.vaccinations_us = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_us.csv"))
        self.vaccinations_us.Day = pd.to_datetime(self.vaccinations_us.Day)
        
        self.vaccinations_us_rescale = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_us_rescale.csv"))
        self.vaccinations_us_rescale.Day = pd.to_datetime(self.vaccinations_us_rescale.Day)

        self.vaccinations_eu_rescale = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_eu_rescale.csv"))
        self.vaccinations_eu_rescale.Day = pd.to_datetime(self.vaccinations_eu_rescale.Day)

        self.vaccinations_isrl_rescale = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/vaccinations_isrl_rescale.csv"))
        self.vaccinations_isrl_rescale.Day = pd.to_datetime(self.vaccinations_isrl_rescale.Day)
        
        healthcare_w = pd.read_csv(os.path.join(path_to_data, name, "vaccinations/healthcare_workers.csv"))
        self.tot_healthcare_workers = healthcare_w["tot_medical_doctors"].values[0] +  healthcare_w["tot_nursing_midwifery"].values[0]
        
        # ifr 
        self.ifr_base = pd.read_csv(os.path.join(path_to_data, name, "ifr_base/ifr_base.csv"))["ifr"].values