from typing import List
from functools import partial
import os

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from pprint import pformat

from oven.data_structures import Product
from oven.phy_props import moisture_content

path = os.path.join(os.path.dirname(__file__), "Copy of Nephi Oven Mole Data Summary.xlsx")
def ftk(x): return (x - 32)*5/9 + 273 # Fahrenheit to Kelvin

# location from 1-20
# TODO this is the input file with all parameters

#############################################################################
## Product specific data

ProductConstants = {
    
    "R13" : {
        # product properties
        "ntimes" : round(24*60/102) + 1, # residence time + 1
        "product_height" : 0.1, # m
        "density_product" : 0.475 * 16.018, # (lb/ft3 * constant) = kg/m3
        "density_air" : 1, # Density of the air in kg/m3 
        "radius_particle" : 4e-6, # Radius of the particle in m
        "enthalpy_vaporization_water" : 2e6, # Enthalpy of vaporization of water in 
        "specific_heat_capacity_air" : 1047, # Specific heat capacity of the air in j/kg/k
        "density_particle" : 2500, # Density of the particle in kg/m3
        "specific_heat_capacity_solid" : 800, # Specific heat capacity of the particle in j/kg/K
        "equilibrium_moisture" : 0.008, # Equilibrium moisture content in dry basis (kg water / kg of dry solid)
    },
    
    "R19" : {
        # product properties
        "ntimes" : round(24*60/88) + 1, # residence time + 1
        "product_height" : 0.24, # m
        "density_product" : 0.314 * 16.018, # (lb/ft3 * constant) = kg/m3
        "density_air" : 1, # Density of the air in kg/m3 
        "radius_particle" : 4e-6, # Radius of the particle in m
        "enthalpy_vaporization_water" : 2e6, # Enthalpy of vaporization of water in 
        "specific_heat_capacity_air" : 1047, # Specific heat capacity of the air in j/kg/k
        "density_particle" : 2500, # Density of the particle in kg/m3
        "specific_heat_capacity_solid" : 800, # Specific heat capacity of the particle in j/kg/K
        "equilibrium_moisture" : 0.008, # Equilibrium moisture content in dry basis (kg water / kg of dry solid)
    },
    
    "R30" : {
        # product properties
        "ntimes" : round(24*60/35) + 1, # residence time + 1
        "product_height" : 0.33, # m
        "density_product" : 0.426 * 16.018, # (lb/ft3 * constant) = kg/m3
        "density_air" : 1, # Density of the air in kg/m3 
        "radius_particle" : 4e-6, # Radius of the particle in m
        "enthalpy_vaporization_water" : 2e6, # Enthalpy of vaporization of water in 
        "specific_heat_capacity_air" : 1047, # Specific heat capacity of the air in j/kg/k
        "density_particle" : 2500, # Density of the particle in kg/m3
        "specific_heat_capacity_solid" : 800, # Specific heat capacity of the particle in j/kg/K
        "equilibrium_moisture" : 0.008, # Equilibrium moisture content in dry basis (kg water / kg of dry solid)
    }
}

OvenConstants = {

    "R13" : {
        # simulation propeties
        "ny" : 20,
        "nzones" : 5,
        "reverse_zone" : 3,
        **ProductConstants["R13"]
    }, 

    "R19" : {
        # simulation propeties
        "ny" : 20,
        "nzones" : 5,
        "reverse_zone" : 3,
        **ProductConstants["R19"]
    },

    "R30" : {
        # simulation propeties
        "ny" : 20,
        "nzones" : 5,
        "reverse_zone" : 3,
        **ProductConstants["R30"]
    }
}

#############################################################################
## Experiment specific data

_Target = {

    "R13SSR1" :  Product(
        targets = { # use the first row as the initial condition of solid
            5 : np.average(pd.read_excel(path, sheet_name = "R13 South Side Run 1", header = 40 - 2, usecols = [3, 4], 
                                          names = ["T2", "T3"], converters = {"T2" : ftk, "T3" : ftk}).values, axis = 1), # bottom
            # 10 : pd.read_excel(path, sheet_name = "R13 South Side Run 1", header =  40 - 1, usecols = [2], 
            #                    names = ["T1"], converters = {"T1" : ftk}).values.flatten() # mid
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([535., 535., 560., 565., 565.]) - 32)*5/9 + 273 - 60, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./102)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]), # jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        targets_mask = {}
    ),

    "R13SSR2" : Product( 
        targets = { # use the first row as the initial condition of solid
            # -3 : pd.read_excel(path, sheet_name = "R13 South Side Run 2", header = 55 - 1, usecols = [2], 
            #                    names = ["T1"], converters = {"T1" : ftk}).values.flatten(), # top
            5 : np.average(pd.read_excel(path, sheet_name = "R13 South Side Run 2", header = 55 - 2, usecols = [3, 4],
                              names = ["T2", "T3"], converters = {"T2" : ftk, "T3" : ftk}).values, axis = 1), # bottom
            # 5 : pd.read_excel(path, sheet_name = "R13 South Side Run 2", header = 55 - 1, usecols = [4], 
            #                   names = ["T1"], converters = {"T1" : ftk}).values.flatten() # mid
        },
        controls = { 
            "init_velocity_air" : jnp.array([0.107, 0.13, 0.115, 0.207, 0.207]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([525., 525., 550., 550., 550.]) - 32)*5/9 + 273 - 60, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./102)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]), # jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        targets_mask = {}
    ),

    "R13SSR4" : Product( 
        targets = { # use the first row as the initial condition of solid
            5 : np.average(pd.read_excel(path, sheet_name = "R13 South Side Run 4", header = 36 - 2, usecols = [3, 4], 
                                          names = ["T2", "T3"], converters = {"T2" : ftk, "T3" : ftk}).values, axis = 1) # bottom
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.087, 0.097, 0.154, 0.207, 0.207]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([520., 515., 550., 550., 550.]) - 32)*5/9 + 273 - 60, # K
            "residence_time" : jnp.atleast_1d([jnp.round(24*60./102)]), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]), # jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        targets_mask = {}
    ),

    "R13NSR1" : Product(
        targets = { # use the first row as the initial condition of solid
            # 5 : pd.read_excel(path, sheet_name = "R13 North Side Run 1", header = 62 - 1, usecols = [2], 
            #                   names = ["T1"], converters = {"T1" : ftk}).values.flatten(),
            5 : np.average(pd.read_excel(path, sheet_name = "R13 North Side Run 1", header = 62 - 2, usecols = [3, 4], 
                                          names = ["T2", "T3"], converters = {"T2" : ftk, "T3" : ftk}).values, axis = 1) # bottom
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([535., 535., 560., 565., 565.]) - 32)*5/9 + 273 - 60, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./102)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]), # jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        targets_mask = {}
    ),

    "R13NSR2" : Product(
        targets = { # use the first row as the initial condition of solid
            5 : np.average(pd.read_excel(path, sheet_name = "R13 North Side Run 2", header = 25 - 2, usecols = [3, 4], 
                                          names = ["T2", "T3"], converters = {"T2" : ftk, "T3" : ftk}).values, axis = 1), # bottom
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([535., 535., 560., 565., 565.]) - 32)*5/9 + 273 - 60, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./102)), ## 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]), # jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        targets_mask = {}
    ),

    "R19SSR1" : Product(
        targets = { # use the first row as the initial condition of solid
            # -3  : pd.read_excel(path, sheet_name = "R19 South Side Run 1", header = 64 - 1, usecols = [2], 
            #                     names = ["T1"], converters = {"T1" : ftk}).values.flatten(), # top
            5 : pd.read_excel(path, sheet_name = "R19 South Side Run 1", header = 64 - 2, usecols = [3], 
                              names = ["T2"], converters = {"T2" : ftk}).values.flatten(), # bottom
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.104, 0.118, 0.181, 0.181, 0.181]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([510., 510., 540., 535., 535.]) - 32)*5/9 + 273, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./88)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        },
        targets_mask = {}
    ),

    "R19NSR1" : Product(
        targets = { # use the first row as the initial condition of solid
            # 10  : pd.read_excel(path, sheet_name = "R19 North Side Run 1", header = 31 - 1, usecols = [2], 
            #                     names = ["T1"], converters = {"T1" : ftk}).values.flatten(), # top
            5 : np.average(pd.read_excel(path, sheet_name = "R19 North Side Run 1", header = 31 - 2, usecols = [3, 4], 
                                          names = ["T2", "T3"], converters = {"T2" : ftk, "T3" : ftk}).values, axis = 1), # bottom
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.104, 0.118, 0.181, 0.181, 0.181]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([510., 510., 540., 535., 535.]) - 32)*5/9 + 273, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./88)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        },
        targets_mask = {}
    ),

    "R19NSR2" : Product(
        targets = { # use the first row as the initial condition of solid
            5 : pd.read_excel(path, sheet_name = "R19 North Side Run 2", header = 30 - 2, usecols = [3], 
                              names = ["T2"], converters = {"T2" : ftk}).values.flatten(), # bottom
            #  : np.average(pd.read_excel(path, sheet_name = "R19 North Side Run 2", header = 30, usecols = [2, 4], 
            #                             names = ["T1", "T2"], converters = {"T1" : ftk, "T2" : ftk}).values, axis = 1),
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.104, 0.118, 0.181, 0.181, 0.181]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([510., 510., 540., 535., 535.]) - 32)*5/9 + 273, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./88)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        },
        targets_mask = {}
    ),

    "R30SSR1" : Product(
        targets = { # use the first row as the initial condition of solid
            5 : pd.read_excel(path, sheet_name = "R30 South Side Run 1", header = 44 - 2, usecols = [4], 
                              names = ["T3"], converters = {"T3" : ftk}).values.flatten(), # bottom
            #  : pd.read_excel(path, sheet_name = "R30 North Side Run 1", header = 44, usecols = [3], 
            #                  names = ["T1"], converters = {"T1" : ftk}).values.flatten(),
            #  : pd.read_excel(path, sheet_name = "R30 North Side Run 1", header = 44, usecols = [4], 
            #                  names = ["T1"], converters = {"T1" : ftk}).values.flatten(),
        },
        controls = { # zone 5 temperature and velocity are assumed to be same as zone 4
            "init_velocity_air" : jnp.array([0.087, 0.13, 0.118, 0.24, 0.24]), # converted from excel sheet
            "init_temperature_air" : (jnp.array([510., 515., 535., 535., 535.]) - 32)*5/9 + 273, # K
            "residence_time" : jnp.atleast_1d(jnp.round(24*60./35)), # 24 ft per zone * 60 sec / linespeed
            "init_moisture_air" : jax.vmap(moisture_content, in_axes = (None, None, 0))(273, 1, jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])),
            "recirculation_ratio" : jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        },
        targets_mask = {} 
    )
}


def add_map(adict):

    for _keys, _product in adict.items():
        _ny = OvenConstants[_keys[:3]]["ny"]
        
        # create map
        k = list(_product.targets.keys())
        _mask = np.zeros(_ny)
        _mask[k] = 1

        # create target map
        row_max = 1000
        _target_map = np.zeros((row_max, _ny))
        for key, value in _product.targets.items():
            _target_map[:, key] += np.pad(value, (0, row_max - len(value)), "constant", constant_values = (0, 0))

        _product.targets_mask["mask"] = jnp.array(_mask)
        _product.targets_mask["target_map"] = jnp.array(_target_map)
        
    return adict

Target = add_map(_Target)

def get_targets(expts : List[str], products : List[str], logger = None):
    # Function to log all the data

    if logger is not None : 
        for prod in products : 
            logger.info(f"{'--'*50} \nProduct and simulation specific oven constants \n{prod} : {pformat(OvenConstants[prod])}")
        
        for _expt in expts :
            logger.info(f"{'--'*50} \nExperiment specific controls and targets \n{_expt} : {pformat(_Target[_expt])}")
    
    return OvenConstants, Target
