"""
Copyright (c) 2023, Met Office
All rights reserved.
"""


def cmip5_names():
    """
    List of all available CMIP model names
    Ref: Slangen et al. (2014)
    :return: List of all CMIP5 models
    """
    print(f'Getting data for all CMIP5 models')

    models = ['ACCESS1-0',
              'bcc-csm1-1',
              'CNRM-CM5',
              'CSIRO-Mk3-6-0',
              'CanESM2',
              'GFDL-ESM2G',
              'GFDL-ESM2M',
              'GISS-E2-R',
              'HadGEM2-CC',
              'HadGEM2-ES',
              'inmcm4',
              'IPSL-CM5A-LR',
              'IPSL-CM5A-MR',
              'MIROC-ESM',
              'MIROC-ESM-CHEM',
              'MIROC5',
              'MPI-ESM-LR',
              'MPI-ESM-MR',
              'MRI-CGCM3',
              'NorESM1-M',
              'NorESM1-ME'
              ]

    return models


def cmip5_names_marginal():
    """
    List of all available CMIP model names that are appropriate for use in
    marginal seas e.g. Mediterranean. Marginal sea is where the native grid of
    the CMIP model has the 'marginal sea' disconnected from the wider ocean
    basin. AR5 recommends excluding these from analysis as values are
    unrealistic.
    Ref: https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.13_SM.
    1.16.14.pdf
    :return: List of CMIP5 models used in AR5 for marginal seas
    """

    print(f'Getting data for subset of CMIP models - marginal selected')
    models = ['bcc-csm1-1',
              'CanESM2',
              'GFDL-ESM2M',
              'HadGEM2-CC',
              'HadGEM2-ES',
              'MIROC-ESM',
              'MIROC-ESM-CHEM',
              'MIROC5'
              ]
    return models
