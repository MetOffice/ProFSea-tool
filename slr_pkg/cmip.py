# """
# Copyright (c) 2023, Met Office
# All rights reserved.
# """


def model_dictionary():
    """
    Define a dictionary that contains metadata of CMIP model experiment data
    ranges, used in filename convention
    :return: model_dict
    """
    model_dict = {'ACCESS1-0': {'historical': '185001-200512'},
                  'bcc-csm1-1': {'historical': '185001-201212'},
                  'CanESM2': {'historical': '185001-200512'},
                  'CNRM-CM5': {'historical': '185001-185912'},
                  'CSIRO-Mk3-6-0': {'historical': '185001-200512'},
                  'GISS-E2-R': {'historical': '185001-200512'},
                  'GFDL-ESM2G': {'historical': '186101-186512'},
                  'GFDL-ESM2M': {'historical': '186101-186512'},
                  'HadGEM2-CC': {'historical': '186001-186012'},
                  'HadGEM2-ES': {'historical': '186001-186012'},
                  'inmcm4': {'historical': '185001-200512'},
                  'IPSL-CM5A-LR': {'historical': '185001-200512'},
                  'IPSL-CM5A-MR': {'historical': '185001-200512'},
                  'MIROC-ESM': {'historical': '185001-200512'},
                  'MIROC-ESM-CHEM': {'historical': '185001-200512'},
                  'MIROC5': {'historical': '185001-201212'},
                  'MPI-ESM-LR': {'historical': '185001-200512'},
                  'MPI-ESM-MR': {'historical': '185001-189912'},
                  'MRI-CGCM3': {'historical': '185001-200512'},
                  'NorESM1-M': {'historical': '185001-200512'},
                  'NorESM1-ME': {'historical': '185001-200512'}}
    return model_dict


def zos_dictionary():
    """
    Define a dictionary that contains metadata of CMIP model experiment data
    ranges, used in filename convention. Where a model has not been run for
    a specific RCP then 'xxxxxx-xxxxxx' is included.
    :return: zos_dict
    """
    zos_dict = {'ACCESS1-0': {'rcp26': {'driftcorr': 'xxxxxx-xxxxxx',
                                        'zostoga': 'xxxxxx-xxxxxx',
                                        'piControl': 'xxxxxx-xxxxxx'},
                              'rcp45': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '030001-080001'},
                              'rcp85': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '030001-080001'}},
                'bcc-csm1-1': {'rcp26': {'driftcorr': '200601-230101',
                                         'zostoga': '200601-230101',
                                         'piControl': '000101-050101'},
                               'rcp45': {'driftcorr': '200601-230101',
                                         'zostoga': '200601-230101',
                                         'piControl': '000101-050101'},
                               'rcp85': {'driftcorr': '200601-230101',
                                         'zostoga': '200601-230101',
                                         'piControl': '000101-050101'}},
                'CanESM2': {'rcp26': {'driftcorr': '200601-230101',
                                      'zostoga': '200601-230101',
                                      'piControl': '201501-301101'},
                            'rcp45': {'driftcorr': '200601-230101',
                                      'zostoga': '200601-230101',
                                      'piControl': '201501-301101'},
                            'rcp85': {'driftcorr': '200601-210101',
                                      'zostoga': '200601-210101',
                                      'piControl': '201501-301101'}},
                'CNRM-CM5': {'rcp26': {'driftcorr': '200601-210101',
                                       'zostoga': '200601-210101',
                                       'piControl': '185001-269912'},
                             'rcp45': {'driftcorr': '200601-230101',
                                       'zostoga': '200601-230101',
                                       'piControl': '185001-269912'},
                             'rcp85': {'driftcorr': '200601-230101',
                                       'zostoga': '200601-230101',
                                       'piControl': '185001-269912'}},
                'CSIRO-Mk3-6-0': {'rcp26': {'driftcorr': '200601-210101',
                                            'zostoga': '200601-210101',
                                            'piControl': '000101-050101'},
                                  'rcp45': {'driftcorr': '200601-210101',
                                            'zostoga': '200601-210101',
                                            'piControl': '000101-050101'},
                                  'rcp85': {'driftcorr': '200601-210101',
                                            'zostoga': '200601-210101',
                                            'piControl': '000101-050101'}},
                'GISS-E2-R': {'rcp26': {'driftcorr': '200601-230101',
                                        'zostoga': '200601-230101',
                                        'piControl': '333101-453101'},
                              'rcp45': {'driftcorr': '200601-230101',
                                        'zostoga': '200601-230101',
                                        'piControl': '333101-453101'},
                              'rcp85': {'driftcorr': '200601-230101',
                                        'zostoga': '200601-230101',
                                        'piControl': '333101-453101'}},
                'GFDL-ESM2G': {'rcp26': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '186101-210001'},
                               'rcp45': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '186101-210001'},
                               'rcp85': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '186101-210001'}},
                'GFDL-ESM2M': {'rcp26': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '186101-210001'},
                               'rcp45': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '186101-210001'},
                               'rcp85': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '186101-210001'}},
                'HadGEM2-CC': {'rcp26': {'driftcorr': 'xxxxxx-xxxxxx',
                                         'zostoga': 'xxxxxx-xxxxxx',
                                         'piControl': 'xxxxxx-xxxxxx'},
                               'rcp45': {'driftcorr': '200512-210101',
                                         'zostoga': '200512-210101',
                                         'piControl': '185912-210001'},
                               'rcp85': {'driftcorr': '200512-210101',
                                         'zostoga': '200512-210101',
                                         'piControl': '185912-210001'}},
                'HadGEM2-ES': {'rcp26': {'driftcorr': '200512-230001',
                                         'zostoga': '200512-230001',
                                         'piControl': '185912-243606'},
                               'rcp45': {'driftcorr': '200512-230001',
                                         'zostoga': '200512-230001',
                                         'piControl': '185912-243606'},
                               'rcp85': {'driftcorr': '200512-230001',
                                         'zostoga': '200512-230001',
                                         'piControl': '185912-243606'}},
                'inmcm4': {'rcp26': {'driftcorr': 'xxxxxx-xxxxxx',
                                     'zostoga': 'xxxxxx-xxxxxx',
                                     'piControl': 'xxxxxx-xxxxxx'},
                           'rcp45': {'driftcorr': '200601-210101',
                                     'zostoga': '200601-210101',
                                     'piControl': '185001-235001'},
                           'rcp85': {'driftcorr': '200601-210101',
                                     'zostoga': '200601-210101',
                                     'piControl': '185001-235001'}},
                'IPSL-CM5A-LR': {'rcp26': {'driftcorr': '200601-230101',
                                           'zostoga': '200601-230101',
                                           'piControl': '180001-280001'},
                                 'rcp45': {'driftcorr': '200601-230101',
                                           'zostoga': '200601-230101',
                                           'piControl': '180001-280001'},
                                 'rcp85': {'driftcorr': '200601-230101',
                                           'zostoga': '200601-230101',
                                           'piControl': '180001-280001'}},
                'IPSL-CM5A-MR': {'rcp26': {'driftcorr': '200601-210101',
                                           'zostoga': '200601-210101',
                                           'piControl': '180001-210001'},
                                 'rcp45': {'driftcorr': '200601-230101',
                                           'zostoga': '200601-230101',
                                           'piControl': '180001-210001'},
                                 'rcp85': {'driftcorr': '200601-210101',
                                           'zostoga': '200601-210101',
                                           'piControl': '180001-210001'}},
                'MIROC-ESM': {'rcp26': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '180001-248001'},
                              'rcp45': {'driftcorr': '200601-230101',
                                        'zostoga': '200601-230101',
                                        'piControl': '180001-248001'},
                              'rcp85': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '180001-248001'}},
                'MIROC-ESM-CHEM': {'rcp26': {'driftcorr': '200601-210101',
                                             'zostoga': '200601-210101',
                                             'piControl': '184601-210101'},
                                   'rcp45': {'driftcorr': '200601-210101',
                                             'zostoga': '200601-210101',
                                             'piControl': '184601-210101'},
                                   'rcp85': {'driftcorr': '200601-210101',
                                             'zostoga': '200601-210101',
                                             'piControl': '184601-210101'}},
                'MIROC5': {'rcp26': {'driftcorr': '200601-210101',
                                     'zostoga': '200601-210101',
                                     'piControl': '200001-267001'},
                           'rcp45': {'driftcorr': '200601-210101',
                                     'zostoga': '200601-210101',
                                     'piControl': '200001-267001'},
                           'rcp85': {'driftcorr': '200601-210101',
                                     'zostoga': '200601-210101',
                                     'piControl': '200001-267001'}},
                'MPI-ESM-LR': {'rcp26': {'driftcorr': '200601-230101',
                                         'zostoga': '200601-230101',
                                         'piControl': '185001-285001'},
                               'rcp45': {'driftcorr': '200601-230101',
                                         'zostoga': '200601-230101',
                                         'piControl': '185001-285001'},
                               'rcp85': {'driftcorr': '200601-230101',
                                         'zostoga': '200601-230101',
                                         'piControl': '185001-285001'}},
                'MPI-ESM-MR': {'rcp26': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '185001-285001'},
                               'rcp45': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '185001-285001'},
                               'rcp85': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '185001-285001'}},
                'MRI-CGCM3': {'rcp26': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '185101-235101'},
                              'rcp45': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '185101-235101'},
                              'rcp85': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '185101-235101'}},
                'NorESM1-M': {'rcp26': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '070001-120101'},
                              'rcp45': {'driftcorr': '200601-230101',
                                        'zostoga': '200601-230101',
                                        'piControl': '070001-120101'},
                              'rcp85': {'driftcorr': '200601-210101',
                                        'zostoga': '200601-210101',
                                        'piControl': '070001-120101'}},
                'NorESM1-ME': {'rcp26': {'driftcorr': '200601-210201',
                                         'zostoga': '200601-210201',
                                         'piControl': '090101-115301'},
                               'rcp45': {'driftcorr': '200601-210301',
                                         'zostoga': '200601-210301',
                                         'piControl': '090101-115301'},
                               'rcp85': {'driftcorr': '200601-210101',
                                         'zostoga': '200601-210101',
                                         'piControl': '090101-115301'}}}
    return zos_dict
