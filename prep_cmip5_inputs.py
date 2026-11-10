import pdb
import numpy as np
import pickle
import matplotlib.pyplot as plt

def main():

    cmip5_dir = '/data/users/lesley.allison/project_sea_level_copy/sea_level_projections/cmip5_timeseries/'

    scenario = 'rcp26'
    
    with open (cmip5_dir+scenario+'_zostoga_tas_timeseries.pickle', 'rb') as f:
        data=pickle.load(f, encoding='latin1')  # encoding to get around UnicodeDecodeError due to old pickle (python2)

    zostoga_years = data['zostoga_yrs']
    tas_years = data['tas_yrs']
    tas_series = data['tas_series']
    zostoga_series = data['zostoga_series']



    
    models = list(data['tas_series'].keys())
    print(models)



    
               
    pdb.set_trace()



if __name__ == '__main__':
    main()
