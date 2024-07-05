import os, os.path, fnmatch
from collections.abc import Sequence

import numpy as np
from scipy.stats import norm
import pandas as pd
import dask.array as da
from config import settings
import matplotlib.pyplot as plt

# First year of AR5 projections
endofhistory=2006

# Last year of AR5 projections
endofAR5=2100
endyr = 2300

# Fraction of SLE from Greenland during 1996 to 2005 assumed to result from
# rapid dynamical change, with the remainder assumed to result from SMB change
fgreendyn=0.5

# m SLE from Greenland during 1996 to 2005 according to AR5 chapter 4
dgreen=(3.21-0.30)*1e-3

# m SLE from Antarctica during 1996 to 2005 according to AR5 chapter 4
dant=(2.37+0.13)*1e-3

# Conversion factor for Gt to m SLE
mSLEoGt=1e12/3.61e14*1e-3

exp_efficiency = 0.12e-24

def vlikely_range(data):
    # Compute median and 5-95% range for the first (or only) axis of data.
    # Return array (stat[,dom1,...]), where stat=0,1,2 for 50-,5-,95-percentile,
    # and dom1,... are any remaining axes of data.
    # NB model 5-95% range is judged to be "likely" for the AR5 projections
    # data -- array-like
    # numpy.array() to convert masked array into unmasked, because percentile()
    # gives a warning if there are no non-masked elements
    return np.percentile(np.array(data),[50,5,95],0)


def actual_range(data):
    # Compute mean and actual range for the first (or only) axis of data
    # Return array (stat[,dom1,...]), where stat=0,1,2 for mean,minimum,maximum
    # of data, and dom1,... are any remaining axes of data.
    # data -- array-like
    return np.array([np.mean(data,0),np.amin(data,0),\
        np.amax(data,0)])
  
  
def project(scenarios=None, **kwargs):
    # input -- str, path to directory containing input files. The directory should
    #   contain files named SCENARIO_QUANTITY_STATISTIC.nc, where QUANTITY is
    #   temperature or expansion, and STATISTIC is mean, sd or models. Each file
    #   contains one field with a dimension of time. The mean and sd fields have
    #   no other dimension and are used if nt>0 (default), while the models fields
    #   have a model dimension and are used if nt==0.
    # scenarios -- str or sequence of str, scenarios for which projections are to be
    #   made, by default all those represented in the input directory
    # output, str, optional -- path to directory in which output files are to be
    #   written. It is created if it does not exist. No files are written if this
    #   argument is omitted.
    # ensemble -- bool, optional, default False, write output files of the ensemble
    #   as well as the statistics.
    # seed -- optional, for numpy.random, default zero
    # nt -- int, optional, number of realisations of the input timeseries for each
    #   scenario, default 450, to be generated using the mean and sd files; specify
    #   0 if the ensemble of individual models is to be used instead, which is read
    #   from the models files.
    # nm -- int, optional, number of realisations of components and of the sum for
    #   each realisation of the input timeseries, default 1000, must be a multiple
    #   of the number of glacier methods.
    # tcv -- float, optional, default 1.0, multiplier for the standard deviation
    #   in the input fields
    # glaciermip -- optional, default False => AR5 parameters, 1 => GlacierMIP
    #   (Hock et al., 2019), 2 => GlacierMIP2 (Marzeion et al., 2020)
    # levermann -- optional, default None, specifies that Antarctic dynamics should
    #   use the fit by Palmer et al. (2020) to the results of Levermann et al.
    #   (2014); if dict, must have an key for each scenario whose value names the
    #   Levermann RCP fit to be used; if str, identifies the single fit to be used
    #   for every scenario; otherwise it is treated as True or False; if True the
    #   scenarios must all be ones that Levermann provides.
    # palmer -- bool, optional, default False, allow integration to end in any year
    #   up to 2300, with the contributions to GMLSR from ice-sheet dynamics, Green-
    #   land SMB and land water storage held at the 2100 rate beyond 2100.


    if scenarios is None:
        # Obtain list of scenarios from the input filenames
        bname=fnmatch.filter(os.listdir(settings['test_input_dir']),'*_*.nc')
        scenarios=[tname.split('_',1)[0] for tname in bname]
        scenarios=sorted(set(scenarios))

    for scenario in scenarios:
        project_scenario(scenario,**kwargs)
        
        
def project_scenario(scenario,output=None,\
    seed=0,nt=450,nm=1000,tcv=1.0,\
    glaciermip=False,ensemble=False,levermann=False,palmer=False):
    # Make GMSLR projection for the specified single scenario
    # Arguments are all the same as project() except for:
    # scenario -- str, name of the scenario
    # levermann -- optional, treated as True/False to specify that Levermann
    #   should be used, if str it identifies the Leverman scenario fit to be used,
    #   otherwise assumed to be the scenario being simulated

    np.random.seed(seed)

    startyr=endofhistory # year when the timeseries for integration begin
        
    # Read the input fields for temperature and expansion into txin. txin has four
    # elements if nt>0: temperature mean, temperature sd, expansion mean, expansion
    # sd. txin has two elements if nt==0, for temperature and expansion, each
    # having a model dimension. Check that each field is one-dimensional in time,
    # that the mean and sd fields for each quantity have equal time axes, that
    # temperature applies to calendar years (indicated by its time bounds), that
    # expansion applies at the ends of the calendar years of temperature, that
    # there is no missing data, and that the model axis (if present) is the same
    # for the two quantities.
    variable = ['temperature','ocean_heat_content_change'] # input quantities

    txin = []
    for v in variable:
        file = os.path.join(settings['test_input_dir'], 'ssprcmip_' + scenario + '_'+ v + '.csv')
        df = pd.read_csv(file)
        df_2300 = df.loc[:, '2007':'2300']
        central_estimate = np.percentile(df_2300.to_numpy(), 0, axis=0)
        std = np.std(df_2300.to_numpy(), axis=0)

        if v == 'ocean_heat_content_change':
            central_estimate = central_estimate * exp_efficiency # convert to thermal expansion 
            std = np.std(df_2300.to_numpy() * exp_efficiency, axis=0)
        
        txin.extend([central_estimate, std])

    nyr=txin[0].shape[0] # number of years in the timeseries

    # Integrate temperature to obtain K yr at ends of calendar years, replacing
    # the time-axis of temperature (which applies at mid-year) with the time-axis
    # of expansion (which applies at year-end)

    itin = [np.cumsum(txin[i]) for i in range(2)]
    
    # Generate a sample of perfectly correlated timeseries fields of temperature,
    # time-integral temperature and expansion, each of them [realisation,time]
    z = np.random.standard_normal(nt)*tcv

    # For each quantity, mean + standard deviation * normal random number
    # reshape to [realisation,time]
    
    zt = z[:, np.newaxis] * txin[1] + txin[0]
    zx = z[:, np.newaxis] * txin[3] + txin[2]
    zit = z[:, np.newaxis] * itin[1] + itin[0]
    zitmean = itin[0]
    
    # Create a cf.Field with the shape of the quantities to be calculated
    # [component_realization,climate_realization,time]
    template=np.full([nm, nt, nyr], np.nan)


    # Obtain ensembles of projected components as cf.Field objects and add them up
    # Report the range of the final year and write output files if requested

    expansion = np.tile(zx, (nm, 1))


    glacier = project_glacier(zitmean,zit,template,glaciermip)

    greensmb = project_greensmb(zt,template,palmer)

    fraction = np.random.rand(nm*nt) # correlation between antsmb and antdyn
    antsmb = project_antsmb(zit,template,fraction)


    greendyn = project_greendyn(scenario,template,palmer)
    greennet = greensmb+greendyn

    antdyn = project_antdyn(template, fraction, output, palmer)
    antnet = antsmb+antdyn

    landwater = project_landwater(template,palmer)

    # add expansion last because it has a lower dimensionality and we want it to
    # be broadcast to the same shape as the others rather than messing up gmslr
    gmslr=glacier+greennet+antnet+landwater+expansion

    sheetdyn=greendyn+antdyn
    
    components = [expansion, glacier, greensmb, greendyn, greennet, antsmb, antdyn, antnet, landwater, sheetdyn, gmslr]
    component_names = ['exp', 'glacier', 'greensmb', 'greendyn', 'greennet', 'antsmb', 'antdyn', 'antnet', 'landwater', 'sheetdyn', 'gmslr']
    for i, component in enumerate(components):
        np.save(f'/data/users/gmunday/slr/montecarlo_tests/{scenario}_{component_names[i]}.npy', component)
    
    # fig = plt.figure(figsize=(24, 5))

    # time = np.arange(2007, 2301)
    # for i, component in enumerate(components):
    #     ax = fig.add_subplot(1, 8, i+1)
    #     ax.plot(time, np.percentile(component, 5, axis=0), color='navy', lw=1)
    #     ax.plot(time, np.percentile(component, 50, axis=0), color='black', label='Central estimate')
    #     ax.plot(time, np.percentile(component, 95, axis=0), color='navy', lw=1)
    #     ax.fill_between(time, np.percentile(component, 5, axis=0), np.percentile(component, 95, axis=0), color='navy', alpha=0.2)
    #     ax.set_title(component_names[i])
    #     ax.set_xlabel('Year')
    #     if i == 0:
    #         ax.set_ylabel('Sea level change (m)')
    #         ax.legend(loc='upper left', fancybox=True)
        
    # plt.tight_layout()
    # plt.savefig('ssp126_components.png', dpi=300)


def project_glacier(it, zit, glacier, glaciermip):
    # Return projection of glacier contribution as a cf.Field
    # it -- cf.Field, time-integral of median temperature anomaly timeseries
    # zit -- cf.Field, ensemble of time-integral temperature anomaly timeseries
    # template -- cf.Field with the required shape of the output
    # glaciermip -- False => AR5 parameters, 1 => fit to Hock et al. (2019),
    #   2 => fit to Marzeion et al. (2020)

    startyr=endofhistory

    dmzdtref=0.95 # mm yr-1 in Marzeion's CMIP5 ensemble mean for AR5 ref period
    dmz=dmzdtref*(startyr-1996)*1e-3 # m from glacier at start wrt AR5 ref period
    glmass=412.0-96.3 # initial glacier mass, used to set a limit, from Tab 4.2
    glmass=1e-3*glmass # m SLE

    nr=glacier.shape[0]
    if glaciermip:
        if glaciermip==1:
            glparm=[dict(name='SLA2012',factor=3.39,exponent=0.722,cvgl=0.15),\
                dict(name='MAR2012',factor=4.35,exponent=0.658,cvgl=0.13),\
                dict(name='GIE2013',factor=3.57,exponent=0.665,cvgl=0.13),\
                dict(name='RAD2014',factor=6.21,exponent=0.648,cvgl=0.17),\
                dict(name='GloGEM',factor=2.88,exponent=0.753,cvgl=0.13)]
            cvgl=0.15 # unnecessary default
        elif glaciermip==2:
            glparm=[dict(name='GLIMB',factor=3.70,exponent=0.662,cvgl=0.206),\
                dict(name='GloGEM',factor=4.08,exponent=0.716,cvgl=0.161),\
                dict(name='JULES',factor=5.50,exponent=0.564,cvgl=0.188),\
                dict(name='Mar-12',factor=4.89,exponent=0.651,cvgl=0.141),\
                dict(name='OGGM',factor=4.26,exponent=0.715,cvgl=0.164),\
                dict(name='RAD2014',factor=5.18,exponent=0.709,cvgl=0.135),\
                dict(name='WAL2001',factor=2.66,exponent=0.730,cvgl=0.206)]
            cvgl=0.20 # unnecessary default
        else: raise KeyError('glaciermip must be 1 or 2')
    else:
        glparm=[dict(name='Marzeion',factor=4.96,exponent=0.685),\
        dict(name='Radic',factor=5.45,exponent=0.676),\
        dict(name='Slangen',factor=3.44,exponent=0.742),\
        dict(name='Giesen',factor=3.02,exponent=0.733)]
        cvgl=0.20 # random methodological error
    ngl=len(glparm) # number of glacier methods
    if nr%ngl:
        raise ValueError('number of realisations '+\
        'must be a multiple of number of glacier methods')
    nrpergl=int(nr/ngl) # number of realisations per glacier method
    r = np.random.standard_normal(nr)
    r = r[:, np.newaxis, np.newaxis]

    # Make an ensemble of projections for each method
    for igl in range(ngl):
        # glacier projection for this method using the median temperature timeseries
        mgl = project_glacier1(it, glparm[igl]['factor'], glparm[igl]['exponent'])
        
        # glacier projections for this method with the ensemble of timeseries
        zgl = project_glacier1(zit, glparm[igl]['factor'], glparm[igl]['exponent'])

        ifirst = igl * nrpergl
        ilast = ifirst + nrpergl
        if glaciermip: cvgl = glparm[igl]['cvgl'] 
        glacier[ifirst:ilast,...] = zgl + (mgl * r[ifirst:ilast] * cvgl)

    glacier += dmz
    glacier = np.where(glacier > glmass, glmass, glacier)
    
    glacier = glacier.reshape(glacier.shape[0]*glacier.shape[1], glacier.shape[2])

    return glacier


def project_glacier1(it,factor,exponent):
    # Return projection of glacier contribution by one glacier method
    scale=1e-3 # mm to m
    return scale * factor * (np.where(it<0, 0, it)**exponent)


def project_greensmb(zt,template,palmer=False):
    # Return projection of Greenland SMB contribution as a cf.Field
    # zt -- cf.Field, ensemble of temperature anomaly timeseries
    # template -- cf.Field with the required shape of the output

    dtgreen=-0.146 # Delta_T of Greenland ref period wrt AR5 ref period  
    fnlogsd=0.4 # random methodological error of the log factor
    febound=[1,1.15] # bounds of uniform pdf of SMB elevation feedback factor

    nr = template.shape[0]
    # random log-normal factor
    fn = np.exp(np.random.standard_normal(nr)*fnlogsd)
    # elevation feedback factor
    fe = np.random.sample(nr)*(febound[1]-febound[0])+febound[0]
    ff = fn * fe
    
    ztgreen = zt - dtgreen
    greensmbrate = ff[:, np.newaxis, np.newaxis] * fettweis(ztgreen)

    if palmer and endyr > endofAR5:
        greensmbrate[:, :, 95:] = greensmbrate[:, :, 94:95]

    greensmb = np.cumsum(greensmbrate, axis=-1)

    np.add(greensmb, (1 - fgreendyn) * dgreen, out=greensmb)

    greensmb = greensmb.reshape(greensmb.shape[0]*greensmb.shape[1], greensmb.shape[2])

    return greensmb


def fettweis(ztgreen):
    # Greenland SMB in m yr-1 SLE from global mean temperature anomaly
    # using Eq 2 of Fettweis et al. (2013)
    return (71.5*ztgreen+20.4*(ztgreen**2)+2.8*(ztgreen**3))*mSLEoGt


def project_antsmb(zit,template,fraction=None):
    # Return projection of Antarctic SMB contribution as a cf.Field
    # zit -- cf.Field, ensemble of time-integral temperature anomaly timeseries
    # template -- cf.Field with the required shape of the output
    # fraction -- array-like, random numbers for the SMB-dynamic feedback

    nr=template.shape[0]
    nt=template.shape[1]
    # antsmb=template.copy()
    # nr,nt,nyr=antsmb.shape

    # The following are [mean,SD]
    pcoK=[5.1,1.5] # % change in Ant SMB per K of warming from G&H06
    KoKg=[1.1,0.2] # ratio of Antarctic warming to global warming from G&H06

    # Generate a distribution of products of the above two factors
    pcoKg=(pcoK[0]+np.random.standard_normal([nr,nt])*pcoK[1])*\
        (KoKg[0]+np.random.standard_normal([nr,nt])*KoKg[1])
    meansmb=1923 # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
    moaoKg=-pcoKg*1e-2*meansmb*mSLEoGt # m yr-1 of SLE per K of global warming

    if fraction is None:
        fraction=np.random.rand(nr,nt)
    elif fraction.size!=nr*nt:
        raise ValueError('fraction is the wrong size')
    else:
        fraction.shape=(nr,nt)

    smax=0.35 # max value of S in 13.SM.1.5
    ainterfactor=1-fraction*smax

    z = moaoKg * ainterfactor
    z = z[:, :, np.newaxis]
    antsmb = z * zit
    antsmb = antsmb.reshape(antsmb.shape[0]*antsmb.shape[1], antsmb.shape[2])

    return antsmb


def project_greendyn(scenario,template,palmer=False):
    # Return projection of Greenland rapid ice-sheet dynamics contribution
    # as a cf.Field
    # scenario -- str, name of scenario
    # template -- cf.Field with the required shape of the output

    # For SMB+dyn during 2005-2010 Table 4.6 gives 0.63+-0.17 mm yr-1 (5-95% range)
    # For dyn at 2100 Chapter 13 gives [20,85] mm for rcp85, [14,63] mm otherwise

    if scenario in ['rcp85','ssp585']:
        finalrange=[0.020,0.085]
    else:
        finalrange=[0.014,0.063]
    return time_projection(0.63*fgreendyn,\
        0.17*fgreendyn,finalrange,template,palmer=palmer)+fgreendyn*dgreen
  

def project_antdyn(template,fraction=None,output=None,
    palmer=False):
    # Return projection of Antarctic rapid ice-sheet dynamics contribution
    # as a cf.Field
    # template -- cf.Field with the required shape of the output
    # fraction -- array-like, random numbers for the dynamic contribution
    # levermann -- optional, str, use Levermann fit for specified scenario

    final=[-0.020,0.185]

    # For SMB+dyn during 2005-2010 Table 4.6 gives 0.41+-0.24 mm yr-1 (5-95% range)
    # For dyn at 2100 Chapter 13 gives [-20,185] mm for all scenarios

    return time_projection(0.41,0.20,final,template,
        palmer=palmer,fraction=fraction)+dant
  
  
def project_landwater(template,palmer=False):
    # Return projection of land water storage contribution as a cf.Field

    # The rate at start is the one for 1993-2010 from the budget table.
    # The final amount is the mean for 2081-2100.
    nyr=2100-2081+1 # number of years of the time-mean of the final amount

    return time_projection(0.38,0.49-0.38,[-0.01,0.09],template,
        nfinal=nyr,palmer=palmer)
    
    
def time_projection(startratemean,startratepm,final,template,
    nfinal=1,fraction=None,palmer=False):
    # Return projection of a quantity which is a quadratic function of time
    # in a cf.Field.
    # startratemean, startratepm -- rate of GMSLR at the start in mm yr-1, whose
    #   likely range is startratemean +- startratepm
    # final -- two-element list giving likely range in m for GMSLR at the endofAR5,
    #   or array-like, giving final values at that time, of the same shape as
    #   fraction and assumed corresponding elements
    # template -- cf.Field with the required shape of the output
    # nfinal -- int, optional, number of years at the end over which finalrange is
    #   a time-mean; by default 1 => finalrange is the value for the last year
    # fraction -- array-like, optional, random numbers in the range 0 to 1,
    #   by default uniformly distributed

    # Create a field of elapsed time since start in years
    timeendofAR5 = endofAR5 - endofhistory + 1
    time = np.arange(endyr - endofhistory) + 1

    # more general than nr,nt,nyr=template.shape
    nr, nt, nyr = template.shape
    
    if fraction is None:
        fraction=np.random.rand(nr,nt)
    elif fraction.size!=nr*nt:
        raise ValueError('fraction is the wrong size')
    fraction = fraction.reshape(nr,nt)

    # Convert inputs to startrate (m yr-1) and afinal (m), where both are
    # arrays with the size of fraction
    momm=1e-3 # convert mm yr-1 to m yr-1
    startrate=(startratemean+\
        startratepm*np.array([-1,1],dtype=float))*momm
    finalisrange=isinstance(final,Sequence)
    if finalisrange:
        if len(final)!=2:
            raise ValueError('final range is the wrong size')
        afinal=(1-fraction)*final[0]+fraction*final[1]
    else:
        if final.shape!=fraction.shape:
            raise ValueError('final array is the wrong shape')
        afinal = final
    startrate=(1-fraction)*startrate[0]+fraction*startrate[1]

    # For terms where the rate increases linearly in time t, we can write GMSLR as
    #   S(t) = a*t**2 + b*t
    # where a is 0.5*acceleration and b is start rate. Hence
    #   a = S/t**2-b/t = (S-b*t)/t**2
    # If nfinal=1, the following two lines are equivalent to
    # halfacc=(final-startyr*nyr)/nyr**2
    finalyr = np.arange(nfinal) - nfinal + 94 + 1  # last element ==nyr
    halfacc=(afinal-startrate*finalyr.mean())/(finalyr**2).mean()
    quadratic=halfacc[:, :, np.newaxis]*(time**2)
    linear=startrate[:, :, np.newaxis]*time

    # If acceleration ceases for t>t0, the rate is 2*a*t0+b thereafter, so
    #   S(t) = a*t0**2 + b*t0 + (2*a*t0+b)*(t-t0)
    #        = a*t0*(2*t - t0) + b*t
    # i.e. the quadratic term is replaced, the linear term unaffected
    # The quadratic also = a*t**2-a*(t-t0)**2

    if palmer:
        y = halfacc[:, :, np.newaxis] * timeendofAR5 * ((2 * time) - timeendofAR5)
        quadratic[:, :, 95:] = y[:, :, 95:]

    np.add(quadratic, linear, out=quadratic)
    
    quadratic = quadratic.reshape(quadratic.shape[0]*quadratic.shape[1], quadratic.shape[2])
    
    return quadratic

if __name__ == '__main__':
    project(['ssp126'], palmer=True)
