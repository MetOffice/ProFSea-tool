"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

# Calculate Monte Carlo projections of GMSLR using methods 
# from Jonathon Gregory and AR5. Staying close to JG's original
# code where possible.
import os
import concurrent
from collections.abc import Sequence

import numpy as np

class GMSLREmulator:
    
    def __init__(
        self, 
        T_change: np.ndarray,
        OHC_change: np.ndarray,
        scenario: str,
        output_dir: str,
        end_yr: int,
        seed: int=0,
        nt: int=450,
        nm: int=1000,
        tcv: float=1.0,
        glaciermip: bool=False,
        input_ensemble: bool=True,
        palmer_method: bool=False):
        # ensemble -- bool, optional, default True, if provided use an input ensemble of 
        #   temperature and ocean heat content change in place of Monte Carlo simulations
        #   for thermosteric sea level rise 
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
        # palmer_method -- bool, optional, default False, allow integration to end in any year
        #   up to 2300, with the contributions to GMLSR from ice-sheet dynamics, Green-
        #   land SMB and land water storage held at the 2100 rate beyond 2100.
        
        self.T_change = T_change
        self.OHC_change = OHC_change
        self.scenario = scenario
        self.output_dir = output_dir
        self.end_yr = end_yr
        self.seed = seed
        self.nt = nt
        self.nm = nm
        self.tcv = tcv
        self.glaciermip = glaciermip
        self.input_ensemble = input_ensemble
        self.palmer_method = palmer_method
        
        # First year of AR5 projections
        self.endofhistory = 2006

        # Last year of AR5 projections
        self.endofAR5 = 2100
        
        # Length of projections
        self.nyr = self.end_yr - self.endofhistory

        # Fraction of SLE from Greenland during 1996 to 2005 assumed to result from
        # rapid dynamical change, with the remainder assumed to result from SMB change
        self.fgreendyn = 0.5

        # m SLE from Greenland during 1996 to 2005 according to AR5 chapter 4
        self.dgreen = (3.21 - 0.30) * 1e-3

        # m SLE from Antarctica during 1996 to 2005 according to AR5 chapter 4
        self.dant = (2.37 + 0.13) * 1e-3

        # Conversion factor for Gt to m SLE
        self.mSLEoGt = 1e12 / 3.61e14 * 1e-3

        # Sensitivity of thermosteric SLR to ocean heat content change
        self.exp_efficiency = 0.12e-24
        
        if input_ensemble:
            self.nt = self.T_change.shape[0]
            
    def get_components(self):
        components_dict = {
            'exp': self.expansion,
            'glacier': self.glacier,
            'greensmb': self.greensmb,
            'greendyn': self.greendyn,
            'greennet': self.greennet,
            'antsmb': self.antsmb,
            'antdyn': self.antdyn,
            'antnet': self.antnet,
            'landwater': self.landwater,
            'sheetdyn': self.sheetdyn,
            'gmslr': self.gmslr
        }
        return components_dict
    
    def save_components(self, output_dir: str, scenario_name: str):
        """Save all SLR components as .npy files to a directory.
        
        Parameters
        ----------
        output_directory: str
            Directory to save components to.
        scenario_name: str
            Name of the scenario you've run the emulator for. 
            
        Returns
        -------
        None
        """
        for name, component in self.get_components().items():
            np.save(
                os.path.join(
                    output_dir, 
                    f'{scenario_name}_{name}.npy'), 
                component.T # Transpose to match the shape of original Monte Carlo simulations
            )
        
    def project(self):
        """Run the emulator to project GMSLR components.

        Returns
        -------
        None
        """
        np.random.seed(self.seed)
        T_ens, Exp_ens, T_int_ens, T_int_med = self.calculate_drivers() 

        self.expansion = np.tile(Exp_ens, (self.nm, 1))
        fraction = np.random.rand(self.nm * self.nt) # correlation between antsmb and antdyn
        
        self.run_parallel_projections(T_int_med, T_int_ens, T_ens, fraction)
        
        self.greennet = self.greensmb + self.greendyn
        self.antnet = self.antsmb + self.antdyn
        self.sheetdyn = self.greendyn + self.antdyn
        self.gmslr = self.glacier + self.greennet + self.antnet + self.landwater + self.expansion
            
    def run_parallel_projections(self, T_int_med: np.ndarray, T_int_ens: np.ndarray, 
                                 T_ens: np.ndarray, fraction: np.ndarray):
        """Run components of the emulator in parallel.
        
        Returns
        -------
        None
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.project_glacier, T_int_med, T_int_ens): 'glacier',
                executor.submit(self.project_greensmb, T_ens): 'greensmb',
                executor.submit(self.project_antsmb, T_int_ens, fraction): 'antsmb',
                executor.submit(self.project_greendyn): 'greendyn',
                executor.submit(self.project_antdyn, fraction): 'antdyn',
                executor.submit(self.project_landwater): 'landwater'
            }
            results = {}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"{key} generated an exception: {e}")

        self.glacier = results['glacier']
        self.greensmb = results['greensmb']
        self.antsmb = results['antsmb']
        self.greendyn = results['greendyn']
        self.antdyn = results['antdyn']
        self.landwater = results['landwater']
        
    def calculate_drivers(self):
        """Calculate the drivers of GMSLR: temperature change and 
        thermosteric sea level rise.
        
        Returns
        -------
        T_ens: np.ndarray
            Ensemble of temperature changes.
        therm_ens: np.ndarray
            Ensemble of thermosteric sea level rise.
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomalies.
        T_int_med: np.ndarray
            Median of time-integral temperature anomalies.
        """
        if self.input_ensemble:
            # Check if dimensions are the right way around 
            if self.T_change.shape[1] != self.nyr: 
                self.T_change = self.T_change.T
            if self.OHC_change.shape[1] != self.nyr: 
                self.OHC_change = self.OHC_change.T
                
            therm_ens = self.OHC_change * self.exp_efficiency
            T_int_ens = np.cumsum(self.T_change, axis=1)
            T_int_med = np.percentile(T_int_ens, 50, axis=0) # using median here instead of mean... CHECK THIS

            return self.T_change, therm_ens, T_int_ens, T_int_med
        
        T_med = np.percentile(self.T_change, 50, axis=0)
        T_std = np.std(self.T_change, axis=0)

        therm_med = np.percentile(self.OHC_change, 50, axis=0) * self.exp_efficiency
        therm_std = np.std(self.OHC_change * self.exp_efficiency, axis=0)
        
        # Time-integral of temperature anomaly
        T_int_med = np.cumsum(T_med)
        T_int_std = np.cumsum(T_std)
        
        # Generate a sample of perfectly correlated timeseries fields of temperature,
        # time-integral temperature and expansion, each of them [realisation,time]
        z = np.random.standard_normal(self.nt) * self.tcv

        # For each quantity, mean + standard deviation * normal random number
        # reshape to [realisation,time]
        T_ens = z[:, np.newaxis] * T_std + T_med
        therm_ens = z[:, np.newaxis] * therm_std + therm_med
        T_int_ens = z[:, np.newaxis] * T_int_std + T_int_med
        
        return T_ens, therm_ens, T_int_ens, T_int_med
        

    def project_glacier(self, T_int_med: np.ndarray, T_int_ens: np.ndarray):
        """Project glacier contribution to GMSLR.
        
        Parameters
        ----------
        T_int_med: np.ndarray
            Time-integral of median temperature anomaly timeseries.
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomaly timeseries.
        
        Returns
        -------
        glacier: np.ndarray
            Glacier contribution to GMSLR.
        """
        # glaciermip -- False => AR5 parameters, 1 => fit to Hock et al. (2019),
        #   2 => fit to Marzeion et al. (2020)
        dmzdtref=0.95 # mm yr-1 in Marzeion's CMIP5 ensemble mean for AR5 ref period
        dmz=dmzdtref*(self.endofhistory-1996)*1e-3 # m from glacier at start wrt AR5 ref period
        glmass=412.0-96.3 # initial glacier mass, used to set a limit, from Tab 4.2
        glmass=1e-3*glmass # m SLE
        
        glacier = np.full((self.nm, self.nt, self.nyr), np.nan)

        if self.glaciermip:
            if self.glaciermip==1:
                glparm=[dict(name='SLA2012',factor=3.39,exponent=0.722,cvgl=0.15),\
                    dict(name='MAR2012',factor=4.35,exponent=0.658,cvgl=0.13),\
                    dict(name='GIE2013',factor=3.57,exponent=0.665,cvgl=0.13),\
                    dict(name='RAD2014',factor=6.21,exponent=0.648,cvgl=0.17),\
                    dict(name='GloGEM',factor=2.88,exponent=0.753,cvgl=0.13)]
                cvgl=0.15 # unnecessary default
            elif self.glaciermip==2:
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
        if self.nm%ngl:
            raise ValueError('number of realisations '+\
            'must be a multiple of number of glacier methods')
            
        nrpergl = self.nm // ngl
        r = np.random.standard_normal(self.nm)
        r = r[:, np.newaxis, np.newaxis]
        
        # Precompute mgl and zgl for all glacier methods
        mgl_all = np.array([self._project_glacier1(T_int_med, glparm[igl]['factor'], glparm[igl]['exponent']) for igl in range(ngl)])
        zgl_all = np.array([self._project_glacier1(T_int_ens, glparm[igl]['factor'], glparm[igl]['exponent']) for igl in range(ngl)])
        cvgl_all = np.array([glparm[igl]['cvgl'] if self.glaciermip else cvgl for igl in range(ngl)])

        # Make an ensemble of projections for each method
        for igl in range(ngl):
            mgl = mgl_all[igl]
            zgl = zgl_all[igl]
            cvgl = cvgl_all[igl]
            
            ifirst = igl * nrpergl
            ilast = ifirst + nrpergl

            glacier[ifirst:ilast, ...] = zgl + (mgl * r[ifirst:ilast] * cvgl)

        glacier += dmz
        np.clip(glacier, None, glmass, out=glacier)
        
        glacier = glacier.reshape(glacier.shape[0]*glacier.shape[1], glacier.shape[2])

        return glacier

    def _project_glacier1(self, T_int: np.ndarray, factor: float, exponent: float):
        """Project glacier contribution by one glacier method.
        
        Parameters
        ----------
        T_int: np.ndarray
            Time-integral temperature anomaly timeseries.
        factor: float
            Factor for the glacier method.
        exponent: float
            Exponent for the glacier method.
            
        Returns
        -------
        np.ndarray
            Projection of glacier contribution.
        """
        scale=1e-3 # mm to m
        return scale * factor * (np.where(T_int<0, 0, T_int)**exponent)

    def project_greensmb(self, T_ens: np.ndarray):
        """Project Greenland SMB contribution to GMSLR.
        
        Parameters
        ----------
        T_ens: np.ndarray
            Ensemble of temperature anomaly timeseries.
            
        Returns
        -------
        greensmb: np.ndarray
            Greenland SMB contribution to GMSLR.
        
        """
        dtgreen = -0.146 # Delta_T of Greenland ref period wrt AR5 ref period  
        fnlogsd = 0.4 # random methodological error of the log factor
        febound = [1, 1.15] # bounds of uniform pdf of SMB elevation feedback factor

        # random log-normal factor
        fn = np.exp(np.random.standard_normal(self.nm)*fnlogsd)
        # elevation feedback factor
        fe = np.random.sample(self.nm) * (febound[1] - febound[0]) + febound[0]
        ff = fn * fe
        
        ztgreen = T_ens - dtgreen
        
        greensmb = ff[:, np.newaxis, np.newaxis] * self._fettweis(ztgreen)

        if self.palmer_method and self.end_yr > self.endofAR5:
            greensmb[:, :, 95:] = greensmb[:, :, 94:95]

        greensmb = np.cumsum(greensmb, axis=-1)

        greensmb += (1 - self.fgreendyn) * self.dgreen

        greensmb = greensmb.reshape(greensmb.shape[0]*greensmb.shape[1], greensmb.shape[2])

        return greensmb

    def _fettweis(self, ztgreen):
        """Calculate Greenland SMB in m yr-1 SLE from global mean temperature 
        anomaly, using Eq 2 of Fettweis et al. (2013).
        
        Parameters
        ----------
        ztgreen: np.ndarray
            Global mean temperature anomaly.
            
        Returns
        -------
        np.ndarray
            Greenland SMB in m yr-1 SLE.
        """
        return (71.5*ztgreen+20.4*(ztgreen**2)+2.8*(ztgreen**3))*self.mSLEoGt

    def project_antsmb(self, T_int_ens, fraction=None):
        """Project Antarctic SMB contribution to GMSLR.
        
        Parameters
        ----------
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomaly timeseries.
        fraction: np.ndarray
            Random numbers for the SMB-dynamic feedback.
            
        Returns
        -------
        antsmb: np.ndarray
            Antarctic SMB contribution to GMSLR.
        """
        # The following are [mean,SD]
        pcoK=[5.1,1.5] # % change in Ant SMB per K of warming from G&H06
        KoKg=[1.1,0.2] # ratio of Antarctic warming to global warming from G&H06

        # Generate a distribution of products of the above two factors
        pcoKg = (pcoK[0] + np.random.standard_normal([self.nm, self.nt]) * pcoK[1]) * \
            (KoKg[0] + np.random.standard_normal([self.nm, self.nt]) * KoKg[1])
        meansmb = 1923 # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
        moaoKg = -pcoKg * 1e-2 * meansmb * self.mSLEoGt # m yr-1 of SLE per K of global warming

        if fraction is None:
            fraction = np.random.rand(self.nm, self.nt)
        elif fraction.size != self.nm * self.nt:
            raise ValueError('fraction is the wrong size')
        else:
            fraction.shape = (self.nm, self.nt)

        smax = 0.35 # max value of S in 13.SM.1.5
        ainterfactor = 1 - fraction * smax

        z = moaoKg * ainterfactor
        z = z[:, :, np.newaxis]
        antsmb = z * T_int_ens
        antsmb = antsmb.reshape(antsmb.shape[0] * antsmb.shape[1], antsmb.shape[2])

        return antsmb

    def project_greendyn(self):
        """Project Greenland rapid ice-sheet dynamics contribution to GMSLR.
        
        Returns
        -------
        np.ndarray
            Greenland rapid ice-sheet dynamics contribution to GMSLR.
        """
        # For SMB+dyn during 2005-2010 Table 4.6 gives 0.63+-0.17 mm yr-1 (5-95% range)
        # For dyn at 2100 Chapter 13 gives [20,85] mm for rcp85, [14,63] mm otherwise
        if self.scenario in ['rcp85','ssp585']:
            finalrange=[0.020,0.085]
        else:
            finalrange=[0.014,0.063]
        return self.time_projection(
            0.63*self.fgreendyn, 0.17*self.fgreendyn, 
            finalrange) + self.fgreendyn*self.dgreen

    def project_antdyn(self, fraction=None):
        """Project Antarctic rapid ice-sheet dynamics contribution to GMSLR.
        
        Parameters
        ----------
        fraction: np.ndarray
            Random numbers for the dynamic contribution.
        
        Returns
        -------
        np.ndarray
            Antarctic rapid ice-sheet dynamics contribution to GMSLR.
        """
        # lcoeff=dict(rcp26=[-2.881, 0.923, 0.000],\
        # rcp45=[-2.676, 0.850, 0.000],\
        # rcp60=[-2.660, 0.870, 0.000],\
        # rcp85=[-2.399, 0.860, 0.000])
        # lcoeff = lcoeff['rcp85']

        # from scipy.stats import norm
        # ascale=norm.ppf(fraction)
        # final=np.exp(lcoeff[2]*ascale**2+lcoeff[1]*ascale+lcoeff[0])
        # final = final.reshape(self.nm, self.nt)
        
        # final=[-0.020, 0.185]
        final = [0.06, 0.49] # AR6, SSP2-4.5

        # For SMB+dyn during 2005-2010 Table 4.6 gives 0.41+-0.24 mm yr-1 (5-95% range)
        # For dyn at 2100 Chapter 13 gives [-20,185] mm for all scenarios

        return self.time_projection(0.41, 0.20, final, fraction=fraction) + self.dant
  
    def project_landwater(self):
        """Project land water storage contribution to GMSLR.
        
        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """
        # The rate at start is the one for 1993-2010 from the budget table.
        # The final amount is the mean for 2081-2100.
        nyr=2100-2081+1 # number of years of the time-mean of the final amount
        # final = [-0.01,0.09] # AR5
        final = [0.01, 0.04] # AR6
        return self.time_projection(0.38, 0.49-0.38, final, nfinal=nyr)
    
    def time_projection(
        self, startratemean: float, startratepm: float, final,
        nfinal: int=1, fraction: np.ndarray=None):
        """Project a quantity which is a quadratic function of time.
        
        Parameters
        ----------
        startratemean: float
            Rate of GMSLR at the start in mm yr-1.
        startratepm: float
            Start rate error in mm yr-1.
        final: list | np.ndarray
            Likely range in m for GMSLR at the end of AR5.
        nfinal: int
            Number of years at the end over which final is a time-mean.
        fraction: np.ndarray
            Random numbers in the range 0 to 1.
            
        Returns
        -------
        np.ndarray
            Projection of the quantity.
        """
        # Create a field of elapsed time since start in years
        timeendofAR5 = self.endofAR5 - self.endofhistory + 1
        time = np.arange(self.end_yr - self.endofhistory) + 1
        
        if fraction is None:
            fraction = np.random.rand(self.nm, self.nt)
        elif fraction.size != self.nm * self.nt:
            raise ValueError('fraction is the wrong size')
        
        fraction = fraction.reshape(self.nm, self.nt)

        # Convert inputs to startrate (m yr-1) and afinal (m), where both are
        # arrays with the size of fraction
        startrate = (startratemean + \
            startratepm * np.array([-1,1],dtype=float)) * 1e-3 # convert mm yr-1 to m yr-1
        finalisrange = isinstance(final, Sequence)
        
        if finalisrange:
            if len(final) != 2:
                raise ValueError('final range is the wrong size')
            afinal = (1 - fraction) * final[0] + fraction * final[1]
        else:
            if final.shape != fraction.shape:
                raise ValueError('final array is the wrong shape')
            afinal = final
            
        startrate = (1 - fraction) * startrate[0] + fraction * startrate[1]

        # For terms where the rate increases linearly in time t, we can write GMSLR as
        #   S(t) = a*t**2 + b*t
        # where a is 0.5*acceleration and b is start rate. Hence
        #   a = S/t**2-b/t = (S-b*t)/t**2
        # If nfinal=1, the following two lines are equivalent to
        # halfacc=(final-startyr*nyr)/nyr**2
        finalyr = np.arange(nfinal) - nfinal + 94 + 1  # last element ==nyr
        halfacc = (afinal - startrate * finalyr.mean()) / (finalyr**2).mean()
        quadratic = halfacc[:, :, np.newaxis] * (time**2)
        linear = startrate[:, :, np.newaxis] * time

        # If acceleration ceases for t>t0, the rate is 2*a*t0+b thereafter, so
        #   S(t) = a*t0**2 + b*t0 + (2*a*t0+b)*(t-t0)
        #        = a*t0*(2*t - t0) + b*t
        # i.e. the quadratic term is replaced, the linear term unaffected
        # The quadratic also = a*t**2-a*(t-t0)**2

        if self.palmer_method:
            y = halfacc[:, :, np.newaxis] * timeendofAR5 * ((2 * time) - timeendofAR5)
            quadratic[:, :, 95:] = y[:, :, 95:]

        quadratic += linear
        
        quadratic = quadratic.reshape(quadratic.shape[0]*quadratic.shape[1], quadratic.shape[2])
        
        return quadratic
