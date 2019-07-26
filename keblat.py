# -*- coding: utf-8 -*-

from glob import glob
import numpy as np
#from scipy import interpolate 
#from ext_func.occultquad import occultquad
#from ext_func.rsky import rsky
import matplotlib.pyplot as plt
import os.path
import time
from collections import OrderedDict
try:
    from helper_funcs import poly_lc_cwrapper, rsky, occultquad
except:
    print('Exception in from helper_funcs import poly_lc_cwrapper, rsky, occultquad')
    import datetime, os
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    print('Current dir: {0}'.format(os.getcwd()))
    print('helper_funcs.py exists? {0}'.format(os.path.isfile('helper_funcs.py')))
    print('helpers_linux.so exists? {0}'.format(os.path.isfile('helpers_linux.so')))
    # print(os.uname())
    try:
        from helper_funcs import poly_lc_cwrapper, rsky, occultquad
        print("worked here")
        # print(os.uname())
    except:
        time.sleep(30)
        # print(os.uname())
        print("just slept for 30")
        from helper_funcs import poly_lc_cwrapper, rsky, occultquad
        print("worked after sleep")

#from nbody import occultquad, rsky
from scipy.optimize import minimize as sp_minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

#############################################
################ constants ##################
#############################################
TWOPI = 2.0*np.pi
d2y = 365.242 #days in a year
r2au = 0.0046491 #solar radii to AU
parnames_dict = {'lc': ['msum', 'rsum', 'rrat', 'period', 'tpe', 'esinw', 'ecosw', 'b', 'frat', 'q1',
                        'q2', 'q3', 'q4'],
                 'sed': ['msum', 'mrat', 'z0', 'age', 'dist', 'ebv', 'h0', 'isoerr'],
                 'sed2': ['msum', 'mrat', 'z0', 'age', 'age2', 'dist', 'ebv', 'h0', 'isoerr'],
                 'rv': ['msum', 'mrat', 'period', 'tpe', 'esinw', 'ecosw', 'inc', 'k0', 'rverr'],
                 'lcsed': ['msum', 'mrat', 'z0', 'age', 'dist', 'ebv', 'h0', 'period', 'tpe','esinw',
                           'ecosw', 'b', 'q1', 'q2', 'q3', 'q4', 'lcerr', 'isoerr'],
                 'lcsed_sigmas': ['msum', 'mrat', 'z0', 'age', 'dist', 'ebv', 'h0', 'period', 'tpe','esinw',
                           'ecosw', 'b', 'q1', 'q2', 'q3', 'q4', 'lcerr', 'isoerr', 'sigma_ebv', 'sigma_d'],
                 'lcsed2': ['msum', 'mrat', 'z0', 'age', 'age2', 'dist', 'ebv', 'h0', 'period', 'tpe','esinw',
                           'ecosw', 'b', 'q1', 'q2', 'q3', 'q4', 'lcerr', 'isoerr'],
                 'lcsed_sigmas2': ['msum', 'mrat', 'z0', 'age', 'age2', 'dist', 'ebv', 'h0', 'period', 'tpe','esinw',
                           'ecosw', 'b', 'q1', 'q2', 'q3', 'q4', 'lcerr', 'isoerr', 'sigma_ebv', 'sigma_d'],

                 'lcrv': ['msum', 'mrat', 'rsum', 'rrat', 'period', 'tpe', 'esinw', 'ecosw', 'b', 'frat',
                          'q1', 'q2', 'q3', 'q4', 'lcerr', 'k0', 'rverr']}
#############################################

class Keblat(object):
    """An EB fitting routine
    
    """
    def __init__(self, vkeb=False, preload=True):
        self.iso = None
        self.sed = None
        self.kic = None
        self.jd = None
        self.flux = None
        self.dflux = None
        self.vkeb = None
        self.fluxerr = None
        self.cadence = None
        self.rv1_obs = None
        self.rv2_obs = None
        self.rv1_err_obs = None
        self.rv2_err_obs = None
        self.rv_t = None
        self.exp = None
        self.clip_tol = 1.5
        self.res = []
        self.ldtype = 1 # quadratic limb darkening
        self.parnames = ['m1', 'm2', 'z0', 'age', 'dist', 'ebv', 'h0', 'period', 'tpe',
                         'esinw', 'ecosw', 'b', 'q1', 'q2', 'q3', 'q4', 'lcerr', 'isoerr',
                         'k0', 'rverr', 'msum', 'mrat', 'rsum', 'rrat', 'r1', 'r2', 'inc', 
                         'frat', 'e', 'omega', 'sigma_ebv', 'sigma_d', 'age2']

        self.pars = OrderedDict(zip(self.parnames, [None]*len(self.parnames)))

        self.parbounds = OrderedDict([('m1', [.1, 12.]), ('m2', [.1, 12.]), 
                          ('z0', [0.001, 0.06]), ('age', [6., 10.1]),
                          ('dist', [10., 15000.]), ('ebv', [0., 5.]),
                          ('h0', [119-1., 119+1.]), ('period', [0.05, 2000.]),
                          ('tpe', [0., 1e8]), ('esinw', [-.99, .99]),
                          ('ecosw', [-.99, .99]), ('b', [-10, 10.]), ('q1', [0., 1.]),
                          ('q2', [0., 1.]), ('q3', [0., 1.]), ('q4', [0., 1.]), 
                          ('lcerr', [-25, -4]), ('isoerr', [-25, 0]),
                                      ('k0', [-1e8, 1e8]), ('rverr', [-8, 12]),
                                      ('msum', [0.2, 24.]), ('mrat', [0.0085, 2.0]),
                                      ('rsum', [0.1, 1e6]), ('rrat', [1e-6, 1e3]),
                                      ('r1', [0.01, 1e6]), ('r2', [0.01, 1e6]),
                                      ('inc', [0., np.pi/2.]), ('frat', [1e-8, 1e2]),
                                      ('e', [0., 0.9]),
                                      ('sigma_d', [-7, 9]), ('sigma_ebv', [-12, 2]),
                                      ('age2', [6., 10.1])])

        if preload:
            self.loadiso2()
            self.loadsed()
            self.loadvkeb()
        if vkeb:
            self.loadvkeb()
        # solar values in cgs units
        self.tsun = 5778.
        self.lsun = 3.846e33
        self.rsun = 6.9598e10
        self.zsun = 0.01524
        self.msun = 1.989e33
        self.gsun = 6.6726e-8 * self.msun / (self.rsun**2)
        self.message='Initialized Properly'
        self.armstrongT1 = None
        self.armstrongdT1 = None
        self.armstrongT2 = None
        self.armstrongdT2 = None
        self.coeval = True

    def updatebounds(self, *args, **kwargs):
        """Forces boundaries of specified arg parameters s.t. they are constrainted to 2% of parameter value

        Parameters
        ----------
        args : str
                for possible parameter names, see keblat.pars

        Example
        -------
        keblat.updatebounds('period', 'tpe', 'esinw', 'ecosw')

        """
        partol = kwargs.pop('partol', 0.02)
        for i in args:
            self.parbounds[i] = [self.pars[i]-abs(self.pars[i])*partol, self.pars[i]+abs(self.pars[i])*partol]
        if self.rv1_obs is not None and self.rv2_obs is not None:
            self.parbounds['k0'] = [min(np.nanmin(self.rv1_obs), np.nanmin(self.rv2_obs)),
                                     max(np.nanmax(self.rv1_obs), np.nanmax(self.rv2_obs))]
        return

    def updatepars(self, **kwargs):
        """Update free parameters to present values"""
        assert type(kwargs) is dict, "Must be dict"
        for key, val in kwargs.iteritems():
            if key in self.pars:
                self.pars[key] = val
            else:
                print("""{} is not a free parameter, sorry.""".format(key))
        #print "Updated pars"
        return
        # for ii in range(len(kwargs.keys())):
        #     if kwargs.keys()[ii] in self.pars:
        #         self.pars[kwargs.keys()[ii]] = kwargs.values()[ii]
        #     else:
        #         print("""{} is not a free parameter, sorry.""".format(kwargs.keys()[ii]))
        # print "made changes to updatepars"
        # return
        # #self.updatephase(self.tpe, self.period, self.clip_tol)

    def getpars(self, partype='allpars'):
        """Returns the values of the parameters for lc only, iso only, and iso+lc fits

        Parameters
        ----------
        partype : str (optional)
            'lc', 'sed', 'lcsed', or 'allpars' (default).

        Returns
        -------
        x : array
            of length 14 (lc), 8 (sed), 18 (all)

        """
        if partype == 'allpars':
            return np.asarray(self.pars.values())
        elif partype == 'lc':
            lcpars = np.array([self.pars['msum'], self.pars['rsum'], self.pars['rrat'],
                    self.pars['period'], self.pars['tpe'], self.pars['esinw'], self.pars['ecosw'],
                    self.pars['b'], self.pars['frat'], self.pars['q1'], self.pars['q2'], self.pars['q3'],
                    self.pars['q4'], self.pars['lcerr']])
            return lcpars
        elif partype == 'sed':
            if self.coeval:
                isopars = np.array([self.pars['msum'], self.pars['mrat'], self.pars['z0'], self.pars['age'],
                                self.pars['dist'], self.pars['ebv'], self.pars['h0'], self.pars['isoerr']])
            else:
                isopars = np.array([self.pars['msum'], self.pars['mrat'], self.pars['z0'], self.pars['age'],
                                self.pars['age2'], self.pars['dist'], self.pars['ebv'], self.pars['h0'], self.pars['isoerr']])
            return isopars
        elif partype == 'rv':
            inc = self.get_inc(self.pars['b'], self.pars['r1'],
                               self.get_a(self.pars['period'], self.pars['msum']))
            rvpars = np.array([self.pars['msum'], self.pars['mrat'], self.pars['period'], self.pars['tpe'],
                               self.pars['esinw'], self.pars['ecosw'], inc, self.pars['k0'], self.pars['rverr']])
            return rvpars
        elif partype == 'lcsed':
            if self.coeval:
                lcsedpars = np.array([self.pars['msum'], self.pars['mrat'], self.pars['z0'], self.pars['age'],
                             self.pars['dist'], self.pars['ebv'], self.pars['h0'], self.pars['period'],
                             self.pars['tpe'], self.pars['esinw'], self.pars['ecosw'], self.pars['b'],
                             self.pars['q1'], self.pars['q2'], self.pars['q3'], self.pars['q4'],
                             self.pars['lcerr'], self.pars['isoerr']])
            else:
                lcsedpars = np.array([self.pars['msum'], self.pars['mrat'], self.pars['z0'], self.pars['age'],
                             self.pars['age2'],
                             self.pars['dist'], self.pars['ebv'], self.pars['h0'], self.pars['period'],
                             self.pars['tpe'], self.pars['esinw'], self.pars['ecosw'], self.pars['b'],
                             self.pars['q1'], self.pars['q2'], self.pars['q3'], self.pars['q4'],
                             self.pars['lcerr'], self.pars['isoerr']])
            return lcsedpars
        else:
            print "Not recognized parameter type, try again"
            return False

    def load_armstrong_temps(self, kic):
        """ Loads in temperature estimates from Armstrong et al and assigns them to the class

        Parameters
        ----------
        kic: int
            The KIC # of target EB

        Returns
        -------
        armstrong: array (4,)
            T1, dT1, T2, dT2 of EB components

        """
        armstrong = np.loadtxt('data/armstrong_keb_cat.csv', delimiter=',', usecols=(0,1,2,3,4))
        indx = self.kiclookup(kic, armstrong[:,0])
        self.armstrongT1 = armstrong[indx,1]
        self.armstrongdT1 = armstrong[indx,2]
        self.armstrongT2 = armstrong[indx,3]
        self.armstrongdT2 = armstrong[indx,4]
        return armstrong[indx, 1:5]

    def kiclookup(self, kic, target=None):
        """Function which returns the index of target ndarray that matches input KIC #
        
        Parameters
        ----------
        kic : int
            KIC # of target EB
        
        target : ndarray
            target array to search for matching input KIC #
            if None (default), search for match in SED matrix
            
        Returns
        -------
        sedidx : int or None
            index of array which corresponds to input KIC #
        
        Examples
        --------
        >>> keblat.kiclookup(9837578)
        38
        """
        if target is None:
            if self.sed is None:
                self.loadsed()
            target = self.sed[:, 0]
        sedidx = np.where(target.astype(int) == kic)[0]
        if len(sedidx)==0:
            print "No data matched to your input KIC number, double check it!"
            return None
        return sedidx[0]
                
    def loadiso2(self, isodir = 'data/', isoname = 'isodata_final.dat', ipnames=None):
        """Function which loads Padova isochrone data
        
        Parameters
        ----------
        isodir : str
            directory which stores isochrone file
        
        Returns
        -------
        True : boolean
            if isodata loading is successful, stored in keblat.iso
        
        Examples
        --------
        >>> keblat.loadiso()
        Isodata.cat already exists; loaded.
        True
        """
        
        self.isodict = {'z': 0, 'logt': 1, 'mini': 2, 'mact': 3, 'logl': 4, 
                        'logte': 5, 'logg': 6, 'mbol': 7, 'gmag': 8, 
                        'rmag': 9, 'imag': 10, 'zmag': 11, 'Umag': 12,  
                        'Bmag': 13, 'Vmag': 14, 'Jmag': 15, 'Hmag': 16, 
                        'Kmag': 17, 'w1': 18, 'w2': 19, 'mkep':20}
#        fmt = ['%.3f', '%.2f', '%.4f', '%.4f', '%.4f', '%.3f', '%.3f', '%.3f', 
#               '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', 
#               '%.3f', '%.3f', '%.3f', '%.3f',  '%.3f']
        isodatafile = isodir + isoname
        if os.path.isfile(isodatafile):
            iso = np.loadtxt(isodatafile, delimiter=',')
            self.iso = iso 
            print "Isodata.cat already exists; loaded."
        else:
            print "Isodata.cat does not exist."
            return False
        self.zvals = np.unique(self.iso[:, self.isodict['z']])
        self.tvals = np.unique(self.iso[:, self.isodict['logt']])
        self.maxz, self.minz = np.max(self.zvals), np.min(self.zvals)
        self.maxt, self.mint = np.max(self.tvals), np.min(self.tvals)

        if ipnames is None:
            self.ipname = np.array(['mact', 'mkep', 'logl', 'logte', 'logg'])
        else:
            self.ipname = ipnames
        self.ipinds = np.array([self.isodict[ii] for ii in self.ipname]).astype(int)

        return True
        
    def loadsed(self, sedfile='data/kepsedall_0216.dat'):
        """Loads SED file 
        
        Parameters
        ----------
        sedfile : str
            name of SED file
        
        Returns
        -------
        True : boolean
            if file loads successfully, info stored in keblat.sed array
        False : boolean
            if file doesn't exist
        
        References
        ----------
        SED information from various sources: 
        RA, Dec, Glon, Glat, Teff, logg, Fe/H from KIC
        UBV (mag, error) from The Kepler Field UBV Survey (Everett+ 2012)
        griz (mag, error) from SDSS DR7 or DR9
        JHK (mag, error) from 2MASS
        w1w2 (mag, error) from WISE
        E(B-V), std E(B-V) from Schlegel dust maps (IRSA IPAC DUST)
        
        Examples
        --------
        >>> keblat.loadsed()
        True
        """

        if os.path.isfile(sedfile):
            self.sed = np.loadtxt(sedfile, delimiter=';')
        else:
            print "SED/obs file does not exist. Try again."
            return False
#        self.redconvert = np.array([4.107, 3.641, 2.682, 3.303, 2.285, 1.698, 
#            1.263, 0.189, 0.146, 0.723, 0.460, 0.310])

# distance grid from PANSTARRS 3D dust map (in pc)
        self.ebv_dist = np.array([63.09573445, 79.43282347, 100., 125.89254118,
            158.48931925, 199.5262315, 251.18864315, 316.22776602,
            398.10717055, 501.18723363, 630.95734448, 794.32823472, 1000.,
           1258.92541179, 1584.89319246, 1995.26231497, 2511.88643151,
           3162.27766017, 3981.07170553, 5011.87233627, 6309.5734448,
           7943.28234724, 10000., 12589.25411794, 15848.93192461,
          19952.62314969, 25118.8643151, 31622.77660168, 39810.71705535,
          50118.72336273, 63095.73444802])
        return True
    
    def getmags(self, kic, w2=True):
        """Fetches SED & extinction data corresponding to input KIC #
        
        Parameters
        ----------
        kic : int
            KIC #

        Returns
        -------
        magobs : array
            observed magnitudes in SED data file
        emagsobs: array
            associated uncertainties in observed magnitudes
        extinction: tuple
            E(B-V) and associated error from Schlegel dust maps
        glat : float
            galactic latitude of KIC
            
        Examples
        --------
        >>> magsobs, emagsobs, extinction, glat = keblat.getmags(9837578)

        """
        sedidx = self.kiclookup(kic)
        if np.equal(sedidx,None):
            print "No data matched to your input KIC number, double check it!"
            return
        magsobs = self.sed[sedidx, 8:-2:2].ravel()
        emagsobs = self.sed[sedidx, 9:-2:2].ravel()
        extinction = (self.sed[sedidx, -2], self.sed[sedidx, -1])
        glat = self.sed[sedidx, 4]
        zkic = 10**self.sed[sedidx, 7] * self.zsun
        if np.isnan(zkic):
            zkic = self.zsun
        return magsobs, emagsobs, extinction, glat, zkic
        
    def isoprep(self, magsobs, emagsobs, extinction, glat, zkic, exclude=[]):
        """Preps isochrone fitting for a given set of SED observations 
        and extinction parameters and gets rid of bad data
        
        Parameters
        ----------
        magsobs : array
            observed magnitudes
        emagsobs : array
            uncertainties of magnitudes
        extinction : tuple
            E(B-V) and std E(B-V)
        glat : float
            galactic longitude of target
        w2 : boolean
            True if want to include wise 2 band data in interpolation       
            
        Returns
        -------
        True : boolean
            if prep successful
            
        Examples
        --------
        >>> magsobs, emagsobs, extinction, glat = keblat.getmags(9837578)
        >>> keblat.isoprep(magsobs, emagsobs, extinction, glat)
        True
        """
        self.glat = glat
        (self.ebv, self.debv) = extinction
        if np.isfinite(zkic):
            self.z0 = zkic
        else:
            self.z0 = self.zsun
        fullmagnames = np.array(['Umag','Bmag','Vmag','gmag','rmag','imag',
                                 'zmag','w1', 'w2', 'Jmag','Hmag','Kmag'])
        fullmaglams = np.array([3733.9, 4308.9, 5516.6, 4716.7, 6165.1, 7475.9,
                            8922.9, 33200., 45700., 12300., 16400., 21600.])
        redconvert = np.array([4.107, 3.641, 2.682, 3.303, 2.285, 1.698, 1.263,
                               0.189, 0.146, 0.723, 0.460, 0.310])

        # according to Everett Howell 2012, typical limits for U, B, V
        self.exclude = exclude
        if magsobs[0] > 18.7:
            self.exclude += ['Umag']
        if magsobs[1] > 19.3:
            self.exclude += ['Bmag']
        if magsobs[2] > 19.1:
            self.exclude += ['Vmag']
        nodata = (magsobs == -99.999)

        # if np.any(magsobs[3:7] > abs(magsobs[0])+0.2):
        #     exclude = exclude + list(np.array(['gmag', 'rmag', 'imag', 'zmag'])[(magsobs[3:7] > abs(magsobs[0])+0.2)])
        suspect = (abs(magsobs-np.median(magsobs[~nodata])) > 4.) * (magsobs != -99.999)
        UBVset = np.array(['Umag','Bmag','Vmag'])
        grizset = np.array(['gmag','rmag','imag', 'zmag'])
        if np.in1d(UBVset, fullmagnames[suspect]).sum() > 0:
            suspect = suspect | ((abs(magsobs-np.median(magsobs[~nodata]))>2.) * (np.in1d(fullmagnames, UBVset)))
        if np.in1d(grizset, fullmagnames[suspect]).sum() > 0:
            suspect = suspect | ((abs(magsobs-np.median(magsobs[~nodata]))>2.) * (np.in1d(fullmagnames, grizset)))
        if np.sum(suspect)>0:
            self.exclude = self.exclude + list(fullmagnames[suspect])

        self.exclude = np.unique(self.exclude)
        print "Excluding ", self.exclude

#        for ii in range(len(exclude)):
#            nodata = nodata | (fullmagnames == exclude[ii])
        # artificially inflate bad data points
        outlier_mask = np.array([(fullmagnames == ii) for ii in self.exclude])
        if len(outlier_mask)>0:
            emagsobs[np.sum(outlier_mask, axis=0).astype(bool)] = 4.0

        obsinds = np.arange(len(fullmagnames))[~nodata]
        self.ipname = np.concatenate((fullmagnames[~nodata], 
                               np.array(['mact', 'mkep', 'logl', 'logte', 'logg'])))
        self.ipinds = np.array([self.isodict[ii] for ii in self.ipname]).astype(int)
        
        self.maglams = fullmaglams[~nodata]
        self.magsobs = magsobs[obsinds]
        self.emagsobs = emagsobs[obsinds]

        if len(self.emagsobs)>0:
            self.emagsobs[self.emagsobs == -99.999] = np.clip(np.max(self.emagsobs), 0.025, 0.25)
        self.a_lam = redconvert[obsinds]

        return True
    
    def isoterpol(self, m0, z0, age):
        """Bi-linear interpolation given a mass, age, and metallicity 
        using Padova isochrones
        
        Parameters
        ----------
        m0 : float
            mass of star in solar mass
        z0 : float
            metallicity (i.e., z=0.017 for the Sun)
        age : float
            age in log10 (i.e., 6 for 1 Myr)
        
        Returns
        -------
        radius : float
            interpolated radius of star in solar radii
        fp : float array
            interpolated magnitudes, temp, and kepler flux
            
        Examples
        --------
        >>> keblat.isoterpol(1., 0.017, 9.4)
        """
        #age = np.log10(age)

        if (z0<=self.maxz) & (z0>=self.minz) & (age<=self.maxt) & (age>=self.mint):
            zbounds = np.digitize([z0], self.zvals)
            if zbounds == len(self.zvals):
                zbounds = np.concatenate((zbounds-2, zbounds-1))
            elif zbounds == 0:
                zbounds = np.concatenate((zbounds, zbounds+1))
            else:
                zbounds = np.concatenate((zbounds-1, zbounds))
            tbounds = np.digitize([age], self.tvals)
            if tbounds == len(self.tvals):
                tbounds = np.concatenate((tbounds-2, tbounds-1))
            elif tbounds == 0:
                tbounds = np.concatenate((tbounds, tbounds+1))
            else:
                tbounds = np.concatenate((tbounds-1, tbounds))
            
        else:
#            print "Error: z or age out of bounds! Note ",self.minz, 
#                "<z<", self.maxz, "and ", self.mint, "<logt<", self.maxt
            return (-np.inf, -np.inf)       

#        intergrid = np.empty((2, 2), dtype=object)
        intergrid = np.empty((2, 2, len(self.ipinds)))

        side = np.array(['left', 'right'])
        for ii in range(2):       
            goodindsz = np.searchsorted(self.iso[:, self.isodict['z']], 
                                        self.zvals[zbounds], side=side[ii])
            for jj in range(2):
                goodindst = np.searchsorted(self.iso[goodindsz[0]:goodindsz[1], 
                    self.isodict['logt']], self.tvals[tbounds], side=side[jj])
                mvals = np.unique(self.iso[goodindsz[0]:goodindsz[1], 
                            self.isodict['mini']][goodindst[0]:goodindst[1]])
                minm, maxm = np.min(mvals), np.max(mvals)
                if (m0<=maxm) & (m0>=minm):
                    mbounds = np.digitize([m0], mvals)
                    if mbounds == len(mvals):
                        mbounds = np.concatenate((mbounds-2, mbounds-1))
                    elif mbounds == 0:
                        mbounds = np.concatenate((mbounds, mbounds+1))
                    else:
                        mbounds = np.concatenate((mbounds-1, mbounds))
                    
                    sorted_minds = np.argsort(self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]])
#                    sorted_minds = np.arange(0, len(self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]]))
                    goodindsm0 = np.searchsorted(self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]], mvals[mbounds], sorter=sorted_minds)
                    goodindsm1 = np.searchsorted(self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]], mvals[mbounds], sorter=sorted_minds, side='right')
#                    if goodindsm0[0] == len(self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]]):
#                        goodindsm0 = goodindsm0-1
#                        goodindsm1 = goodindsm0
#                    if goodindsm0[0] == goodindsm1[1]:
#                        goodindsm1 = goodindsm0+1
#
                    try:
#                        intergrid[ii, jj] = interpolate.interp1d(self.iso[goodindsz[0]:goodindsz[1], self.isodict['mact']][goodindst[0]:goodindst[1]][goodindsm0[0]:goodindsm1[1]], self.iso[goodindsz[0]:goodindsz[1]][goodindst[0]:goodindst[1]][goodindsm0[0]:goodindsm1[1]][:, self.ipinds], axis=0, bounds_error=False)
                        intergrid[ii, jj, :] = np.array([np.interp(m0, self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]][sorted_minds][goodindsm0[0]:goodindsm1[1]], zz) for zz in self.iso[goodindsz[0]:goodindsz[1]][goodindst[0]:goodindst[1]][sorted_minds][goodindsm0[0]:goodindsm1[1]][:, self.ipinds].T])
                        #print "GOOD:", m0, age, z0, mvals[mbounds], self.tvals[tbounds], self.zvals[zbounds]
                        #print self.iso[goodindsz[0]:goodindsz[1], self.isodict['mini']][goodindst[0]:goodindst[1]][goodindsm0[0]:goodindsm1[1]]
                        #print self.iso[goodindsz[0]:goodindsz[1]][goodindst[0]:goodindst[1]][goodindsm0[0]:goodindsm1[1]][:, self.ipinds]
                    except Exception as e: 
                        print(e)
                        print "BAD:", m0, age, z0, mvals[mbounds], self.tvals[tbounds], self.zvals[zbounds], goodindsz, goodindst, goodindsm0, goodindsm1
                        return (-np.inf, -np.inf)
                        #print goodindsz, goodindst, goodindsm0, goodindsm1
                        #print self.iso[goodindsz[0]:goodindsz[1], self.isodict['mact']][goodindst[0]:goodindst[1]][goodindsm0[0]:goodindsm1[1]]
                        #print self.iso[goodindsz[0]:goodindsz[1]][goodindst[0]:goodindst[1]][goodindsm0[0]:goodindsm1[1]][:, self.ipinds]
                else:
                    #print "Error: mass out of bounds! Note 0.1<M/Msun<12"
                    return (-np.inf, -np.inf)

#        fq11 = intergrid[0,0](m0) #low z, low t
#        fq21 = intergrid[1,0](m0) #high z, low t
#        fq12 = intergrid[0,1](m0) #low z, high t
#        fq22 = intergrid[1,1](m0) #high z, high t
            
        zdiff = np.diff(self.zvals[zbounds])
        tdiff = np.diff(self.tvals[tbounds])
        fr1 = ((self.zvals[zbounds][1] - z0) / zdiff * intergrid[0,0,:]) + \
                ((z0 - self.zvals[zbounds][0]) / zdiff * intergrid[1,0,:])
        fr2 = ((self.zvals[zbounds][1] - z0) / zdiff * intergrid[0,1,:]) + \
                ((z0 - self.zvals[zbounds][0]) / zdiff * intergrid[1,1,:])
        fp = ((self.tvals[tbounds][1] - age) / tdiff * fr1) + \
                ((age - self.tvals[tbounds][0]) / tdiff * fr2)
        self.ipdict = dict(zip(self.ipname, list(np.arange(len(self.ipname)))))
        r1 = np.sqrt( 10**fp[self.ipdict['logl']] ) * (self.tsun/(10**fp[self.ipdict['logte']]))**2
#        r2 = np.sqrt( m0 * self.gsun / (10**fp[self.ipdict['logg']]) )
#        if abs(r1/r2 - 1.) > 0.05:
#            print "Greater than 5% difference in radii: ", r1, r2, r1/r2
#            print m0, z0, age
#            self.message="Greater than 5% difference in radii: {} {} {} for m,z,tau={},{},{}".format(r1, r2, r1/r2, m0, z0, age)
#            #return (-np.inf, -np.inf)
#        #radius = (r1+r2)/2.
#        #lum = 10**self.fp[ipdict['logl']]
        return r1, fp
    
    def isofit(self, isopars, marginalize_distance=False):
        """Returns extincted, interpolated magnitudes corresponding to observed 
        wavelengths
        
        Parameters
        ----------
        isopars : float array
            parameters = (m1, m2, z0, age, dist, ebv, h0, isoerr)
            masses in solar mass, age in log10, dist & scaleheight in pc
        
        Returns
        -------
        magsmod : float array
            dust-extincted model mags 
        
        Examples
        --------
        >>> keblat.isofit([1., 1., 0.017, 9.4, 800., 0.032, 119., 0.04])
        """
        if self.coeval:
            msum, mrat, z0, age, dist, ebv, h0, isoerr = isopars
            age2=age
        else:
            msum, mrat, z0, age, age2, dist, ebv, h0, isoerr = isopars
        m1, m2 = self.sumrat_to_12(msum, mrat)
        #age = np.log10(age)
        self.r1, fp1 = self.isoterpol(m1, z0, age)
        self.r2, fp2 = self.isoterpol(m2, z0, age2)
        if np.isinf(self.r1) or np.isinf(self.r2):
            #print "^Bad."
            return -np.inf

        mags1 = fp1[:len(self.magsobs)]
        mags2 = fp2[:len(self.magsobs)]
        self.temp1 = fp1[self.ipdict['logte']]
        self.temp2 = fp2[self.ipdict['logte']]
        self.logg1 = fp1[self.ipdict['logg']]
        self.logg2 = fp2[self.ipdict['logg']]
        self.f1 = fp1[self.ipdict['mkep']]
        self.f2 = fp2[self.ipdict['mkep']]
        self.frat = 10**((self.f2-self.f1)/(-2.5))
        self.mact1 = fp1[self.ipdict['mact']]
        self.mact2 = fp2[self.ipdict['mact']]

        self.updatepars(m1=m1, m2=m2, z0=z0, age=age, dist=dist,
                        ebv=ebv, h0=h0, isoerr=isoerr, msum=msum, mrat=mrat,
                        frat=self.frat, r1=self.r1, r2=self.r2,
                        rsum=self.r1+self.r2, rrat=self.r2/self.r1)

        absmagsmod = mags1 - 2.5 * np.log10(1. + 10**((mags1-mags2)/2.5))

        if marginalize_distance or dist<10. or dist>15000.:
            def magsmod_fn(x):
                magsmod = absmagsmod + 5. * np.log10(x / 10.) + \
                       self.a_lam * ebv * (1. - np.exp(-x * np.sin(self.glat * np.pi/180.) / h0))
                return np.sum(((magsmod-self.magsobs)/self.emagsobs)**2)
            res = sp_minimize(magsmod_fn, dist, method='L-BFGS-B', bounds=((10., 15000.),))
            dist = res.x
            self.message='marginalized distance'
        magsmod = absmagsmod + 5. * np.log10(dist / 10.) + \
            self.a_lam * ebv * (1. - np.exp(-dist * np.sin(self.glat * \
            np.pi/180.) / h0))
        return magsmod

    def loadlc(self, kic, properties, user_specified = None, 
               pdc = False, lc = True, clip_tol = 1.5, 
               raw=False, local_db=True, outdir='/astro/store/gradscratch/tmp/windemut/KEB/'):
        """Loads Kepler SAP from database
        
        Parameters
        ----------
        kic : int
            KIC #
        properties : list
            EB period, time of PE, PE width, SE width, separation btw. eclipses
            note: this information should come from loadvkeb() function
        user_specified : 5 x Ndata ndarray
            Default is None. If NOT none, user_specified must be an ndarray 
            with the following structure: 
                    JD, FLUX, DFLUX, QUARTER, CROWDSAP
            where each of the five specified arrays have length Ndata
        pdc : boolean
            if True, returns Kepler's PDC data instead of SAP. Default False
        lc : boolean
            if True, returns long cadence data (else short cadence data). Default True
        clip_tol : float
            specifies tolerance around each eclipse to fit the data and model. Default = 1.5, ie includes
            eclipse and 1/2 eclipse durations before and after eclipse.
        raw : boolean
            if True, returns raw counts, else median divided. Default False

        Returns
        -------
        True : boolean
            if jd, phase, flux, fluxerr, crowd, clip loads successfully
        
        Examples
        --------
        >>> keblat.loadvkeb()
        >>> kic = 9837578
        >>> goodv = np.where(keblat.vkeb[:, 0] == kic)[0]
        >>> keblat.loadlc(kic, keblat.vkeb[goodv, [1, 2, 5, 6, 7]])
        True
        """
        self.kic = kic
        (self.period, self.tpe, self.pwidth, self.swidth, self.sep) = properties

        if self.tpe > 50000.:
            print "Your time of primary eclipse is > 50,000. Subtracting 54833 (Kepler BJD offset) from input value."
            self.tpe -= 54833.
        self.tpe0 = self.tpe*1.0
        self.period0 = self.period*1.0
#        if user_specified is not None:
#            self.jd = user_specified[0, :]
#            self.flux, self.dflux = user_specified[1, :], user_specified[2, :]
#            self.quarter, self.crowd = user_specified[3, :], user_specified[4, :]
#            try:
#                self.quality = user_specified[5, :]
#            except:
#                self.quality = self.jd*0.
#            self.cadnum = None
        if user_specified is not None:
            print("Loading {}".format(user_specified))
            datanpz = np.load(user_specified)
            self.jd, self.flux, self.dflux, self.cadnum, self.quarter, self.quality, self._crowdsap = datanpz['jd'], datanpz['flux'], datanpz['dflux'], datanpz['cadnum'], datanpz['quarter'], datanpz['quality'], datanpz['_crowdsap']
            self.crowd = self.broadcast_crowd(self.quarter, self._crowdsap)
        else:
            try:
                from loadlc_db import loadlc_db, dbconnect
                _ = dbconnect()
                local_db_exists = True
            except Exception as e:
                print("{}, so using kplr instead".format(e))
                local_db_exists = False
            if local_db and local_db_exists:
                self.jd, self.flux, self.dflux, self.cadnum, self.quarter, self.quality, self._crowdsap = loadlc_db(kic, usepdc = pdc, lc = lc, raw = raw)
                self.crowd = self.broadcast_crowd(self.quarter, self._crowdsap)
            else:
                import kplr
                star = kplr.API().star(kic).get_light_curves(short_cadence=not lc)
                self.jd, self.flux, self.dflux = np.array([], dtype='float64'), np.array([], dtype='float64'), np.array([], dtype='float64')
                self.cadnum, self.quarter = np.array([], dtype='float64'), np.array([], dtype='float64')
                self.quality, self.crowd = np.array([], dtype='float64'), np.array([], dtype='float64')
                for ii in range(len(star)):
                    _s = star[ii].open(clobber=False)
                    _Npts = _s[1].data.shape[0]
                    self.jd = np.append(self.jd, _s[1].data['TIME'])
                    _fluxq = np.nanmedian(_s[1].data['SAP_FLUX'])
                    self.flux = np.append(self.flux, _s[1].data['SAP_FLUX']/_fluxq)
                    self.dflux = np.append(self.dflux, _s[1].data['SAP_FLUX_ERR']/_fluxq)
                    self.cadnum = np.append(self.cadnum, _s[1].data['CADENCENO'])
                    self.quarter = np.append(self.quarter, np.zeros(_Npts, dtype=int) + _s[0].header['QUARTER'])
                    self.quality = np.append(self.quality, _s[1].data['SAP_QUALITY'])
                    self.crowd = np.append(self.crowd, np.zeros(_Npts) + _s[1].header['CROWDSAP'])
                    naninds = np.isnan(self.flux)
                    self.jd = self.jd[~naninds]
                    self.dflux = self.dflux[~naninds]
                    self.cadnum = self.cadnum[~naninds]
                    self.quarter = self.quarter[~naninds]
                    self.quality = self.quality[~naninds]
                    self.crowd = self.crowd[~naninds]
                    self.flux = self.flux[~naninds]
                _, _uind = np.unique(self.quarter, return_index=True)
                self._crowdsap = self.crowd[_uind]
#                np.savez(outdir+str(kic), jd=self.jd, flux=self.flux, dflux=self.dflux, 
#                         cadnum=self.cadnum, quarter=self.quarter, quality=self.quality,
#                         crowd = self.crowd)
        a = np.nanmedian(self.flux)
        self.quarter = self.quarter.astype(int)
        self.cadnum = self.cadnum.astype(int)
        if a>1:
            self.flux = self.flux/a
            self.dflux = abs(self.dflux/a)
#         self.phase = ((self.jd-self.tpe) % self.period)/self.period
#         self.phase[self.phase<-np.clip(self.pwidth*3., 0., 0.2)]+=1.
#         self.phase[self.phase>np.clip(self.sep+self.swidth*3., self.sep, 1.0)]-=1.
#         self.clip_tol = clip_tol
#         self.clip = (abs(self.phase)<self.clip_tol*self.pwidth) | \
#                     (abs(self.phase-self.sep)<self.clip_tol*self.swidth)
        if kic == 6864859:
            bad = (self.jd<122.873) * (self.jd>122.228)
            self.jd = self.jd[~bad]
            self.flux = self.flux[~bad]
            self.dflux = self.dflux[~bad]
            self.cadnum = self.cadnum[~bad]
            self.quarter = self.quarter[~bad]
            self.quality = self.quality[~bad]
            self.crowd = self.crowd[~bad]
            print("Removed a weird chunk in LC for KIC {}".format(kic))

        self.updatephase(self.tpe, self.period, clip_tol=clip_tol)

        self.cadence = 0.0006811
        self.exp = 1

        self.fluxerr_tol = np.nanmedian(abs(np.diff(self.flux)))
        self.fluxerr = self.dflux.copy()
        self.updateErrors()
        if lc:
            self.cadence = 0.0204305556
            self.exp = 30
        print("LC data for KIC {0} loaded.".format(kic))
        return True

    def updateErrors(self, qtol=8, etol=10.):
        self.dflux = self.fluxerr.copy()
        self.dflux[(self.quality>qtol)] = etol*self.fluxerr_tol

    def updatephase(self, tpe, period, clip_tol=1.5):
        self.tpe = tpe
        self.period = period
        self.phase = ((self.jd-self.tpe) % self.period)/self.period
        self.phase[self.phase<-np.clip(self.pwidth*2., 0., 0.2)]+=1.
        self.phase[self.phase>1.-np.clip(self.pwidth*2., 0., 0.2)] -= 1.
        self.phase[self.phase>np.clip(self.sep+self.swidth*2., self.sep, 1.0)]-=1.
        self.clip_tol = clip_tol
        self.clip = ((abs(self.phase)<self.clip_tol*self.pwidth) | \
                     (abs(self.phase-1.0)<self.clip_tol*self.pwidth) | \
                    (abs(self.phase-self.sep)<self.clip_tol*self.swidth))
#        self.clip = self.clip * (self.quality >= 0)
        return True
        
    def loadvkeb(self, 
                 filename='data/kebproperties_0216.dat', 
                 user_specified=None):
        """Loads information from Villanova Kepler EB database into keblat.vkeb, 
        including period, time of PE, PE width, SE width, sep, morph, ecosw, esinw
        
        Parameters
        ----------
        filename : str
            name of villanova KEB properties file
        
        Returns
        -------
        True : boolean
            if keblat.vkeb loads successfully, has structure of:
            kic#, period, bjd0, pdepth, sdepth, pwidth, swidth, sep, morph ,ecosw, esinw
            
        Examples
        --------
        >>> keblat.loadvkeb()
        >>> print keblat.vkeb        
        """
        if user_specified is not None:
            self.vkeb = user_specified
        else:
            self.vkeb = np.loadtxt(filename,delimiter=";", 
                                   usecols=(0,1,2,3,4,5,6,7,8))
        # ^ kic#, period, bjd0, pdepth, sdepth, pwidth, swidth, sep
        ecosw = (self.vkeb[:, -2]*2. - 1.) * np.pi/4.
        esinw = ((self.vkeb[:, -4] / self.vkeb[:, -3]) - 1.) / ((self.vkeb[:, -4] / self.vkeb[:, -3]) + 1.)

#        switched = (self.vkeb[:, -3]<self.vkeb[:, -2])
#        esinw[switched] = ((self.vkeb[:, -2][switched] / self.vkeb[:, -3][switched]) - 1.) / \
#                          ((self.vkeb[:, -2][switched] / self.vkeb[:, -3][switched]) + 1.)


        self.vkeb[:, 2][self.vkeb[:,2]>50000] -= 54833.0
        self.vkeb = np.hstack((self.vkeb, ecosw[:, np.newaxis], esinw[:, np.newaxis]))
        return True        

    def start_errf(self, erfname):
        """Starts the error file
        Parameters
        ----------
        erfname : str
            name of error file
        """
        self.erfname = erfname
        errf = open(self.erfname, "w")
        errf.close()
        return True

    def lctemplate(self, lcpars, period, omega, e, a, inc, bgr, ldcoeffs, 
                   rrat, tc, t0, cadence, exp, pe=True):
        """Computes a template Mandel & Agol (2002) eclipse lightcurve with
        correction for long-cadence binning (Kipping 2013)

        Parameters
        ----------
        period : float
            period of EB
        omega : float
            argument of periastron
        e : float
            eccentricity
        a : float
            semi-major axis in AU
        inc : float
            inclination; 90 = edge on
        bgr : float
            radius of star being eclipsed in solar radius
        ldcoeffs: float array
            limb darkening coefficients; tuple if quadratic, 
            4 elements if quartic non-linear law
        rrat : float
            ratio of radii (eclipsing/eclipsed)
        tc : float
            time of center of eclipse (either PE or SE)
        t0 : float
            time of periastron passage
        cadence : float
            cadence (in days)
        exp : int
            number of Kepler exposures to bin (LC = 30, SC = 1)
        pe : boolean
            if True, template for PE
            
        Returns
        -------
        tmean : float array
            array of times for PE (or SE)
        resmean : float array
            array of flux values for PE (or SE)
        
        """
        
        if pe:
            half0 = self.pwidth*period * 2.
        else:
            half0 = self.swidth*period * 2.

        #half0 = period/4.

        t = np.linspace(-half0, +half0, int(2*half0/0.0006811) + 1)
        t += tc
        maf = rsky(e, period, t0, 1e-8, t)
        r = a*(1.-e**2) / (1.+e*np.cos(maf))
        z = r*np.sqrt(1.-np.sin(omega+maf)**2*np.sin(inc)**2)
        if not np.all(np.isfinite(z)):
            badz = np.isfinite(z)
            print t0, maf[~badz], e, a, inc, z[~badz], r[~badz]
            errfile = open(self.erfname, "a")
            errfile.write("inf z: {0} {1} {2} {3} {4} {5} {6}\n".format(e, a, inc, t[~badz], maf[~badz], r[~badz], z[~badz]))
            errfile.close()
            try:
                z[~badz] = np.interp(t[~badz], t[badz], z[badz])
            except:
                return -np.inf, -np.inf

        if self.ldtype == 0:
            # print "non-linear case:", ldcoeffs
            res = self.occultnltheta(z/(bgr*r2au), rrat, ldcoeffs)
        else:
            # print "Quad case:", ldcoeffs
            res = occultquad(z/(bgr*r2au), ldcoeffs[0], ldcoeffs[1], rrat, len(z))

        bad = (res<-1e-4) | (res-1.>1e-4)        
        if np.sum(bad)>0:
            cz = z[bad]/(bgr*r2au)
            interp_res = np.interp(t[bad], t[~bad], res[~bad])
            errfile = open(self.erfname, "a")
            errfile.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(min(z)/(bgr*r2au), max(z)/(bgr*r2au), ldcoeffs[0], ldcoeffs[1], rrat, " ".join([str(ii) for ii in cz]), " ".join([str(ii) for ii in res[bad]]), " ".join([str(ii) for ii in interp_res])))
            errfile.close()
            res[bad] = interp_res
        if cadence < 0.02:
            return t, res
        as_strided = np.lib.stride_tricks.as_strided
        tt = as_strided(t, (len(t)+1-self.exp, self.exp), (t.strides * 2))
        rres = as_strided(res, (len(res)+1-self.exp, self.exp), (res.strides * 2))
        if pe:
            self.pe_dur = tt[(rres<1.)][-1] - tt[(rres<1.)][0] if np.sum(rres<1.)>1 else 0
            self.pe_depth = np.min(rres)
        else:
            self.se_dur = tt[(rres<1.)][-1] - tt[(rres<1.)][0] if np.sum(rres<1.)>1 else 0
            self.se_depth = np.min(rres)
        return np.mean(tt, axis=1), np.mean(rres, axis=1)
        
    def rvprep(self, t, rv1, rv2, drv1, drv2):
        """Stores observed radial velocity data points

        Parameters
        ----------
        t : float array or scalar
            times of observations
        rv1 : float array or scalar
            RV of primary in m/s
        rv2 : float array or scalar
            RV of secondary in m/s
        dr1 : float array of scalar
            RV err of primary in m/s
        dr2 : float array or scalar
            RV err of secondary in m/s

        Returns
        -------
        m1, m2, k0 : tuple
            guesses for the masses and systemic velocity of the binary from the RV semi-amplitudes
        """
        self.rv1_obs = rv1
        self.rv2_obs = rv2
        self.rv1_err_obs = drv1
        self.rv2_err_obs = drv2
        self.rv_t = t
        self.bad1 = np.isnan(self.rv1_obs) | np.isnan(self.rv1_err_obs) | (self.rv1_err_obs == 0.)
        self.bad2 = np.isnan(self.rv2_obs) | np.isnan(self.rv2_err_obs) | (self.rv2_err_obs == 0.)
        k1 = (np.nanmax(self.rv1_obs[~self.bad1]) - np.nanmin(self.rv1_obs[~self.bad1]))/2.
        k2 = (np.nanmax(self.rv2_obs[~self.bad2]) - np.nanmin(self.rv2_obs[~self.bad2]))/2.
        k0 = np.nanmedian(np.append(self.rv1_obs[~self.bad1], self.rv2_obs[~self.bad2]))
        m1, m2 = self.rvpars_guess_mass(k1, k2, self.pars['period'],
                                        np.sqrt(self.pars['esinw']**2 + self.pars['ecosw']**2))
        return m1, m2, k0


    def rvfit(self, rvpars, t):
        """Computes the radial velocities of each binary component.

        Parameters
        ----------
        rvpars : float array or list
                msum, mrat, period, tpe, esinw, ecosw, inc, k0, rverr
        t : float array or scalar
                times of observations to compute RV

        Returns
        -------
        vr1, vr2 : tuple
                RVs in m/s, where vr1 and vr2 are of shape/type of input t
        """

        msum, mrat, period, tpe, esinw, ecosw, inc, k0, rverr = rvpars
        e = np.sqrt(esinw**2+ecosw**2)
        omega = np.arctan2(esinw, ecosw)
        inc = inc % TWOPI

        fpe = np.pi/2. - omega

        m1, m2 = self.sumrat_to_12(msum, mrat)

        self.updatepars(msum=msum, mrat=mrat, m1=m1, m2=m2, period=period, tpe=tpe,
                        esinw=esinw, ecosw=ecosw, k0=k0, rverr=rverr, inc=inc)

#        t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
#            (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
#            np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))

        t0 = tpe - self.sudarsky(fpe, e, period)
        maf = rsky(e, period, t0, 1e-8, t)
        amp = 29794.509 / np.sqrt(1-e**2) * (period/d2y)**(-1/3.) / (m1+m2)**(2/3.)
        vr2 = -amp * m1 * np.sin(inc+np.pi) * \
              (np.cos(omega+maf) + e * np.cos(omega))
        omega+=np.pi # periapse of primary is 180 offset
        vr1 = -amp * m2 * np.sin(inc+np.pi) * \
              (np.cos(omega+maf) + e * np.cos(omega))
        return vr1+k0, vr2+k0


    def lcfit(self, lcpars, jd, quarter, flux, dflux, crowd,
              polyorder=2, ooe=True):
        """Computes light curve model
        
        Parameters
        ----------
        lcpars : float array
            parameters for LC fitting: 
            msum, rsum, rratio, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4
        jd : float array
            time array
        quarter : float array
            corresponding kepler quarter for a given time
        flux : float array
            observed flux
        dflux : float array
            flux error
        crowd : float array
            array of crowding values (additional flux)
        polyorder : int
            order of polynomial to detrend lightcurve
        
        Returns
        -------
        totmod : float array
            array of model fluxes
        totpol : float array
            array of polynomials for detrending
        """
        # r1, r2, frat derive from m1, m2, z0, t0, dist, E(B-V), scaleheight
        msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, \
            q1, q2, q3, q4 = lcpars

        # LD transformations (Kipping 2013)
        c1 = 2.*np.sqrt(q1)*q2
        c2 = np.sqrt(q1)*(1.-2.*q2)
        c3 = 2.*np.sqrt(q3)*q4
        c4 = np.sqrt(q3)*(1.-2.*q4)
        ldcoeffs1 = np.array([c1, c2])
        ldcoeffs2 = np.array([c3, c4])
            
#        if r2 > r1:
#            r1, r2 = r2, r1
#            m1, m2 = m2, m1
#            frat = 1./frat
        omega=np.arctan2(esinw,ecosw)
        e=np.sqrt(esinw**2+ecosw**2)

        # nip it at the bud.
        if (e>=1.):
            #print "e>=1", e
            return -np.inf, -np.inf
            
        # r1 = rsum/(1.+rrat)
        # r2 = rsum/(1.+1./rrat)
        r1, r2 = self.sumrat_to_12(rsum, rrat)
        a = self.get_a(period, msum)
        inc = self.get_inc(b, r1, a) % TWOPI
        #inc = np.arccos(b*r1/(a/r2au))
        
        if np.isnan(inc):
            #print "inc is nan", inc
            return -np.inf, -np.inf

        self.updatepars(msum=msum, rsum=rsum, rrat=rrat, period=period, tpe=tpe,
                       esinw=esinw, ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4,
                       frat=frat, r1=r1, r2=r2, inc=inc)
        fpe = np.pi/2. - omega
        fse = -np.pi/2. - omega

        # transform time of center of PE to time of periastron (t0)
        # from Eq 9 of Sudarsky et al (2005)
        t0 = tpe - self.sudarsky(fpe, e, period)
        tse = t0 + self.sudarsky(fse, e, period)
        # t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
        #     (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
        #     np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))
        # tse = t0 + (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
        #     (e*np.sin(fse)/(1.+e*np.cos(fse)) - 2.*(1.-e**2)**(-0.5) * \
        #     np.arctan(np.sqrt(1.-e**2) * np.tan((fse)/2.) / (1.+e)))
        self.tpe = tpe
        self.tse = tse
        # if tse<tpe:
        #     tse+=period
            
        tempt1, tempres1 = self.lctemplate(lcpars, period, omega, e, a, inc, r1,
                                           ldcoeffs1, r2/r1, tpe, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=True)

        tempt2, tempres2 = self.lctemplate(lcpars, period, omega, e, a, inc, r2,
                                           ldcoeffs2, r1/r2, tse, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=False)

        if np.any(np.isinf(tempt1)) or np.any(np.isinf(tempt2)):
            return -np.inf, -np.inf
        
        tempt1 = tempt1 % period
        tempt2 = tempt2 % period
        tempres1 = (tempres1 - 1.)/(1. + frat) + 1.
        tempres2 = (tempres2 - 1.)/(1. + 1./frat) + 1.

        sorting1 = np.argsort(tempt1)
        sorting2 = np.argsort(tempt2)

        tempres1 = tempres1[sorting1]
        tempt1 = tempt1[sorting1]
        tempres2 = tempres2[sorting2]
        tempt2 = tempt2[sorting2]

        #not including crowdsap term.
        #tempres1 = (tempres1 + frat) / (1.+frat)
        #tempres2 = (tempres2 * frat + 1.) / (1. + frat)
        totmod, totpol = np.ones(len(jd)), np.ones(len(jd))

        maf = rsky(e, period, t0, 1e-8, jd)
        r = a*(1.-e**2) / (1.+e*np.cos(maf))
        zcomp = np.sin(omega+maf) * np.sin(inc) 
        pe = ((r*zcomp>0.)) #& (z <= 1.05*(r1+r2)*r2au))
        se = ((r*zcomp<0.)) #& (z <= 1.05*(r1+r2)*r2au))
        tt = jd % period
        if pe.any():
            totmod[pe] = np.interp(tt[pe], tempt1, tempres1)
            totmod[pe] = (totmod[pe] - 1.) * crowd[pe] + 1.
        if se.any():
            totmod[se] = np.interp(tt[se], tempt2, tempres2)
            totmod[se] = (totmod[se] - 1.) * crowd[se] + 1.
        
        if polyorder>0:
            if (self.sep-self.clip_tol*(self.pwidth+self.swidth) < self.pwidth):
                chunk = np.array(np.where(np.diff(jd) > np.median(np.diff(jd))*4.))[0]
            else:
                chunk = np.array(np.where(np.diff(jd) > self.pwidth*period))[0]
            #put in dummy first and last element # placeholders
            
            chunk = np.append(chunk, len(jd)-2).flatten()
            _, chunk3 = np.unique(np.searchsorted(jd[chunk], jd), return_index=True)
            chunk=chunk3
            chunk[-1]+=1
            chunk = np.unique(np.sort(np.append(chunk, np.where(np.diff(quarter)>0)[0]+1)))
            totpol = poly_lc_cwrapper(jd, flux, dflux, totmod, chunk, porder=polyorder, ooe=ooe)
#            phase = ((jd - tpe) % period) / period
#            sorting = np.argsort(phase)
#            nopoly = (totpol[sorting] == 1.)
#            if (np.sum(nopoly)>0) and (np.sum(nopoly)<len(totpol)*0.1):
#                _totpol = totpol[sorting]
#                tmp = np.interp(phase[sorting][nopoly], phase[sorting][~nopoly], flux[sorting][~nopoly]/totpol[sorting][~nopoly])
#                #print np.sum(nopoly), np.sum(~nopoly)
#                _totpol[nopoly] = flux[sorting][nopoly] / tmp
#                totpol[sorting] = _totpol
        return totmod, totpol
        
    @staticmethod
    def ephem(N, period, tpe):
        return np.arange(N)*period + tpe

    def ilnlike(self, fisopars, lc_constraints=None, ebv_dist=None, ebv_arr=None,
                residual=False, retpars=False):
        """Computes log likelihood of isochrone fit portion of KEBLAT
        
        Parameters
        ----------
        fisopars : dict or float array
            either from lmfit parameter class or just an array of vals
        residual: boolean
            True if want to return residual array
            False if return loglikelihood val
        
        Returns
        -------
        loglike : float
            returns -np.inf if model mags have invalid values
        isores : float array
            if residual = True
        
        """
        if self.coeval:
            parnames = parnames_dict['sed']
        else:
            parnames = parnames_dict['sed2']
        isopars=np.empty(len(parnames))
        for jj in range(len(parnames)):
            try:
                isopars[jj] = fisopars[parnames[jj]].value
            except KeyError:
                isopars[jj] = 0.0
            except ValueError:
                isopars[jj] = fisopars[jj]
            except IndexError:
                isopars[jj] = fisopars[jj]
        if ebv_arr is not None:
            isopars[5] = np.interp(isopars[4], ebv_dist, ebv_arr)
        if retpars:
            return isopars
        # m1 = isopars[0] / (1. + isopars[1])
        # m2 = isopars[0] / (1. + 1./isopars[1])
        # isopars[0], isopars[1] = m1, m2
        # print m1, m2, isopars[0], isopars[1]
        isoerr = np.exp(isopars[-1])
        magsmod = self.isofit(isopars)
 #/ np.sqrt(self.emagsobs**2 + isoerr**2)
        if np.isinf(self.r2) or np.isinf(self.r1):
            if lc_constraints is None:
                lc_block = np.array([])
            else:
                lc_block = lc_constraints
            if residual:
                return np.ones(len(self.magsobs) + len(lc_block))*1e12
            else:
            	return -np.inf, str((0,0,0))
            
        if np.any(np.isinf(magsmod)):
            if residual:
                return np.ones(len(self.magsobs) + len(lc_constraints))*1e12
            return -np.inf, str((0,0,0))

        lc_inputs = np.array([(self.r1+self.r2)/(isopars[0])**(1./3.), self.r2/self.r1, self.frat])

        if lc_constraints is None:
            lc_priors = np.array([])
        else:
            lc_priors = (lc_inputs-lc_constraints)/(np.array([0.05, 0.02, 0.02]) * lc_constraints)

#        lc_uncertainty = np.array([0.002, 0.002, 0.002])

        isores = np.concatenate(((magsmod - self.magsobs) / np.sqrt(self.emagsobs**2 + isoerr**2),
                                 lc_priors)) #/ np.sqrt(self.emagsobs**2 + isoerr**2)
        for ii, dii, jj in zip([self.armstrongT1, self.armstrongT2],
                               [self.armstrongdT1, self.armstrongdT2],
                               [10**self.temp1, 10**self.temp2]):
            if ii is not None:
                if dii is None:
                    dii=0.05*ii
                isores = np.append(isores, (ii-jj)/dii)
        if residual:
            return isores

        chisq = np.sum(isores**2) + np.sum(np.log(TWOPI * \
                    (self.emagsobs**2 + isoerr**2)))
#        chisq += ((isopars[5] - self.ebv)/(self.debv))**2
        #chisq += ((isopars[6] - 119.)/(15.))**2
        return -0.5*chisq, str((self.r1, self.r2, self.frat))
    
    def ilnprior(self, isopars):
        """Returns log-prior for isochrone fitting portion only
        
        Parameters
        ----------
        isopars : float array
            array of val for (m1, m2, z0, age, dist, ebv, h0, isoerr)        
        
        Returns
        lnp : float
            0 if within flat priors, -np.inf if outside
        """
        #m1, m2, z0, age, dist, ebv, h0, lnisoerr = isopars
#        bounds = np.array([(.1, 12.), (.1, 12.), (0.001, 0.06), (6., 10.1),
#                           (10., 15000.), (0.0, 1.0), (119-20., 119+20.),
#                            (-8, 1.)])
        bounds = np.array([self.parbounds[ii] for ii in ['msum', 'mrat', 
                           'z0', 'age', 'dist', 'ebv', 'h0', 'isoerr']])

        pcheck = np.all((np.array(isopars) >= bounds[:, 0]) & \
                        (np.array(isopars) <= bounds[:, 1]))
        if pcheck:
            return 0.0 + isopars[3]*np.log(10.) + np.log(np.log(10.))
        else:
            return -np.inf
    
    def ilnprob(self, isopars, lc_constraints=None):
        """Returns log probability (loglike + logprior)
        
        Parameters
        ----------
        isopars : float array
            array of val for (m1, m2, z0, age, dist, ebv, h0, isoerr)        
        
        Returns
        lnprob : float
            -np.inf if invalid values     
        """
        lp = self.ilnprior(isopars)
        if np.isinf(lp):
            return -np.inf, str((-np.inf, -np.inf, -np.inf))
        ll, blobs = self.ilnlike(isopars, lc_constraints=lc_constraints)
        if (np.isnan(ll) or np.isinf(ll)):
            return -np.inf, str((-np.inf, -np.inf, -np.inf))
        return lp + ll, blobs

    def lnlike(self, allpars, lc_constraints=None, qua=[1], polyorder=2,
               residual=False, ld4=False, clip=None, ooe=True, crowd_fit=None,
               period0=None, tpe0=None):
        """Returns loglikelihood for both SED + LC fitting
        
        Parameters
        ----------
        allpars : float array
            array of parameter values
        qua : list
            list of quarters to include in LC analysis; default = [1]
        polyorder : int
            order of polynomial for detrending; default = 2
        residual : boolean
            True if want to return (weighted) residual array
        ld4 : boolean
            True if want quartic non-linear LD law; default = False (quadratic)
            
        Returns
        -------
        lili : float
            log likelihood value of model; returns -np.inf if invalid
        res : float array
            residuals of model and data if residuals = True
        """
        
        if ld4:
            self.ldtype=0
#            allpars = self.getvals(fitpars, 22)
            msum, mrat, z0, age, dist, ebv, h0, period, tpe, esinw, \
                ecosw, b, q11, q12, q13, q14, q21, q22, q23, q24, \
                lcerr, isoerr  = allpars
            ldcoeffs = np.array([q11, q12, q13, q14, q21, q22, q23, q24])        
        else:
            self.ldtype=1
#            allpars = self.getvals(fitpars, 18)
            msum, mrat, z0, age, dist, ebv, h0, period, tpe, esinw, \
                ecosw, b, q1, q2, q3, q4, lcerr, isoerr = allpars
            ldcoeffs = np.array([q1, q2, q3, q4])
            # m1 = msum / (1. + mrat)
            # m2 = msum / (1. + 1./mrat)
            self.updatepars(z0=z0, age=age, dist=dist, ebv=ebv,
                            h0=h0, period=period, tpe=tpe, esinw=esinw, 
                            ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4, 
                            lcerr=lcerr, isoerr=isoerr, msum=msum, mrat=mrat)
        # do isochrone matching first
        isopars = [msum, mrat, z0, age, dist, ebv, h0, isoerr]
        magsmod = self.isofit(isopars)
        if np.any(np.isinf(magsmod)):
            return -np.inf #/ np.sqrt(self.emagsobs**2 + isoerr**2)

        lc_inputs = np.array([(self.r1+self.r2)/(msum)**(1./3.), self.r2/self.r1, self.frat])

        if np.any(np.isinf(lc_inputs)):
            return -np.inf
        if lc_constraints is None:
            lc_priors = np.array([])
        else:
            lc_priors = (lc_inputs-lc_constraints)/(0.03*lc_constraints)
        isores = np.concatenate(((magsmod - self.magsobs) / np.sqrt(self.emagsobs**2 + isoerr**2),
                                 lc_priors)) #/ np.sqrt(self.emagsobs**2 + isoerr**2)
        for ii, dii, jj in zip([self.armstrongT1, self.armstrongT2],
                               [self.armstrongdT1, self.armstrongdT2],
                               [10**self.temp1, 10**self.temp2]):
            if ii is not None:
                if dii is None:
                    dii=0.05*ii
                isores = np.append(isores, (ii-jj)/dii)
        chisq = np.sum(isores**2) + np.sum(np.log((self.emagsobs**2 + isoerr**2)))

        #now the light curve fitting part
        lcpars = np.concatenate((np.array([msum, self.r1+self.r2, 
                                           self.r2/self.r1, period, 
                                           tpe, esinw, ecosw, b, self.frat]),
                                           ldcoeffs))

        # conds = [self.quarter == ii for ii in np.array(qua)]
        # conds = np.sum(np.array(conds), axis=0)
        # conds = np.array(conds, dtype=bool)
        #self.clip2 = (self.clip) #* conds
        if clip is None:
            clip = self.clip
        if crowd_fit is None:
            crowding = self.crowd[clip]
        else:
            crowding = self.broadcast_crowd(self.quarter, crowd_fit)[clip]
        lcmod, lcpol = self.lcfit(lcpars, self.jd[clip],
                                  self.quarter[clip], self.flux[clip],
                                self.fluxerr[clip], crowding,
                                polyorder=polyorder, ooe=ooe)

        if np.any(np.isinf(lcmod)):
            return -np.inf

        lcres = (lcmod*lcpol - self.flux[clip]) / np.sqrt(self.fluxerr[clip]**2 + lcerr**2)

        totres = np.concatenate((isores, lcres))
        self.chi = np.sum(totres**2)
        
        chisq += np.sum(lcres**2) + \
                    np.sum(np.log((self.fluxerr[clip]**2 + lcerr**2)))
#        chisq += ((isopars[5] - self.ebv)/(self.debv))**2
#        chisq += ((isopars[2] - self.z0)/(0.2 * np.log(10) * self.z0))**2

        #chisq += ((isopars[6] - 119.)/(15.))**2        

        if residual:
            # print lc_priors, totres.shape, isores.shape, self.magsobs.shape, self.jd[self.clip].shape, \
            #     self.clip.sum(), clip.sum(), lcres.shape, lcmod.shape, lcpol.shape
            return totres
        lili = -0.5 * chisq
        return lili

    def gaia_prior(self, r_est, r_lo, r_hi):
        self.dist = r_est #pc
        self.dist_err = (r_hi-r_lo)/2.0
        return

    def get_blobs(self):
        return self.r1, self.r2, self.frat, self.logg1, self.logg2, self.temp1, self.temp2, self.f1, self.f2

    @staticmethod
    def rvpars_guess_mass(k1, k2, P, e):
        mrat = k1/k2
        msum = ((k1+k2)/29794.509)**3 * (1-e**2)**(3./2.) * (P/d2y)
        return msum/(1.+mrat), msum/(1.+1./mrat)

    def lnlike_lcrv(self, lcrvpars, qua=[1], polyorder=2, residual=False, 
                    clip=None, ooe=True):
        self.ldtype = 1
        #            allpars = self.getvals(fitpars, 18)
        msum, mrat, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4, lcerr, k0, rverr = lcrvpars

        m1, m2 = self.sumrat_to_12(msum, mrat)
        self.updatepars(m1=m1, m2=m2, period=period, tpe=tpe, esinw=esinw,
                        ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4,
                        lcerr=lcerr, k0=k0, rverr=rverr, frat=frat)

        lcpars = [msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4]
        # conds = [self.quarter == ii for ii in np.array(qua)]
        # conds = np.sum(np.array(conds), axis=0)
        # conds = np.array(conds, dtype=bool)
        # self.clip2 = (self.clip) #* conds
        if clip is None:
            clip = self.clip
        lcmod, lcpol = self.lcfit(lcpars, self.jd[clip],
                                  self.quarter[clip], self.flux[clip],
                                  self.fluxerr[clip], self.crowd[clip],
                                  polyorder=polyorder, ooe=ooe)
        lcres = (lcmod * lcpol - self.flux[clip]) / np.sqrt(self.fluxerr[clip] ** 2 + lcerr ** 2)

        if np.any(np.isinf(lcmod)):
            return -np.inf

        rvpars = [msum, mrat, period, tpe, esinw, ecosw, self.pars['inc'], k0, rverr]
        rv1, rv2 = self.rvfit(rvpars, self.rv_t)

        rvres = np.concatenate(((rv1[~self.bad1] - self.rv1_obs[~self.bad1]) /
                                np.sqrt(self.rv1_err_obs[~self.bad1] ** 2 + rverr ** 2),
                                (rv2[~self.bad2] - self.rv2_obs[~self.bad2]) /
                                np.sqrt(self.rv2_err_obs[~self.bad2] ** 2 + rverr ** 2)))

        totres = np.concatenate((lcres, rvres))
        self.chi = np.sum(totres ** 2)
        if residual:
            return totres

        chisq = np.sum(lcres ** 2) + \
                 np.sum(np.log((self.fluxerr[clip] ** 2 + lcerr ** 2)))
        chisq += np.sum(rvres ** 2) + \
                 np.sum(np.log((self.rv1_err_obs[~self.bad1] ** 2 + rverr ** 2))) + \
                 np.sum(np.log((self.rv2_err_obs[~self.bad2] ** 2 + rverr ** 2)))

        lili = -0.5 * chisq
        return lili

    def lnprior_lcrv(self, lcrvpars):
        """Returns logprior of LC + RV model
        
        Parameters
        ----------
        allpars : float array
            full list of parameters:
            (msum, mrat, rsum, rrat, period, tpe, esinw, ecosw, b, frat
            q1, q2, q3, q4, lcerr, k0, rverr)
            
        Returns
        -------
        res : float
            0 if within flat priors, -np.inf if without
        """        
        msum, mrat, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4, lcerr, k0, rverr = lcrvpars
        e = np.sqrt(esinw**2 + ecosw**2)
        pars2check = np.array([msum, mrat, rsum, rrat, period, tpe, e, b, frat, 
                               q1, q2, q3, q4, lcerr, k0, rverr])
        bounds = np.array([self.parbounds[ii] for ii in ['msum', 'mrat', 'rsum', 
                           'rrat', 'period', 'tpe', 'e', 'b', 'frat', 
                               'q1', 'q2', 'q3', 'q4', 'lcerr', 'k0', 'rverr']])
        bounds[-1,:] = [-8, 12]
        bounds[-3,:] = [-16, -3]
        pcheck = np.all((pars2check >= bounds[:,0]) & (pars2check <= bounds[:,1]))
        if pcheck:
            return 0.0 - np.log(e)
        else:
            return -np.inf
    

    def lnprior(self, allpars, gaussprior=False):
        """Returns logprior of SED + LC model
        
        Parameters
        ----------
        allpars : float array
            full list of parameters:
            (m1, m2, z0, age, dist, ebv, h0, period, tpe, 
            esinw, ecosw, b, q1, q2, q3, q4, lcerr, isoerr)
            
        Returns
        -------
        res : float
            0 if within flat priors, -np.inf if without
        """
        msum, mrat, z0, age, dist, ebv, h0, period, tpe, esinw, ecosw, \
            b, q1, q2, q3, q4, lcerr, isoerr = allpars
        e = np.sqrt(esinw**2 + ecosw**2)
        pars2check = np.array([msum, mrat, z0, age, dist, ebv, h0, \
            period, tpe, e, b, q1, q2, q3, q4, lcerr, isoerr])
        bounds = np.array([self.parbounds['msum'], self.parbounds['mrat'], 
                           self.parbounds['z0'], self.parbounds['age'],
                           self.parbounds['dist'], self.parbounds['ebv'], 
                           self.parbounds['h0'], self.parbounds['period'], 
                           self.parbounds['tpe'], self.parbounds['e'], self.parbounds['b'], 
                           self.parbounds['q1'], self.parbounds['q2'],
                           self.parbounds['q3'], self.parbounds['q4'], 
                           self.parbounds['lcerr'], self.parbounds['isoerr']])

        # FLAT PRIORS
        pcheck = np.all((pars2check >= bounds[:,0]) & \
                        (pars2check <= bounds[:,1]))
        
        # GAUSSIAN PRIORS
        gpcheck = ((period-self.period0)/(0.02*self.period0))**2 + \
                        ((tpe - self.tpe0)/(0.05*self.tpe0))**2 
        gpcheck *= -0.5

        if pcheck:
            return 0.0 + age*np.log(10.) + np.log(np.log(10.)) + gaussprior*gpcheck - np.log(e)
        else:
            return -np.inf
    
    def lnprob(self, allpars, qua=[1], gaussprior=False):
        """Returns logprob of SED + LC model
        
        Parameters
        ----------
        allpars : float array
            full list of parameters:
            (m1, m2, z0, age, dist, ebv, h0, period, tpe, 
            esinw, ecosw, b, q1, q2, q3, q4, lcerr, isoerr)
        qua : list
            list of integers for Kepler quarters to be included in analysis
            
        Returns
        -------
        lnprob, blob : tuple 
            lnprob is lnprior + lnlike
            blob is a string of chi^2, r1, r2, f2/f1 values
        """
        
        lp = self.lnprior(allpars, gaussprior=gaussprior)
        if np.isinf(lp):
            return -np.inf, str((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf))
        ll = self.lnlike(allpars, qua=qua)
        if (np.isnan(ll) or np.isinf(ll)):
            return -np.inf, str((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf))
        return lp + ll, str((self.chi, self.r1, self.r2, self.frat, self.mact1, self.mact2))

    def plot_sed(self, mlpars, prefix, suffix='sed', lc_constraints=None, 
                       savefig=True):
        isoerr = mlpars[-1]
        if isoerr < 0:
            isoerr = np.exp(isoerr)
        magsmod = self.isofit(mlpars)
        #magsmod, r, T, logg = isofit_single(mlpars[:5])
        if np.any(np.isinf(magsmod)):
            print "Input isopars give -inf magsmod: ", mlpars, magsmod
            return magsmod
        plt.figure()
        plt.subplot(311)
        plt.errorbar(self.maglams, self.magsobs, 
                     np.sqrt(self.emagsobs**2 + isoerr**2), fmt='k.')
        plt.plot(self.maglams, magsmod, 'r.')
        plt.ylabel('magnitude')
    
        plt.subplot(312)
        plt.errorbar(self.maglams, self.magsobs-magsmod, 
                     np.sqrt(self.emagsobs**2 + isoerr**2), fmt='k.')
        plt.xlabel('wavelength (angstrom)')
        plt.ylabel('data-model')
        plt.suptitle('KIC '+str(self.kic) +' SED (only)')
    
        plt.subplot(313)
        if lc_constraints is not None:
            plt.plot(lc_constraints, 'kx')
        lc_inputs = np.array([(self.r1 +self.r2)/(mlpars[0]+mlpars[1])**(1./3.), 
                              self.r2/self.r1, self.frat])
        plt.plot(lc_inputs, 'rx')
        plt.xlim((-1, 3))
        if savefig:
            plt.savefig(prefix + suffix+'.png')
        return magsmod

        
    def plot_lc(self, lcpars, prefix, suffix='lc', savefig=True, 
                      polyorder=2, ooe=True, clip_tol=1.5, crowd=None, thresh=10.):
        if crowd is None:
            crowd = self.crowd
        self.updatephase(lcpars[4], lcpars[3], clip_tol=self.clip_tol)
        if len(lcpars)>13:
            self.pars['lcerr'] = lcpars[-1]
            lcpars = lcpars[:-1]
        lcmod, lcpol = self.lcfit(lcpars, self.jd[self.clip], self.quarter[self.clip],
        				self.flux[self.clip], self.dflux[self.clip],
        				crowd[self.clip], polyorder=polyorder, ooe=ooe)
    
        lcres = self.flux[self.clip] - lcmod*lcpol
        lc_MAD = np.nanpercentile(abs(lcres), 67)
        
        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax.errorbar(self.phase[self.clip], self.flux[self.clip]/lcpol,
                     self.dflux[self.clip], marker='.', color='0.3', 
                     linestyle="", ecolor='0.9', zorder=1)
        ax.plot(self.phase[self.clip], lcmod, marker='.', color='indianred', linestyle="", lw=0, zorder=10)
        ax.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
        ax.set_ylabel('Kepler Flux')
    
        divider = make_axes_locatable(ax)
        axb = divider.append_axes("bottom", size="25%", pad=0, sharex=ax)
        axb.errorbar(self.phase[self.clip], lcres,
                     np.sqrt(self.dflux[self.clip]**2 + self.pars['lcerr']**2), 
                     marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
    
        axb.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
        axb.set_ylabel('Data - Model')
        axb.set_xlabel('Phase (Primary Eclipse)')
        #axb.set_yticklabels(axb.yaxis.get_majorticklabels()[1:])
    
        ax2.errorbar(self.phase[self.clip], self.flux[self.clip]/lcpol,
                     self.dflux[self.clip], marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
        ax2.plot(self.phase[self.clip], lcmod, marker='.', color='indianred', linestyle="", lw=0, zorder=10)
        ax2.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
    
        divider2 = make_axes_locatable(ax2)
        ax2b = divider2.append_axes("bottom", size="25%", pad=0, sharex=ax2)
        ax2b.errorbar(self.phase[self.clip], lcres,
                     np.sqrt(self.dflux[self.clip]**2 + self.pars['lcerr']**2), 
                     marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
    
        ax2b.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
        ax2b.set_xlabel('Phase (Secondary Eclipse)')
        axb.set_ylim((-thresh*lc_MAD, thresh*lc_MAD))
        ax2b.set_ylim((-thresh*lc_MAD, thresh*lc_MAD))
        
        fig.suptitle('KIC '+str(self.kic)+' LC (only)')
        plt.subplots_adjust(hspace=0)
        if savefig:
            plt.savefig(prefix + suffix+'.png')
        return True

    def plot_sedlc(self, allpars, prefix, suffix='sedlc', savefig=True, 
                         polyorder=2, ooe=True, printpars=True, crowd=None,
                         thresh=10., maskoutliers=True):
        if allpars[17] < 0:
            allpars[17] = np.exp(allpars[17])
        if allpars[4] < 10:
            allpars[4] = np.exp(allpars[4])
        if allpars[16] < 0:
            allpars[16] = np.exp(allpars[16])
        residuals = self.lnlike(allpars[:18], lc_constraints=None, qua=np.unique(self.quarter),
                                  polyorder=polyorder, residual=True)
        lcpars = self.getpars(partype='lc')[:13]
    
#        fig = plt.figure()
#        ax = fig.add_subplot(223)
#        ax.errorbar(self.maglams, self.magsobs, self.emagsobs, fmt='k.')
#        ax.plot(self.maglams, residuals[:len(self.maglams)] * \
#                 np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2) + self.magsobs, 'r.')
#        for ii in range(len(self.maglams)):
#            ax.text(self.maglams[ii], self.magsobs[ii], self.ipname[ii].replace('mag', ''))
#        ax.set_ylabel('Magnitude')
#    
#        divider = make_axes_locatable(ax)
#        ax2 = divider.append_axes("bottom", size=2.0, pad=0, sharex=ax)
#        ax2.errorbar(self.maglams, residuals[:len(self.maglams)] * \
#                 np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2),
#                     np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2), fmt='k.')
#        ax2.set_xlabel('Wavelength (Angstrom)')
#        ax2.set_ylabel('Data - Model')
#        ax2.set_ylim((-0.3, 0.3))
#        plt.setp(ax.get_xticklabels(), visible=False)
#    
#        plt.suptitle('KIC '+str(self.kic)+' SED (simultaneous)')
#        if savefig:
#            plt.savefig(prefix+suffix+'_SED.png')
        if crowd is None:
            crowd = self.crowd
        self.updatephase(lcpars[4], lcpars[3], clip_tol=self.clip_tol)
        lcmod, lcpol = self.lcfit(lcpars, self.jd[self.clip], self.quarter[self.clip],
        				self.flux[self.clip], self.dflux[self.clip],
        				crowd[self.clip], polyorder=polyorder, ooe=ooe)
    
        lcres = self.flux[self.clip] - lcmod*lcpol
        lc_MAD = np.nanpercentile(abs(lcres), 67)

#        if printpars:
#            fig = plt.figure(figsize=(10, 7.5))
#            ax = fig.add_subplot(221)
#            ax.errorbar(self.phase[self.clip], self.flux[self.clip]/lcpol,
#                         self.dflux[self.clip], fmt='k.', ecolor='0.9', zorder=1)
#            ax.plot(self.phase[self.clip], lcmod, 'r.', zorder=10)
#            ax.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
#            ax.set_ylabel('Kepler Flux')
#        
#            divider = make_axes_locatable(ax)
#            axb = divider.append_axes("bottom", size="25%", pad=0, sharex=ax)
#            axb.errorbar(self.phase[self.clip], lcres,
#                         np.sqrt(self.dflux[self.clip]**2 + self.pars['lcerr']**2), fmt='k.', ecolor='0.9')
#        
#            axb.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
#            axb.set_ylabel('Data - Model')
#            axb.set_xlabel('Phase (Primary Eclipse)')
#            #axb.set_yticklabels(axb.yaxis.get_majorticklabels()[1:])
#        
#            ax2 = fig.add_subplot(222)
#            ax2.errorbar(self.phase[self.clip], self.flux[self.clip]/lcpol,
#                         self.dflux[self.clip], fmt='k.', ecolor='0.9', zorder=1)
#            ax2.plot(self.phase[self.clip], lcmod, 'r.', zorder=10)
#            ax2.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
#        
#            divider2 = make_axes_locatable(ax2)
#            ax2b = divider2.append_axes("bottom", size="25%", pad=0, sharex=ax2)
#            ax2b.errorbar(self.phase[self.clip], lcres,
#                         np.sqrt(self.dflux[self.clip]**2 + self.pars['lcerr']**2), fmt='k.', ecolor='0.9')
#        
#            ax2b.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
#            ax2b.set_xlabel('Phase (Secondary Eclipse)')
#        
#            #ax2b.set_yticklabels(ax2b.yaxis.get_majorticklabels()[1:])
#        
#        
#            plt.setp(ax.get_xticklabels(), visible=False)
#            plt.setp(ax2.get_xticklabels(), visible=False)
#    
#            ax = fig.add_subplot(223)
#            ax.errorbar(self.maglams, self.magsobs, self.emagsobs, fmt='k.', zorder=1)
#            ax.plot(self.maglams, residuals[:len(self.maglams)] * \
#                     np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2) + self.magsobs, 'r.', zorder=10)
#            for ii in range(len(self.maglams)):
#                ax.text(self.maglams[ii], self.magsobs[ii], self.ipname[ii].replace('mag', ''))
#            ax.set_ylabel('Magnitude')
#        
#            divider = make_axes_locatable(ax)
#            ax2 = divider.append_axes("bottom", size="30%", pad=0, sharex=ax)
#            ax2.errorbar(self.maglams, residuals[:len(self.maglams)] * \
#                     np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2),
#                         np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2), fmt='k.')
#            ax2.set_xlabel('Wavelength (Angstrom)')
#            ax2.set_ylabel('Data - Model')
#            ax2.set_ylim((-0.3, 0.3))
#            plt.setp(ax.get_xticklabels(), visible=False)
#        
#            ax = fig.add_subplot(224)
#            ax.axis('off')
#            naninds = np.isnan(np.array(self.pars.values(), dtype=np.float))
#            kic_properties = """{}""".format("\n".join("{} = {:.2e}".format(key, val) \
#                                for key, val in zip(np.array(self.pars.keys())[~naninds], 
#                                                    np.array(self.pars.values())[~naninds])))
#            kp1, kp2 = kic_properties.split('lcerr')
#            kp2 = 'lcerr'+kp2
#            ax.text(0, 0.95, kp1, size='smaller', va='top')
#            ax.text(0.5, 0.95, kp2, size='smaller', va='top')
        fig = plt.figure(figsize=(10, 7.5))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[4,3])
        ax = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])

        #divider = make_axes_locatable(ax4)
        #ax4b = divider.append_axes("top", size="20%", pad=0)
        #ax4b.axis('off')
        if printpars:
#            fig = plt.figure(figsize=(10, 7.5))
#            ax = fig.add_subplot(221)
#            ax2 = fig.add_subplot(222)
#
#            ax3 = fig.add_subplot(223)
#
#            ax4 = fig.add_subplot(224)
            #ax4b.text(0.3, 0.8, 'Parameters')
            ax4.axis('off')
            naninds = np.isnan(np.array(self.pars.values(), dtype=np.float))
            kic_properties = """{}""".format("\n".join("{} = {:.2e}".format(key, val) \
                                for key, val in zip(np.array(self.pars.keys())[~naninds], 
                                                    np.array(self.pars.values())[~naninds])))
            kp1, kp2 = kic_properties.split('lcerr')
            kp2 = 'lcerr'+kp2
            ax4.text(0, 0.95, kp1, size='smaller', va='top')
            ax4.text(0.5, 0.95, kp2, size='smaller', va='top')

        else:
#            fig = plt.figure(figsize=(8, 8))
#            ax = plt.subplot2grid((5,6),(0,0), rowspan=3, colspan=3)
#            ax2 = plt.subplot2grid((5,6),(0,3), rowspan=3, colspan=3)
#            ax3 = plt.subplot2grid((5,6),(3,0), rowspan=2, colspan=4)
#            ax4 = plt.subplot2grid((5,6),(3,4), rowspan=2, colspan=2)
#            import matplotlib.gridspec as gridspec
#            gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[3,2])
#            ax = plt.subplot(gs[0])
#            ax2 = plt.subplot(gs[1])
#            ax3 = plt.subplot(gs[2])
#            ax4 = plt.subplot(gs[3])
            #ax4b.text(0.3, 0.8, 'Predicted RV')
            tmod = np.linspace(0, self.pars['period'], 1000)
            phasemod = ((tmod-self.pars['tpe'])%self.pars['period'])/self.pars['period']
            psort = np.argsort(phasemod)
            rv1, rv2 = self.rvfit(np.append(self.getpars(partype='rv')[:-2], [0,0]), tmod)
            ax4.plot(phasemod[psort], rv1[psort]*1e-3, lw=2, color='steelblue', label='Primary')
            ax4.plot(phasemod[psort], rv2[psort]*1e-3, lw=2, color='peru', label='Secondary')
            ax4.set_xlabel('Phase')
            ax4.set_ylabel('RV (km/s)')
            ax4.set_xlim((0, 1))
            ax4.legend(loc=0)
        
        toplot = (self.quality[self.clip]>=0.)
        if maskoutliers:
            toplot = (self.quality[self.clip]<=8)
        ax.errorbar(self.phase[self.clip][toplot], (self.flux[self.clip]/lcpol)[toplot],
                     self.dflux[self.clip][toplot], marker='.', color='0.3', 
                     linestyle="", ecolor='0.9', zorder=1)
        ax.plot(self.phase[self.clip][toplot], lcmod[toplot], marker='.', color='indianred', linestyle="", lw=0, zorder=10)
        ax.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
        ax.set_ylabel('Kepler Flux')
    
        divider = make_axes_locatable(ax)
        axb = divider.append_axes("bottom", size="25%", pad=0, sharex=ax)
        axb.errorbar(self.phase[self.clip][toplot], lcres[toplot],
                     np.sqrt(self.dflux[self.clip][toplot]**2 + self.pars['lcerr']**2), 
                     marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
    
        axb.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
        axb.set_ylabel('Data - Model')
        axb.set_xlabel('Phase (Primary Eclipse)')
        #axb.set_yticklabels(axb.yaxis.get_majorticklabels()[1:])
    
        ax2.errorbar(self.phase[self.clip][toplot], (self.flux[self.clip]/lcpol)[toplot],
                     self.dflux[self.clip][toplot], marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
        ax2.plot(self.phase[self.clip][toplot], lcmod[toplot], marker='.', color='indianred', linestyle="", lw=0, zorder=10)
        ax2.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
    
        divider2 = make_axes_locatable(ax2)
        ax2b = divider2.append_axes("bottom", size="25%", pad=0, sharex=ax2)
        ax2b.errorbar(self.phase[self.clip][toplot], lcres[toplot],
                     np.sqrt(self.dflux[self.clip][toplot]**2 + self.pars['lcerr']**2), 
                     marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
    
        ax2b.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
        ax2b.set_xlabel('Phase (Secondary Eclipse)')
        axb.set_ylim((-thresh*lc_MAD, thresh*lc_MAD))
        ax2b.set_ylim((-thresh*lc_MAD, thresh*lc_MAD))
        #ax2b.set_yticklabels(ax2b.yaxis.get_majorticklabels()[1:])
    
        ax3.errorbar(self.maglams, self.magsobs, self.emagsobs, marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
        ax3.plot(self.maglams, residuals[:len(self.maglams)] * \
                 np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2) + self.magsobs, 
                 marker='.', color='indianred', linestyle="", lw=0, zorder=10)
        for ii in range(len(self.maglams)):
            ax3.text(self.maglams[ii], self.magsobs[ii], self.ipname[ii].replace('mag', ''))
        ax3.set_ylabel('Magnitude')
        divider = make_axes_locatable(ax3)
        ax3b = divider.append_axes("bottom", size="30%", pad=0, sharex=ax3)
        ax3b.errorbar(self.maglams, residuals[:len(self.maglams)] * \
                 np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2),
                     np.sqrt(self.emagsobs**2 + self.pars['isoerr']**2), marker='.', color='0.3', 
                     linestyle="",  ecolor='0.9', zorder=1)
        ax3b.set_xlabel('Wavelength (Angstrom)')
        ax3b.set_ylabel('Data - Model')
        ax3b.set_ylim((-0.3, 0.3))
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)

        plt.suptitle('KIC '+str(self.kic)+' simultaneous')
#        plt.tight_layout()
        if savefig:
            plt.savefig(prefix+suffix+'.png', dpi=300)
        return True

    def _lcfit_brute(self, lcpars, jd, quarter, flux, dflux, crowd,
              polyorder=2, ooe=True):
        # r1, r2, frat derive from m1, m2, z0, t0, dist, E(B-V), scaleheight
        msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, \
            q1, q2, q3, q4 = lcpars

        # LD transformations (Kipping 2013)
        c1 = 2.*np.sqrt(q1)*q2
        c2 = np.sqrt(q1)*(1.-2.*q2)
        c3 = 2.*np.sqrt(q3)*q4
        c4 = np.sqrt(q3)*(1.-2.*q4)
        ldcoeffs1 = np.array([c1, c2])
        ldcoeffs2 = np.array([c3, c4])
            
#        if r2 > r1:
#            r1, r2 = r2, r1
#            m1, m2 = m2, m1
#            frat = 1./frat
        omega=np.arctan2(esinw,ecosw)
        e=np.sqrt(esinw**2+ecosw**2)

        # nip it at the bud.
        if (e>=1.):
            #print "e>=1", e
            return -np.inf, -np.inf
            
        # r1 = rsum/(1.+rrat)
        # r2 = rsum/(1.+1./rrat)
        r1, r2 = self.sumrat_to_12(rsum, rrat)
        a = self.get_a(period, msum)
        inc = self.get_inc(b, r1, a) % TWOPI
        #inc = np.arccos(b*r1/(a/r2au))
        
        if np.isnan(inc):
            #print "inc is nan", inc
            return -np.inf, -np.inf

        self.updatepars(msum=msum, rsum=rsum, rrat=rrat, period=period, tpe=tpe,
                       esinw=esinw, ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4,
                       frat=frat, r1=r1, r2=r2, inc=inc)
        fpe = np.pi/2. - omega
        fse = -np.pi/2. - omega

        # transform time of center of PE to time of periastron (t0)
        # from Eq 9 of Sudarsky et al (2005)
        t0 = tpe - self.sudarsky(fpe, e, period)
        tse = t0 + self.sudarsky(fse, e, period)
        # t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
        #     (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
        #     np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))
        # tse = t0 + (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
        #     (e*np.sin(fse)/(1.+e*np.cos(fse)) - 2.*(1.-e**2)**(-0.5) * \
        #     np.arctan(np.sqrt(1.-e**2) * np.tan((fse)/2.) / (1.+e)))
        self.tpe = tpe
        self.tse = tse
        # if tse<tpe:
        #     tse+=period
        tpad = (np.arange(self.exp))*0.0006811
        tt = jd[:, np.newaxis] + tpad[np.newaxis,:] - 0.0006811*(self.exp-1)/2.
        totmod = tt*0. + 1.
        for ii in range(len(tpad)):
            maf = rsky(e, period, t0, 1e-8, tt[:,ii]) % TWOPI
            r = a*(1.-e**2) / (1.+e*np.cos(maf))
            zcomp = np.sin(omega+maf) * np.sin(inc) 
            z = r*np.sqrt(1.-np.sin(omega+maf)**2*np.sin(inc)**2)
            if not np.all(np.isfinite(z)):
                badz = np.isfinite(z)
                z[~badz] = np.interp(t[~badz], t[badz], z[badz])
                print("badz at ",ii)
            pe = ((r*zcomp>0.)) #& (z <= 1.05*(r1+r2)*r2au)
            se = ((r*zcomp<0.)) #& (z <= 1.05*(r1+r2)*r2au)
            if pe.any():
#                bad = ((z[pe]/(r1*r2au) > 0.5 * abs(rrat-0.5)) * (z[pe]/(r1*r2au) < 1.+rrat)) | \
#                        (rrat>0.5 * (z[pe]/(r1*r2au) >= abs(1.-rrat)) * (z[pe]/(r1*r2au)<rrat))
                res = occultquad(z[pe]/(r1*r2au), c1, c2, rrat, len(z[pe]))
                totmod[:, ii][pe] = (((res - 1.)/(1.+frat) + 1.) - 1.)*crowd[pe] + 1.
#                totmod[:, ii][pe][bad] = np.nan
            if se.any():
#                bad = ((z[se]/(r2*r2au) > 0.5 * abs(1./rrat-0.5)) * (z[se]/(r2*r2au) < 1.+1./rrat)) | \
#                        (1./rrat>0.5 * (z[se]/(r2*r2au) >= abs(1.-1./rrat)) * (z[se]/(r2*r2au)<1./rrat))
                res = occultquad(z[se]/(r2*r2au), c3, c4,1./rrat, len(z[se]))
                totmod[:, ii][se] = (((res - 1.)/(1.+1./frat) + 1.) - 1.)*crowd[se] + 1.
#                totmod[:, ii][se][bad] = np.nan
        return np.nanmean(totmod, axis=-1), jd*0.+1.
        

    def plot_lcrv(self, allpars, prefix, suffix='lcrv', savefig=True, 
                  polyorder=2, ooe=True, thresh=10., maskoutliers=True, 
                  secondary_color='C1',plot_systemic=False):
        if allpars[-1] < 0:
            allpars[-1] = np.exp(allpars[-1])
        if allpars[-3] < 0:
            allpars[-3] = np.exp(allpars[-3])
        
        residuals = self.lnlike_lcrv(allpars, qua=np.unique(self.quarter), polyorder=polyorder,
                                residual=True)
        lcpars = self.getpars(partype='lc')[:13]
        self.updatephase(lcpars[4], lcpars[3], clip_tol=self.clip_tol)
        lcmod, lcpol = self.lcfit(lcpars, self.jd[self.clip], self.quarter[self.clip],
        				self.flux[self.clip], self.dflux[self.clip],
        				self.crowd[self.clip], polyorder=2, ooe=ooe)
    
#        phase = ((self.jd[self.clip]-lcpars[4]) % lcpars[3])/lcpars[3]
#        phase[phase<-np.clip(self.pwidth*3., 0., 0.2)]+=1.
#        phase[phase>np.clip(self.sep+self.swidth*3., self.sep, 1.0)]-=1.
        
        lcres = self.flux[self.clip] - lcmod*lcpol
        lc_MAD = np.nanpercentile(abs(lcres), 67)
        plt.rc('font', size=14, weight='normal')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=14)
        plt.rc('xtick.major', size=6, width=1)
        plt.rc('axes', labelsize=18, titlesize=22)
        plt.rc('figure', titlesize=22)
        from matplotlib.ticker import MaxNLocator

        toplot = (self.quality[self.clip]>=0.)
        if maskoutliers:
            toplot = (self.quality[self.clip]<=8)

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot2grid((2,2),(0,0))
        ax.errorbar(self.phase[self.clip][toplot], (self.flux[self.clip]/lcpol)[toplot],
                     np.sqrt(self.fluxerr[self.clip][toplot]**2+self.pars['lcerr']**2), 
                     fmt='.', color='0.3', linestyle="",
                     ecolor='0.9', zorder=1)
        ax.plot(self.phase[self.clip][toplot], lcmod[toplot], marker=".", color='C0', 
                linestyle="", lw=0, zorder=10, label='Primary Eclipse')
        ax.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
        ax.set_ylim((np.min(lcmod)*0.98, np.max(lcmod)*1.02))
        ax.set_ylabel('Kepler Flux')
        ax.legend(loc='lower right', shadow=False, frameon=False, markerscale=0, fontsize=10)
        
        divider = make_axes_locatable(ax)
        axb = divider.append_axes("bottom", size=0.8, pad=0, sharex=ax)
    
        axb.errorbar(self.phase[self.clip][toplot], lcres[toplot],
                     np.sqrt(self.fluxerr[self.clip]**2 + self.pars['lcerr']**2)[toplot], 
                     fmt='.', color='0.3', linestyle="",
                     ecolor='0.9', zorder=1)
    
        axb.set_xlim((-1.2*self.pwidth, 1.2*self.pwidth))
        axb.set_ylim((-thresh*lc_MAD, thresh*lc_MAD))
#        axb.set_ylabel('Data - Model')
#        axb.set_xlabel('Phase (Primary Eclipse)')
    
        ax2 = plt.subplot2grid((2,2),(0,1))

        ax2.errorbar(self.phase[self.clip][toplot], (self.flux[self.clip]/lcpol)[toplot],
                     np.sqrt(self.fluxerr[self.clip][toplot]**2+self.pars['lcerr']**2), 
                     fmt='.', color='0.3', linestyle="",
                     ecolor='0.9', zorder=1)
        ax2.plot(self.phase[self.clip][toplot], lcmod[toplot], marker=".", color='C0', 
                 linestyle="", lw=0, zorder=10, label='Secondary Eclipse')
        ax2.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
        ax2.set_ylim((np.min(lcmod)*0.98, np.max(lcmod)*1.02))
        ax2.legend(loc='lower right', shadow=False, frameon=False, markerscale=0, fontsize=10)
        
        divider2 = make_axes_locatable(ax2)
        ax2b = divider2.append_axes("bottom", size=0.8, pad=0, sharex=ax2)
        ax2b.errorbar(self.phase[self.clip][toplot], lcres[toplot],
                     np.sqrt(self.fluxerr[self.clip]**2 + self.pars['lcerr']**2)[toplot], 
                     fmt='.', color='0.3', linestyle="",
                     ecolor='0.9', zorder=1)
    
        ax2b.set_xlim((-1.2*self.swidth+self.sep, 1.2*self.swidth+self.sep))
        ax2b.set_ylim((-thresh*lc_MAD, thresh*lc_MAD))
#        ax2b.set_xlabel('Phase (Secondary Eclipse)')
    
        #ax2b.set_yticklabels(ax2b.yaxis.get_majorticklabels()[1:])
    
    
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        rvpars = np.array([self.pars['msum'], self.pars['mrat'], self.pars['period'], self.pars['tpe'],
                           self.pars['esinw'], self.pars['ecosw'], self.pars['inc'], self.pars['k0'], self.pars['rverr']])
    
        rv_fit = self.rvfit(rvpars, self.rv_t)
        ax = plt.subplot2grid((2,2),(1,0), colspan=2)
        rvphase = (self.rv_t - self.pars['tpe'])%self.pars['period']/self.pars['period']
        ax.errorbar(rvphase[~self.bad1], self.rv1_obs[~self.bad1]*1e-3, np.sqrt(self.rv1_err_obs**2+rvpars[-1]**2)[~self.bad1]*1e-3, fmt='C3.', ecolor='gray')
        ax.errorbar(rvphase[~self.bad2], self.rv2_obs[~self.bad2]*1e-3, np.sqrt(self.rv2_err_obs**2+rvpars[-1]**2)[~self.bad2]*1e-3, fmt='.',color=secondary_color, ecolor='gray')
        if plot_systemic:
            ax.axhline(rvpars[-2]*1e-3, ls='--', color='0.3', alpha=0.8, lw=1)
        rvt = np.linspace(0, 1, 100)*self.pars['period']+self.pars['tpe']
        rvmod = self.rvfit(rvpars, rvt)
        ax.plot(np.linspace(0, 1, 100), rvmod[0]*1e-3, 'C3-', lw=2)
        ax.plot(np.linspace(0, 1, 100), rvmod[1]*1e-3, 'C1-', lw=2)

        ax.set_ylabel('RV (km/s)')
        ax.set_xlim((0,1))
        locator=MaxNLocator(prune='both',nbins=5)
        ax.yaxis.set_major_locator(locator)
        
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size=0.8, pad=0, sharex=ax)
        ax2.errorbar(rvphase[~self.bad1], (self.rv1_obs-rv_fit[0])[~self.bad1]*1e-3, np.sqrt(self.rv1_err_obs**2+rvpars[-1]**2)[~self.bad1]*1e-3, fmt='C3.')
        ax2.errorbar(rvphase[~self.bad2], (self.rv2_obs-rv_fit[1])[~self.bad2]*1e-3, np.sqrt(self.rv2_err_obs**2+rvpars[-1]**2)[~self.bad2]*1e-3, fmt='.', color=secondary_color)
        ax2.set_ylim((-1.2*np.nanmax(abs((self.rv1_obs-rv_fit[0])[~self.bad1]*1e-3)), 1.2*np.nanmax(abs((self.rv1_obs-rv_fit[0])[~self.bad1]*1e-3))))
        ax2.set_xlim((0,1))
        
        ax2.set_xlabel('Phase')
#        ax2.set_ylabel('Data - Model')
        #ax2.set_yticklabels(ax2.yaxis.get_majorticklabels()[:-1])
        plt.setp(ax.get_xticklabels(), visible=False)
        #plt.setp(ax2.get_yticklabels()[-1], visible=False)
    
        plt.suptitle('KIC '+str(self.kic))
        if savefig:
            plt.savefig(prefix+suffix+'_simu.png', dpi=300)
    
        return True


    def plot_rv(self, rvpars, prefix, suffix='rv', savefig=True):  
        rv_fit = self.rvfit(rvpars, self.rv_t)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rvphase = (self.rv_t - self.pars['tpe'])%self.pars['period']/self.pars['period']
        ax.errorbar(rvphase[~self.bad1], self.rv1_obs[~self.bad1], self.rv1_err_obs[~self.bad1], fmt='b*')
        ax.errorbar(rvphase[~self.bad2], self.rv2_obs[~self.bad2], self.rv2_err_obs[~self.bad2], fmt='r*')
        rvt = np.linspace(0, 1, 100)*self.pars['period']+self.pars['tpe']
        rvmod = self.rvfit(rvpars, rvt)
        ax.plot(np.linspace(0, 1, 100), rvmod[0], 'b-')
        ax.plot(np.linspace(0, 1, 100), rvmod[1], 'r-')
        ax.set_ylabel('RV [m/s]')
        #ax.set_yticklabels(ax.yaxis.get_majorticklabels()[:-1])
    
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size=2.0, pad=0, sharex=ax)
        ax2.errorbar(rvphase[~self.bad1], (self.rv1_obs-rv_fit[0])[~self.bad1], np.sqrt(self.rv1_err_obs**2+rvpars[-1]**2)[~self.bad1], fmt='b.')
        ax2.errorbar(rvphase[~self.bad2], (self.rv2_obs-rv_fit[1])[~self.bad2], np.sqrt(self.rv2_err_obs**2+rvpars[-1]**2)[~self.bad2], fmt='r.')
    
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Data - Model')
        #ax2.set_yticklabels(ax2.yaxis.get_majorticklabels()[:-1])
        plt.setp(ax.get_xticklabels(), visible=False)
        #plt.setp(ax2.get_yticklabels()[-1], visible=False)
    
        plt.suptitle('KIC '+str(self.kic)+' RV (simultaneous)')
        if savefig:
            plt.savefig(prefix+suffix+'_RV.png')
    
        return True
        

    @staticmethod
    def get_pars2vals(fisopars, partype='lc', qcrow=None):
        try:
            parnames = parnames_dict[partype][:]
        except:
            print("You entered: {0}. Partype options are 'lc', 'sed', 'rv', 'lcsed', 'lcrv'. Try again.".format(partype))
            return
        if qcrow is not None:
            parnames += ['cr' + str(ii) for ii in qcrow]
        parvals = np.zeros(len(parnames))
        novalue = (len(fisopars) == len(parnames))
        if isinstance(fisopars, lmfit.parameter.Parameters):
            for j in range(len(parnames)):
                parvals[j] = fisopars[parnames[j]].value
        elif isinstance(fisopars, dict):
            for j in range(len(parnames)):
                parvals[j] = fisopars[parnames[j]]
        else:
            for j in range(len(parnames)):
                parvals[j] = fisopars[j]*novalue
        return parvals
        
    @staticmethod
    def broadcast_crowd(qrt, cro):
        """Takes in crowding parameters for each quarter and broadcasts to all cadences

        Parameters
        ----------
        qrt: array of length equal to time array
            array that specifies the quarter # of each cadence
        cro: array of length N quarters
            array that specifies the crowing values for each unique quarter

        Returns
        -------
        cro_casted: array of length equal to time array
            output array with crowding values broadcasted to each cadence
        """
        cro = cro.ravel()
        good = np.where(np.diff(qrt)>0)[0] + 1
        good = np.insert(good, 0, 0)
        good = np.insert(good, len(good), len(qrt))
        cro_casted = qrt*1.0
        for ii in range(len(good)-1):
            cro_casted[good[ii]:good[ii+1]] = cro[ii]
        return cro_casted

        
    @staticmethod
    def get_P(a, msum):
        return np.sqrt(a**3 / msum) * d2y

    @staticmethod
    def transform_ew(esinw, ecosw):
        e = np.sqrt(esinw**2 + ecosw**2)
        omega = np.arctan2(esinw, ecosw)
        return e, omega

    @staticmethod
    def get_a(period, msum):
        """Computes semi-major axis given period and total mass

        Parameters
        ----------
        period: Period of system in days
        msum: sum of masses in Msun

        Returns
        -------
        a: semi-major axis in AU
        """
        return ((period/d2y)**2 * (msum))**(1./3.)

    @staticmethod
    def sumrat_to_12(xsum, xrat):
        """Transforms sum and ratios of 2 quantities into individual components

        Parameters
        ----------
        xsum: x1 + x2 value
        xrat: x2/x1 value

        Returns
        -------
        x1, x2
        """
        x1 = xsum / (1+xrat)
        x2 = xsum / (1 + 1./xrat)
        return x1, x2

    @staticmethod
    def sudarsky(theta, e, period):
        """Computes time as a function of mean anomaly+omega given period, eccentricity

        Parameters
        ----------
        theta : float array or scalar
                value of omega + maf (arg. periapse + mean anomaly) in radians
        e : scalar
            eccentricity of binary
        period : scalar
            period of binary

        Returns
        -------
        t : float array or scalar
            time given theta, e, P
        """
        tt = (-np.sqrt(1. - e ** 2) * period / TWOPI) * \
             (e * np.sin(theta) / (1. + e * np.cos(theta)) - 2. * (1. - e ** 2) ** (-0.5) *
              np.arctan(np.sqrt(1. - e ** 2) * np.tan((theta) / 2.) / (1. + e)))
        return tt

    @staticmethod
    def get_inc(b, r1, a):
        """Transforms impact parameter into inclination

        Parameters
        ----------
        b: impact parameter
        r1: radius of primary star (in Rsun)
        a: semi-major axis in AU

        Returns
        -------
        inc: inclination of system"""
        return np.arccos(b*r1 / (a/r2au))

    @staticmethod
    def find_closest(base, target):
        """Returns indices of closest match between a base and target arrays
        
        Parameters
        ----------
        base : array
            base list of items to compare
        target : array
            list of target items to which to compare to the base array
            
        Returns
        -------
        indx : int
            index of matched item
        
        Examples
        --------
        >>> a = np.arange(10)
        >>> b = 4.3
        >>> print 
        """
        
        #A must be sorted
        idx = base.searchsorted(target)
        idx = np.clip(idx, 1, len(base)-1)
        left = base[idx-1]
        right = base[idx]
        idx -= target - left < right - target
        return idx


    @staticmethod
    def feh2z(feh):
        """Converts [Fe/H] value to Z metallicity fraction (zsun * 10**feh)

        Parameters
        ----------
        feh : array or scalar
            [Fe/H] value of object

        Returns
        -------
        z : array or scalar
            Z metallicity"""
        return 0.01524 * 10**feh

    @staticmethod
    def z2feh(z):
        """Converts Z metallicity value to [Fe/H] (log10(z/zsun))

        Parameters
        ----------
        z : array or scalar

        Returns
        -------
        feh : array or scalar
        """
        return np.log10(z/0.01524)
        
    @staticmethod
    def kipping_q2u(q1, q2):
        """Transforms parameterized limb darkening parameters to traditional 
        quadratic LD via Kipping 2010
        
        Parameters
        ----------
        q1 : scalar (0 to 1)
        q2 : scalar (0 to 1)
        
        Returns
        -------
        c1, c2 : scalars (quadratic limb darkening parameters)
        """
        c1 = 2.*np.sqrt(q1)*q2
        c2 = np.sqrt(q1)*(1.-2.*q2)
        return c1, c2
    
    @staticmethod
    def kipping_u2q(u1, u2):
        """Transforms quadratic limb darkening to parameterized parameters 
        via Kipping 2010
        
        Parameters
        ----------
        u1 : scalar
        u2 : scalar
        
        Returns
        -------
        q1, q2 : scalars (parameterized LD [0, 1])
        """
        q1 = (u1+u2)**2
        q2 = 0.5 * u1 / (u1+u2)
        return q1, q2
    

###############################################################################
################# a whole bunch of defunct, old functions #####################
###############################################################################
    def loadiso_defunct(self, isodir = 'data/'):
        """Function which loads Padova isochrone data
        
        Parameters
        ----------
        isodir : str
            directory which stores isochrone file
        
        Returns
        -------
        True : boolean
            if isodata loading is successful, stored in keblat.iso
        
        Examples
        --------
        >>> keblat.loadiso()
        Isodata.cat already exists; loaded.
        True
        """
        
        self.isodict = {'z': 0, 'logt': 1, 'mini': 2, 'mact': 3, 'logl': 4, 
                        'logte': 5, 'logg': 6, 'mbol': 7, 'mkep': 8, 'gmag': 9, 
                        'rmag': 10, 'imag': 11, 'zmag': 12, 'md51': 13, 
                        'Jmag': 14, 'Hmag': 15, 'Kmag': 16, 'Umag': 17, 
                        'Bmag': 18, 'Vmag': 19, 'w1': 20, 'w2': 21}
        fmt = ['%.4f', '%.2f', '%.8f', '%.4f', '%.4f', '%.4f', '%.4f', '%.3f', 
               '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', 
               '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f']
        isodatafile = isodir + 'isodata.dat'
        if os.path.isfile(isodatafile):
            iso = np.loadtxt(isodatafile, delimiter=',')
            self.iso = iso 
            print "Isodata.cat already exists; loaded."

        else:            
            sdssfiles = isodir + 'pad*.sdss.dat'
            noaofiles = isodir + 'pad*.noao.dat'
            wisefiles = isodir + 'pad*.wise.dat'
            
            sdss = glob(sdssfiles)
            noao = glob(noaofiles)
            wise = glob(wisefiles)
            
            sdss = np.sort(sdss)
            nsdss = len(sdss)
            noao = np.sort(noao)
            wise = np.sort(wise)
            
            sdss1 = np.loadtxt(sdss[0])
            noao1 = np.loadtxt(noao[0])
            wise1 = np.loadtxt(wise[0])
            iso = np.hstack((sdss1[:, :-2], noao1[:, [9,10,12]], 
                             wise1[:, 18:-4]))
            # things will be in mbol, mkep, g, r, i, z, 
            # d51, J, H, Ks, U, B, V, w1, w2
            for ii in range(1, nsdss):
                sdss1 = np.loadtxt(sdss[ii])
                noao1 = np.loadtxt(noao[ii])
                wise1 = np.loadtxt(wise[ii])
                iso1 = np.hstack((sdss1[:, :-2], 
                                  noao1[:, [9,10,12]], wise1[:, 18:-4]))
                iso = np.concatenate((iso, iso1))
            import operator
            self.iso = np.array(sorted(iso, key=operator.itemgetter(0, 1, 3)))
            np.savetxt(isodatafile, self.iso, delimiter=',', fmt=fmt)
            print "Isodata.dat created."

        self.zvals = np.unique(self.iso[:, self.isodict['z']])
        self.tvals = np.unique(self.iso[:, self.isodict['logt']])
        self.maxz, self.minz = np.max(self.zvals), np.min(self.zvals)
        self.maxt, self.mint = np.max(self.tvals), np.min(self.tvals)  
        return True

    # Computes a template eclipse light curve with Mandel & Agol (2002) algorithm
    def lctemplate_slow(self, lcpars, period, omega, e, a, inc, bgr, ldcoeffs,
                   rrat, tc, t0, cadence, exp, pe=True):
        """Computes a template Mandel & Agol (2002) eclipse lightcurve with
        correction for long-cadence binning (Kipping 2013)

        Parameters
        ----------
        period : float
            period of EB
        omega : float
            argument of periastron
        e : float
            eccentricity
        a : float
            semi-major axis in AU
        inc : float
            inclination; 90 = edge on
        bgr : float
            radius of star being eclipsed in solar radius
        ldcoeffs: float array
            limb darkening coefficients; tuple if quadratic,
            4 elements if quartic non-linear law
        rrat : float
            ratio of radii (eclipsing/eclipsed)
        tc : float
            time of center of eclipse (either PE or SE)
        t0 : float
            time of periastron passage
        cadence : float
            cadence (in days)
        exp : int
            number of Kepler exposures to bin (LC = 30, SC = 1)
        pe : boolean
            if True, template for PE

        Returns
        -------
        tmean : float array
            array of times for PE (or SE)
        resmean : float array
            array of flux values for PE (or SE)

        """

        if pe:
            half0 = (self.pwidth*period/2.)*self.clip_tol
        else:
            half0 = (self.swidth*period/2.)*self.clip_tol

#        half0 = period/2.

        t = np.linspace(-half0, +half0, int(2*half0/0.0006811) + 1)
        t += tc
        maf = rsky(e, period, t0, 1e-8, t)
        r = a*(1.-e**2) / (1.+e*np.cos(maf))
        z = r*np.sqrt(1.-np.sin(omega+maf)**2*np.sin(inc)**2)
        finite_z = np.isfinite(z)
        if np.sum(~finite_z)>0:
            print maf, e, a, inc, z
            if np.sum(~finite_z) < 0.05*len(z):
                z[~finite_z] = np.interp(t[~finite_z], t[finite_z], z[finite_z])
            else:
                exit
        if self.ldtype == 0:
            # print "non-linear case:", ldcoeffs
            res = self.occultnltheta(z/(bgr*r2au), rrat, ldcoeffs)
        else:
            # print "Quad case:", ldcoeffs
            res = occultquad(z/(bgr*r2au), ldcoeffs[0], ldcoeffs[1], rrat, len(z))

        bad = (res<-1e-4) | (res-1.>1e-4)
        if np.sum(bad)>0:
            cz = z[bad]/(bgr*r2au)
            interp_res = np.interp(t[bad], t[~bad], res[~bad])
            errfile = open(self.erfname, "a")
            errfile.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(min(z)/(bgr*r2au), max(z)/(bgr*r2au), ldcoeffs[0], ldcoeffs[1], rrat, " ".join([str(ii) for ii in cz]), " ".join([str(ii) for ii in res[bad]]), " ".join([str(ii) for ii in interp_res])))
            errfile.close()
            res[bad] = interp_res
        if cadence < 0.02:
            return t, res

        tt = t[:, np.newaxis] * np.ones(exp)
        rres = res[:, np.newaxis] * np.ones(exp)
        tt[:, 0] = t
        rres[:, 0] = res
        for ii in range(1, exp):
            tt[:-ii, ii] = t[ii:]
            rres[:-ii, ii] = res[ii:]
        tmean = np.mean(tt[:-exp+1, :], axis=1)
        resmean = np.mean(rres[:-exp+1, :], axis=1)
        return tmean, resmean

    # LIGHT CURVE MODEL
    def lcfit_slow(self, lcpars, jd, phase, flux, dflux, crowd, 
              polyorder=2):
        """Computes light curve model
        
        Parameters
        ----------
        lcpars : float array
            parameters for LC fitting: 
            msum, rsum, rratio, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4
        jd : float array
            time array
        phase : float array
            corresponding phase
        flux : float array
            observed flux
        dflux : float array
            flux error
        crowd : float array
            array of crowding values (additional flux)
        polyorder : int
            order of polynomial to detrend lightcurve
        
        Returns
        -------
        totmod : float array
            array of model fluxes
        totpol : float array
            array of polynomials for detrending
        """
        # r1, r2, frat derive from m1, m2, z0, t0, dist, E(B-V), scaleheight
        msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, \
            q1, q2, q3, q4 = lcpars       
#        self.updatepars(m1=m1, m2=m2, period=period, tpe=tpe, esinw=esinw, 
#                    ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4)
        # LD transformations (Kipping 2013)
        c1 = 2.*np.sqrt(q1)*q2
        c2 = np.sqrt(q1)*(1.-2.*q2)
        c3 = 2.*np.sqrt(q3)*q4
        c4 = np.sqrt(q3)*(1.-2.*q4)
        ldcoeffs1 = np.array([c1, c2])
        ldcoeffs2 = np.array([c3, c4])
            
#        if r2 > r1:
#            r1, r2 = r2, r1
#            m1, m2 = m2, m1
#            frat = 1./frat
        omega=np.arctan2(esinw,ecosw)
        e=np.sqrt(esinw**2+ecosw**2)

        # nip it at the bud.
        if (e>=1.):
            #print "e>=1", e
            return -np.inf, -np.inf
            


        r1 = rsum/(1.+rrat)
        r2 = rsum/(1.+1./rrat)
        a = ((period/d2y)**2 * (msum))**(1./3.)
        inc = np.arccos(b*r1/(a/r2au))
        
        if np.isnan(inc):
            #print "inc is nan", inc
            return -np.inf, -np.inf
        
        fpe = np.pi/2. - omega
        fse = -np.pi/2. - omega

        # transform time of center of PE to time of periastron (t0)
        # from Eq 9 of Sudarsky et al (2005)
        t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
            (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
            np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))
        tse = t0 + (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
            (e*np.sin(fse)/(1.+e*np.cos(fse)) - 2.*(1.-e**2)**(-0.5) * \
            np.arctan(np.sqrt(1.-e**2) * np.tan((fse)/2.) / (1.+e)))

        # if tse<tpe:
        #     tse+=period
            
        tempt1, tempres1 = self.lctemplate(lcpars, period, omega, e, a, inc, r1,
                                           ldcoeffs1, r2/r1, tpe, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=True)
        tempt2, tempres2 = self.lctemplate(lcpars, period, omega, e, a, inc, r2,
                                           ldcoeffs2, r1/r2, tse, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=False)
        if np.any(np.isinf(tempt1)) or np.any(np.isinf(tempt2)):
            return -np.inf, -np.inf

        tempt1 = tempt1 % period
        tempt2 = tempt2 % period
        tempres1 = (tempres1 - 1.)/(1. + frat) + 1.
        tempres2 = (tempres2 - 1.)/(1. + 1./frat) + 1.

        sorting1 = np.argsort(tempt1)
        sorting2 = np.argsort(tempt2)

        tempres1 = tempres1[sorting1]
        tempt1 = tempt1[sorting1]
        tempres2 = tempres2[sorting2]
        tempt2 = tempt2[sorting2]

        #not including crowdsap term.
        #tempres1 = (tempres1 + frat) / (1.+frat)
        #tempres2 = (tempres2 * frat + 1.) / (1. + frat)
        totpol, totmod = np.ones(len(jd)), np.ones(len(jd))

        if polyorder>0:
    
            # mask out continuum data
            #clip = ((abs(phase)<1.5*self.pwidth) | (abs(phase-self.sep) < 1.5*self.swidth))
            clip = (jd>0)
    
            chunk = np.array(np.where(np.diff(jd[clip]) > self.pwidth*period))[0]
            #put in dummy first and last element # placeholders
            
            chunk = np.append(chunk, len(jd[clip])-2).flatten()
            _, chunk3 = np.unique(np.searchsorted(jd[clip][chunk], jd), return_index=True)
            chunk=chunk3
            chunk[-1]+=1
    #        plt.plot(self.jd, self.flux, 'ro', self.jd[clip], self.flux[clip], 'go', self.jd[chunk[:-1]],self.flux[chunk[:-1]], 'mo')
    #        plt.show()
            for i in range(len(chunk)-1):
                #print i, chunk[i], chunk[i+1], self.jd[chunk[i]:chunk[i+1]]
                t = jd[chunk[i]:chunk[i+1]]
                f = flux[chunk[i]:chunk[i+1]]
                ef = dflux[chunk[i]:chunk[i+1]]
                crow = crowd[chunk[i]:chunk[i+1]]
                maf = rsky(e, period, t0, 1e-8, t)
                npts=len(maf)
                #use this version for full lightcurve treatment...
                r = a*(1.-e**2) / (1.+e*np.cos(maf))
                zcomp = np.sin(omega+maf) * np.sin(inc) 
                #z = r*np.sqrt(1.-zcomp**2)
                pe = ((r*zcomp>0.))# & (z <= 1.05*(r1+r2)*r2au))
                se = ((r*zcomp<0.))# & (z <= 1.05*(r1+r2)*r2au))
                model = np.ones(npts)
    #            sse = (((maf+omega) % (TWOPI))>np.pi)
    #            ppe = (((maf+omega) % (TWOPI))<=np.pi)
                
    #            plt.plot(t, f, 'ro', t[pe], f[pe], 'go', t[se], f[se], 'bo')
    #            plt.title(str(i))
    #            plt.show()
                if pe.any():
#                    shift = period * np.round((np.mean(t[pe]) - tpe)/period)
                    model[pe] = np.interp(t[pe]%period, tempt1, tempres1)
                    model[pe] = (model[pe] - 1.) * crow[pe] + 1.
    #                print "PE: mean(t[pe]), tpe, (mean(t[pe])-tpe)/period, round ver"
    #                print np.mean(t[pe]), tpe, (np.mean(t[pe]) - tpe)/period, np.round((np.mean(t) - tpe)/period), len(t[pe]), len(f[pe]), len(tempt1), len(tempres1), len(model[pe])
    #                plt.plot(t[pe]-shift, f[pe], 'ro', tempt1, (tempres1-1.)*crow[pe][0] + 1., 'bo', t[pe]-shift, model[pe], 'go')
    #                plt.title('pe')
    #                plt.show()
    #                plt.close('all')
                if se.any():
#                    shift = period * np.round((np.mean(t[se]) - tse)/period)
                    model[se] = np.interp(t[se]%period, tempt2, tempres2)
                    model[se] = (model[se] - 1.) * crow[se] + 1.
    
    #                print "SE"
    #                print np.mean(t[se]), tse, (np.mean(t[se]) - tse)/period, np.round((np.mean(t[se]) - tse)/period)
    #                plt.plot(t[se]-shift, f[se], 'ro', tempt2, (tempres2-1.)*crow[se][0] + 1., 'bo', t[se]-shift, model[se], 'go')
    #                plt.title('se')
    #                plt.show()
    #                plt.close('all')
    #            else:
    #                print "This data bundle does not belong to SE or PE"
                # marginalization (2nd order polynomial fit to residuals)

                bad = (model<1)
                tt = t[~bad]
                mmodel = model[~bad]
                ff = f[~bad]
                eef = ef[~bad]
                nnpts = len(ff)
                tnew = tt - np.mean(tt)
                #if len(t[~bad]) < 1:
                    #print "Npts ooe = ",len(t[~bad])
    
                # Bk = sum over i (D_i/M_i)(tdiff_i)^k / (sigma_i/M_i)^2
                # matrix 3 rows x npts columns since quadratic polynomial 
                # fit requires 3 coeffs
                if bad[0] or bad[-1]:
                    poly_remember = polyorder
                    polyorder=1
                #number of 'i' data or model points; polynomial order
                order_pow = np.arange(polyorder+1)
                t_pow = tnew[:,np.newaxis]**order_pow
                Bk = np.ones(shape=(polyorder+1,nnpts))*((ff/mmodel)/(eef/mmodel)**2)
                Bk*=t_pow.T
                #sum along 'i' (or along each row)
                Bksum = np.sum(Bk,axis=1)
                #Mjk = sum over i (tdiff_i)^j * (tdiff_i)^k / (sigma_i/M_i)^2
                #construct 3 rows x npts columns 
                Mj = np.ones(shape=(polyorder+1,nnpts))/(eef/mmodel)**2
                Mj*=t_pow.T
                #transform from 2D (j rows x i columns) to 3D (k x j x i)
                t_pow_3d = tnew[:,np.newaxis,np.newaxis]**order_pow
                Mjk = t_pow_3d.T * Mj[np.newaxis,:,:]
                #now sum along 'i' 
                Mjksum = np.sum(Mjk,axis=2)
                #do matrix inversion solver thing to get polynomial coeffs
                try:
                    Aj = np.linalg.lstsq(Mjksum,Bksum)[0]
                    pol = np.polyval(Aj[::-1],t-np.mean(t))
                except: 
                    pol = np.ones(npts)
                #Aj = np.dot(np.linalg.pinv(Mjksum), Bksum)
    #                plt.plot(t, f, 'ro', t, model*pol, 'go')
    #                plt.plot(t, pol, 'ms', tt, np.polyval(Aj[::-1],tnew), 'cs')
    #                plt.show()
                if bad[0] or bad[-1]:
                    polyorder = poly_remember
                totmod[chunk[i]:chunk[i+1]] = model
                totpol[chunk[i]:chunk[i+1]] = pol

        else:
            maf = rsky(e, period, t0, 1e-8, jd)
            r = a*(1.-e**2) / (1.+e*np.cos(maf))
            zcomp = np.sin(omega+maf) * np.sin(inc) 
            #z = r*np.sqrt(1.-zcomp**2)
            pe = ((r*zcomp>0.)) #& (z <= 1.05*(r1+r2)*r2au))
            se = ((r*zcomp<0.)) #& (z <= 1.05*(r1+r2)*r2au))
            tt = jd % period
            if pe.any():
                totmod[pe] = np.interp(tt[pe], tempt1, tempres1)
                totmod[pe] = (totmod[pe] - 1.) * crowd[pe] + 1.
            if se.any():
                totmod[se] = np.interp(tt[se], tempt2, tempres2)
                totmod[se] = (totmod[se] - 1.) * crowd[se] + 1.
        #     if np.sum(totmod[se]-1.) == 0.:
        #         return np.ones_like(totmod), totpol
        # if np.sum(totmod-1.) == 0.:
        #     return totmod, totmod
        return totmod, totpol

    def lcfit_slow2(self, lcpars, jd, phase, flux, dflux, crowd, 
              polyorder=2):
        """Computes light curve model
        
        Parameters
        ----------
        lcpars : float array
            parameters for LC fitting: 
            msum, rsum, rratio, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4
        jd : float array
            time array
        phase : float array
            corresponding phase
        flux : float array
            observed flux
        dflux : float array
            flux error
        crowd : float array
            array of crowding values (additional flux)
        polyorder : int
            order of polynomial to detrend lightcurve
        
        Returns
        -------
        totmod : float array
            array of model fluxes
        totpol : float array
            array of polynomials for detrending
        """
        # r1, r2, frat derive from m1, m2, z0, t0, dist, E(B-V), scaleheight
        msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, \
            q1, q2, q3, q4 = lcpars       
#        self.updatepars(m1=m1, m2=m2, period=period, tpe=tpe, esinw=esinw, 
#                    ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4)
        # LD transformations (Kipping 2013)
        c1 = 2.*np.sqrt(q1)*q2
        c2 = np.sqrt(q1)*(1.-2.*q2)
        c3 = 2.*np.sqrt(q3)*q4
        c4 = np.sqrt(q3)*(1.-2.*q4)
        ldcoeffs1 = np.array([c1, c2])
        ldcoeffs2 = np.array([c3, c4])
            
#        if r2 > r1:
#            r1, r2 = r2, r1
#            m1, m2 = m2, m1
#            frat = 1./frat
        omega=np.arctan2(esinw,ecosw)
        e=np.sqrt(esinw**2+ecosw**2)

        # nip it at the bud.
        if (e>=1.):
            #print "e>=1", e
            return -np.inf, -np.inf
            


        r1 = rsum/(1.+rrat)
        r2 = rsum/(1.+1./rrat)
        a = ((period/d2y)**2 * (msum))**(1./3.)
        inc = np.arccos(b*r1/(a/r2au))
        
        if np.isnan(inc):
            #print "inc is nan", inc
            return -np.inf, -np.inf
        
        fpe = np.pi/2. - omega
        fse = -np.pi/2. - omega

        # transform time of center of PE to time of periastron (t0)
        # from Eq 9 of Sudarsky et al (2005)
        t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
            (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
            np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))
        tse = t0 + (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
            (e*np.sin(fse)/(1.+e*np.cos(fse)) - 2.*(1.-e**2)**(-0.5) * \
            np.arctan(np.sqrt(1.-e**2) * np.tan((fse)/2.) / (1.+e)))

        # if tse<tpe:
        #     tse+=period
            
        tempt1, tempres1 = self.lctemplate(lcpars, period, omega, e, a, inc, r1,
                                           ldcoeffs1, r2/r1, tpe, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=True)
        tempt2, tempres2 = self.lctemplate(lcpars, period, omega, e, a, inc, r2,
                                           ldcoeffs2, r1/r2, tse, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=False)

        tempt1 = tempt1 % period
        tempt2 = tempt2 % period
        tempres1 = (tempres1 - 1.)/(1. + frat) + 1.
        tempres2 = (tempres2 - 1.)/(1. + 1./frat) + 1.

        sorting1 = np.argsort(tempt1)
        sorting2 = np.argsort(tempt2)

        tempres1 = tempres1[sorting1]
        tempt1 = tempt1[sorting1]
        tempres2 = tempres2[sorting2]
        tempt2 = tempt2[sorting2]

        #not including crowdsap term.
        #tempres1 = (tempres1 + frat) / (1.+frat)
        #tempres2 = (tempres2 * frat + 1.) / (1. + frat)
        totpol, totmod = np.ones(len(jd)), np.ones(len(jd))

        if polyorder>0:
    
            # mask out continuum data
            #clip = ((abs(phase)<1.5*self.pwidth) | (abs(phase-self.sep) < 1.5*self.swidth))
            clip = (jd>0)
    
            chunk = np.array(np.where(np.diff(jd[clip]) > self.pwidth*period))[0]
            #put in dummy first and last element # placeholders
            
            chunk = np.append(chunk, len(jd[clip])-2).flatten()
            _, chunk3 = np.unique(np.searchsorted(jd[clip][chunk], jd), return_index=True)
            chunk=chunk3
            chunk[-1]+=1
    #        plt.plot(self.jd, self.flux, 'ro', self.jd[clip], self.flux[clip], 'go', self.jd[chunk[:-1]],self.flux[chunk[:-1]], 'mo')
    #        plt.show()
            for i in range(len(chunk)-1):
                #print i, chunk[i], chunk[i+1], self.jd[chunk[i]:chunk[i+1]]
                t = jd[chunk[i]:chunk[i+1]]
                f = flux[chunk[i]:chunk[i+1]]
                ef = dflux[chunk[i]:chunk[i+1]]
                crow = crowd[chunk[i]:chunk[i+1]]
                maf = rsky(e, period, t0, 1e-8, t)
                npts=len(maf)
                #use this version for full lightcurve treatment...
                r = a*(1.-e**2) / (1.+e*np.cos(maf))
                zcomp = np.sin(omega+maf) * np.sin(inc) 
                #z = r*np.sqrt(1.-zcomp**2)
                pe = ((r*zcomp>0.))# & (z <= 1.05*(r1+r2)*r2au))
                se = ((r*zcomp<0.))# & (z <= 1.05*(r1+r2)*r2au))
                model = np.ones(npts)
    #            sse = (((maf+omega) % (TWOPI))>np.pi)
    #            ppe = (((maf+omega) % (TWOPI))<=np.pi)
                
    #            plt.plot(t, f, 'ro', t[pe], f[pe], 'go', t[se], f[se], 'bo')
    #            plt.title(str(i))
    #            plt.show()
                if pe.any():
#                    shift = period * np.round((np.mean(t[pe]) - tpe)/period)
                    model[pe] = np.interp(t[pe]%period, tempt1, tempres1)
                    model[pe] = (model[pe] - 1.) * crow[pe] + 1.
    #                print "PE: mean(t[pe]), tpe, (mean(t[pe])-tpe)/period, round ver"
    #                print np.mean(t[pe]), tpe, (np.mean(t[pe]) - tpe)/period, np.round((np.mean(t) - tpe)/period), len(t[pe]), len(f[pe]), len(tempt1), len(tempres1), len(model[pe])
    #                plt.plot(t[pe]-shift, f[pe], 'ro', tempt1, (tempres1-1.)*crow[pe][0] + 1., 'bo', t[pe]-shift, model[pe], 'go')
    #                plt.title('pe')
    #                plt.show()
    #                plt.close('all')
                if se.any():
#                    shift = period * np.round((np.mean(t[se]) - tse)/period)
                    model[se] = np.interp(t[se]%period, tempt2, tempres2)
                    model[se] = (model[se] - 1.) * crow[se] + 1.
    
    #                print "SE"
    #                print np.mean(t[se]), tse, (np.mean(t[se]) - tse)/period, np.round((np.mean(t[se]) - tse)/period)
    #                plt.plot(t[se]-shift, f[se], 'ro', tempt2, (tempres2-1.)*crow[se][0] + 1., 'bo', t[se]-shift, model[se], 'go')
    #                plt.title('se')
    #                plt.show()
    #                plt.close('all')
    #            else:
    #                print "This data bundle does not belong to SE or PE"
                # marginalization (2nd order polynomial fit to residuals)

                bad = (model<1)
                tt = t[~bad]
                mmodel = model[~bad]
                ff = f[~bad]
                eef = ef[~bad]
                nnpts = len(ff)
                tnew = tt - np.mean(tt)
                #if len(t[~bad]) < 1:
                    #print "Npts ooe = ",len(t[~bad])
    
                # Bk = sum over i (D_i/M_i)(tdiff_i)^k / (sigma_i/M_i)^2
                # matrix 3 rows x npts columns since quadratic polynomial 
                # fit requires 3 coeffs
                if bad[0] or bad[-1]:
                    polyorder=1
                #number of 'i' data or model points; polynomial order
                order_pow = np.arange(polyorder+1)
                t_pow = tnew[:,np.newaxis]**order_pow
                Bk = np.ones(shape=(polyorder+1,nnpts))*((ff-mmodel)/(eef)**2)
                Bk*=t_pow.T
                #sum along 'i' (or along each row)
                Bksum = np.sum(Bk,axis=1)
                #Mjk = sum over i (tdiff_i)^j * (tdiff_i)^k / (sigma_i/M_i)^2
                #construct 3 rows x npts columns 
                Mj = np.ones(shape=(polyorder+1,nnpts))/(eef)**2
                Mj*=t_pow.T
                #transform from 2D (j rows x i columns) to 3D (k x j x i)
                t_pow_3d = tnew[:,np.newaxis,np.newaxis]**order_pow
                Mjk = t_pow_3d.T * Mj[np.newaxis,:,:]
                #now sum along 'i' 
                Mjksum = np.sum(Mjk,axis=2)
                #do matrix inversion solver thing to get polynomial coeffs
                try:
                    Aj = np.linalg.lstsq(Mjksum,Bksum)[0]
                    pol = np.polyval(Aj[::-1],t-np.mean(t))
                except: 
                    pol = np.ones(npts)
                #Aj = np.dot(np.linalg.pinv(Mjksum), Bksum)
    #                plt.plot(t, f, 'ro', t, model*pol, 'go')
    #                plt.plot(t, pol, 'ms', tt, np.polyval(Aj[::-1],tnew), 'cs')
    #                plt.show()
    
                totmod[chunk[i]:chunk[i+1]] = model
                totpol[chunk[i]:chunk[i+1]] = pol
        else:
            maf = rsky(e, period, t0, 1e-8, jd)
            r = a*(1.-e**2) / (1.+e*np.cos(maf))
            zcomp = np.sin(omega+maf) * np.sin(inc) 
            #z = r*np.sqrt(1.-zcomp**2)
            pe = ((r*zcomp>0.)) #& (z <= 1.05*(r1+r2)*r2au))
            se = ((r*zcomp<0.)) #& (z <= 1.05*(r1+r2)*r2au))
            tt = jd % period
            if pe.any():
                totmod[pe] = np.interp(tt[pe], tempt1, tempres1)
                totmod[pe] = (totmod[pe] - 1.) * crowd[pe] + 1.
            if se.any():
                totmod[se] = np.interp(tt[se], tempt2, tempres2)
                totmod[se] = (totmod[se] - 1.) * crowd[se] + 1.
        #     if np.sum(totmod[se]-1.) == 0.:
        #         return np.ones_like(totmod), totpol
        # if np.sum(totmod-1.) == 0.:
        #   return totmod, totmod
        return totmod, totpol

    def _rvfit_old(self, rvpars, t):
        m1, m2, period, tpe, esinw, ecosw, inc, k0, rverr = rvpars

        a = ((period / d2y) ** 2 * (m1+m2)) ** (1. / 3.)
        e = np.sqrt(esinw**2+ecosw**2)
        omega = np.arctan2(esinw, ecosw)

        fpe = np.pi/2. - omega

        t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
            (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
            np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))


        maf = rsky(e, period, t0, 1e-8, t)

        # egative sign b/c observers' ref. frame is flipped
        vr2 = -np.sqrt(8.875985e12 * m1**2 / (m1+m2) / (a * (1-e**2))) * np.sin(inc) * \
              (np.cos(omega+maf) + e * np.cos(omega))
        omega+=np.pi # periapse of primary is 180 offset
        vr1 = -np.sqrt(8.875985e12 * m2**2 / (m1+m2) / (a * (1-e**2))) * np.sin(inc) * \
              (np.cos(omega+maf) + e * np.cos(omega))
        return vr1+k0, vr2+k0

    def _lcfit(self, lcpars, jd, quarter, flux, dflux, crowd,
              polyorder=2, ooe=True, flares=None, mult=1.7):
        """Computes light curve model
        
        Parameters
        ----------
        lcpars : float array
            parameters for LC fitting: 
            msum, rsum, rratio, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4
        jd : float array
            time array
        quarter : float array
            corresponding kepler quarter for a given time
        flux : float array
            observed flux
        dflux : float array
            flux error
        crowd : float array
            array of crowding values (additional flux)
        polyorder : int
            order of polynomial to detrend lightcurve
        
        Returns
        -------
        totmod : float array
            array of model fluxes
        totpol : float array
            array of polynomials for detrending
        """
        # r1, r2, frat derive from m1, m2, z0, t0, dist, E(B-V), scaleheight
        msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, \
            q1, q2, q3, q4 = lcpars

        # LD transformations (Kipping 2013)
        c1 = 2.*np.sqrt(q1)*q2
        c2 = np.sqrt(q1)*(1.-2.*q2)
        c3 = 2.*np.sqrt(q3)*q4
        c4 = np.sqrt(q3)*(1.-2.*q4)
        ldcoeffs1 = np.array([c1, c2])
        ldcoeffs2 = np.array([c3, c4])
            
#        if r2 > r1:
#            r1, r2 = r2, r1
#            m1, m2 = m2, m1
#            frat = 1./frat
        omega=np.arctan2(esinw,ecosw)
        e=np.sqrt(esinw**2+ecosw**2)

        # nip it at the bud.
        if (e>=1.):
            #print "e>=1", e
            return -np.inf, -np.inf
            
        # r1 = rsum/(1.+rrat)
        # r2 = rsum/(1.+1./rrat)
        r1, r2 = self.sumrat_to_12(rsum, rrat)
        a = self.get_a(period, msum)
        inc = self.get_inc(b, r1, a)
        #inc = np.arccos(b*r1/(a/r2au))
        
        if np.isnan(inc):
            #print "inc is nan", inc
            return -np.inf, -np.inf

        self.updatepars(msum=msum, rsum=rsum, rrat=rrat, period=period, tpe=tpe,
                       esinw=esinw, ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4,
                       frat=frat, r1=r1, r2=r2, inc=inc)
        fpe = np.pi/2. - omega
        fse = -np.pi/2. - omega

        # transform time of center of PE to time of periastron (t0)
        # from Eq 9 of Sudarsky et al (2005)
        t0 = tpe - self.sudarsky(fpe, e, period)
        if np.isnan(t0) or np.isinf(t0):
            print("Bad t0: {0}".format([tpe, fpe, e, period, t0]))
            return -np.inf, -np.inf
        tse = t0 + self.sudarsky(fse, e, period)
        # t0 = tpe - (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
        #     (e*np.sin(fpe)/(1.+e*np.cos(fpe)) - 2.*(1.-e**2)**(-0.5) * \
        #     np.arctan(np.sqrt(1.-e**2) * np.tan((fpe)/2.) / (1.+e)))
        # tse = t0 + (-np.sqrt(1.-e**2) * period / (2.*np.pi)) * \
        #     (e*np.sin(fse)/(1.+e*np.cos(fse)) - 2.*(1.-e**2)**(-0.5) * \
        #     np.arctan(np.sqrt(1.-e**2) * np.tan((fse)/2.) / (1.+e)))
        self.tpe = tpe
        self.tse = tse
        # if tse<tpe:
        #     tse+=period
            
        tempt1, tempres1 = self.lctemplate(lcpars, period, omega, e, a, inc, r1,
                                           ldcoeffs1, r2/r1, tpe, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=True)

        tempt2, tempres2 = self.lctemplate(lcpars, period, omega, e, a, inc, r2,
                                           ldcoeffs2, r1/r2, tse, t0,
                                           cadence = self.cadence,
                                           exp = self.exp, pe=False)
        if np.any(np.isinf(tempt1)) or np.any(np.isinf(tempt2)):
            return -np.inf, -np.inf

        tempt1 = tempt1 % period
        tempt2 = tempt2 % period
        tempres1 = (tempres1 - 1.)/(1. + frat) + 1.
        tempres2 = (tempres2 - 1.)/(1. + 1./frat) + 1.

        sorting1 = np.argsort(tempt1)
        sorting2 = np.argsort(tempt2)

        tempres1 = tempres1[sorting1]
        tempt1 = tempt1[sorting1]
        tempres2 = tempres2[sorting2]
        tempt2 = tempt2[sorting2]

        #not including crowdsap term.
        #tempres1 = (tempres1 + frat) / (1.+frat)
        #tempres2 = (tempres2 * frat + 1.) / (1. + frat)
        totmod, totpol = np.ones(len(jd)), np.ones(len(jd))

        maf = rsky(e, period, t0, 1e-8, jd)
        r = a*(1.-e**2) / (1.+e*np.cos(maf))
        zcomp = np.sin(omega+maf) * np.sin(inc) 
        pe = ((r*zcomp>0.)) #& (z <= 1.05*(r1+r2)*r2au))
        se = ((r*zcomp<0.)) #& (z <= 1.05*(r1+r2)*r2au))
        tt = jd % period

        if pe.any():
            totmod[pe] = np.interp(tt[pe], tempt1, tempres1)
            totmod[pe] = (totmod[pe] - 1.) * crowd[pe] + 1.
            
        if se.any():
            totmod[se] = np.interp(tt[se], tempt2, tempres2)
            totmod[se] = (totmod[se] - 1.) * crowd[se] + 1.
        
        if polyorder>0:
            clip = (abs((tt - tpe%period))<self.pe_dur%period*mult) | \
                    (abs((tt - tpe%period))>period-self.pe_dur%period*mult) | \
                    (abs((tt - tse%period))<self.se_dur%period*mult) | \
                    (abs((tt - tse%period))>period-self.se_dur%period*mult)
            self._clip = clip
            chunk = np.where(abs(np.diff(clip.astype(int)))>0)[0]
            chunk = np.append(chunk, 0)
            chunk = np.unique(np.sort(np.append(chunk, len(tt))))
            self.chunk = chunk
            if flares is None:
                totpol = poly_lc_cwrapper(jd, flux, dflux, totmod, chunk, porder=polyorder, ooe=ooe)
            else:
                totpol = np.ones(len(jd))
                for ii in range(len(chunk)-1):
                    bad = (totmod[chunk[ii]:chunk[ii+1]] < 1) | flares[chunk[ii]:chunk[ii+1]]
                    tt = jd[chunk[ii]:chunk[ii+1]][~bad]
                    mmodel = totmod[chunk[ii]:chunk[ii+1]][~bad]
                    ff = flux[chunk[ii]:chunk[ii+1]][~bad]
                    eef = dflux[chunk[ii]:chunk[ii+1]][~bad]
                    tnew = tt - np.mean(tt)
                    nnpts = len(ff)
                    npts = len(bad)
                    if bad[0] or bad[-1]:
                        poly_remember = polyorder
                        polyorder=1
                    order_pow = np.arange(polyorder+1)
                    t_pow = tnew[:,np.newaxis]**order_pow
                    Bk = np.ones(shape=(polyorder+1,nnpts))*((ff/mmodel)/(eef/mmodel)**2)
                    Bk*=t_pow.T
                    #sum along 'i' (or along each row)
                    Bksum = np.sum(Bk,axis=1)
                    #Mjk = sum over i (tdiff_i)^j * (tdiff_i)^k / (sigma_i/M_i)^2
                    #construct 3 rows x npts columns 
                    Mj = np.ones(shape=(polyorder+1,nnpts))/(eef/mmodel)**2
                    Mj*=t_pow.T
                    #transform from 2D (j rows x i columns) to 3D (k x j x i)
                    t_pow_3d = tnew[:,np.newaxis,np.newaxis]**order_pow
                    Mjk = t_pow_3d.T * Mj[np.newaxis,:,:]
                    #now sum along 'i' 
                    Mjksum = np.sum(Mjk,axis=2)
                    #do matrix inversion solver thing to get polynomial coeffs
                    try:
                        Aj = np.linalg.lstsq(Mjksum,Bksum)[0]
                        pol = np.polyval(Aj[::-1],jd[chunk[ii]:chunk[ii+1]]-np.mean(jd[chunk[ii]:chunk[ii+1]]))
                    except: 
                        pol = np.ones(npts)
                    #Aj = np.dot(np.linalg.pinv(Mjksum), Bksum)
        #                plt.plot(t, f, 'ro', t, model*pol, 'go')
        #                plt.plot(t, pol, 'ms', tt, np.polyval(Aj[::-1],tnew), 'cs')
        #                plt.show()
                    if bad[0] or bad[-1]:
                        polyorder = poly_remember
                    totpol[chunk[ii]:chunk[ii+1]] = pol         
#            phase = ((jd - tpe) % period) / period
#            sorting = np.argsort(phase)
#            nopoly = (totpol[sorting] == 1.)
#            if (np.sum(nopoly)>0) and (np.sum(nopoly)<len(totpol)*0.1):
#                _totpol = totpol[sorting]
#                tmp = np.interp(phase[sorting][nopoly], phase[sorting][~nopoly], flux[sorting][~nopoly]/totpol[sorting][~nopoly])
#                #print np.sum(nopoly), np.sum(~nopoly)
#                _totpol[nopoly] = flux[sorting][nopoly] / tmp
#                totpol[sorting] = _totpol
        return totmod, totpol
        
    def _getvals_defunct2(self, fit_params, partype='allpars'):
        """Grabs the values of input

        Parameters
        ----------
        fit_params : float array or dict
                    if dict, updates keblat parameters and returns pars according to input keys
                    if array, returns array

        """
        if type(fit_params) is dict:
            self.updatepars(**fit_params)
            return self.getpars(partype=partype)
        elif type(fit_params) is np.ndarray:
            return fit_params
        else:
            self.updatepars(**fit_params.valuesdict())
            return self.getpars(partype=partype)

#    def _getvals_defunct(self, fit_params, ntotpars, lctype):
#        """Fetch values from parameters
#
#        Parameters
#        ----------
#        fit_params : dict or float array
#            can input lmfit's Parameter class or just an array of parameter vals
#        ntotpars : int
#            total number of parameters to be fit
#        lcpars : Boolean
#            if True, returns set of parameters for light-curve fitting only
#            if False, returns all parameters
#
#        Returns
#        -------
#        guess : float array
#            array of values for each parameter
#        """
#
#        guess=np.empty(ntotpars)
#        for jj in range(ntotpars):
#            try:
#                guess[jj] = fit_params[self.pars.keys()[jj]].value
#            except KeyError:
#                guess[jj] = self.pars.values()[jj]
#            except ValueError:
#                guess[jj] = fit_params[jj]
#            except IndexError:
#                guess[jj] = fit_params[jj]
#        if lcpars:
#            return np.array([guess[0], guess[1], self.r1+self.r2,
#                                           self.r2/self.r1, guess[7], guess[8],
#                                           guess[9], guess[10], guess[11],
#                                            self.frat, guess[12], guess[13],
#                                            guess[14], guess[15]])
#        return guess
