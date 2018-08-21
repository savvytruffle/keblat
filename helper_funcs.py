import numpy as np
import matplotlib.pyplot as plt
import ctypes
import os
from numpy.ctypeslib import ndpointer
import platform
import time
import subprocess
import itertools
#import warnings
import sys
import kplr

#############################################
################ constants ##################
#############################################
TWOPI = 2.0*np.pi
d2y = 365.242 #days in a year
r2au = 0.0046491 #solar radii to AU
au = 1.49597e13 #AU in cm
cad = 0.0204340278 #(long) cadence in days

# distance grid from PANSTARRS 3D dust map (in pc)
ebv_dist = np.array([63.09573445, 79.43282347, 100., 125.89254118,
    158.48931925, 199.5262315, 251.18864315, 316.22776602,
    398.10717055, 501.18723363, 630.95734448, 794.32823472, 1000.,
   1258.92541179, 1584.89319246, 1995.26231497, 2511.88643151,
   3162.27766017, 3981.07170553, 5011.87233627, 6309.5734448,
   7943.28234724, 10000., 12589.25411794, 15848.93192461,
  19952.62314969, 25118.8643151, 31622.77660168, 39810.71705535,
  50118.72336273, 63095.73444802])


# these are "bad regions" list from welsh et al. must convert these
# dates to the correct time offset (from BJD-2,455,000 to BJD-start of
# Kepler data taking...)
badregions = np.array([   13.512,    18.703,    35.98 ,    36.05 ,    38.97 ,    39.02 ,
      42.12 ,    42.17 ,    43.9  ,    79.2  ,    81.67 ,    83.947,
     104.544,   198.351,   203.76 ,   269.399,   326.473,   466.67 ,
     467.   ,   475.18 ,   477.355,   478.3  ,   491.98 ,   492.128,
     496.255,   570.3  ,   573.47 ,   585.26 ,   683.95 ,   688.9  ,
     714.13 ,   719.8  ,   762.9  ,   768.88 ,   827.9  ,   837.16 ,
     837.447,   843.1  ,   841.47 ,   845.2  ,   851.31 ,   854.3  ,
     857.1  ,   863.2  ,   871.29 ,   926.6  ,   929.91 ,   930.018,
     943.35 ,   949.8  ,   951.   ,   951.5  ,   954.3  ,   955.2  ,
     960.   ,   988.1  ,   994.   ,   995.   ,   996.   ,   997.   ,
    1011.85 ,  1064.7 ,  1101.98 ,  1111.6  ]) + 5000. - 4833.

badregions = np.append(badregions, np.array([1093.64, 1093.8, 1093.95, 246.2, 373.3]))

suspectregions = np.array([200.2, 246.2, 373.3])

transit_x = np.array([-1.        , -0.99999997, -0.99999975, -0.99999893, -0.99999673,
       -0.99999187, -0.99998243, -0.99996575, -0.99993828, -0.99989547,
       -0.99983165, -0.99973989, -0.99961185, -0.9994377 , -0.99920598,
       -0.99890342, -0.99851485, -0.99802306, -0.99740861, -0.99664973,
       -0.99572212, -0.9945988 , -0.99324991, -0.99164249, -0.98974027,
       -0.98750342, -0.98488824, -0.98184684, -0.97832675, -0.97427047,
       -0.969615  , -0.96429112, -0.95822276, -0.95132602, -0.94350812,
       -0.93466609, -0.92468507, -0.91343624, -0.90077419, -0.88653352,
       -0.87052436, -0.85252643, -0.83228096, -0.80947943, -0.78374739,
       -0.75462061, -0.72150825, -0.68363324, -0.63992968, -0.6       ,
       -0.56470588, -0.52941176, -0.49411765, -0.45882353, -0.42352941,
       -0.38823529, -0.35294118, -0.31764706, -0.28235294, -0.24705882,
       -0.21176471, -0.17647059, -0.14117647, -0.10588235, -0.07058824,
       -0.03529412,  0.        ,  0.03529412,  0.07058824,  0.10588235,
        0.14117647,  0.17647059,  0.21176471,  0.24705882,  0.28235294,
        0.31764706,  0.35294118,  0.38823529,  0.42352941,  0.45882353,
        0.49411765,  0.52941176,  0.56470588,  0.6       ,  0.63992968,
        0.68363324,  0.72150825,  0.75462061,  0.78374739,  0.80947943,
        0.83228096,  0.85252643,  0.87052436,  0.88653352,  0.90077419,
        0.91343624,  0.92468507,  0.93466609,  0.94350812,  0.95132602,
        0.95822276,  0.96429112,  0.969615  ,  0.97427047,  0.97832675,
        0.98184684,  0.98488824,  0.98750342,  0.98974027,  0.99164249,
        0.99324991,  0.9945988 ,  0.99572212,  0.99664973,  0.99740861,
        0.99802306,  0.99851485,  0.99890342,  0.99920598,  0.9994377 ,
        0.99961185,  0.99973989,  0.99983165,  0.99989547,  0.99993828,
        0.99996575,  0.99998243,  0.99999187,  0.99999673,  0.99999893,
        0.99999975,  0.99999997,  1.        ])

transit_y = np.array([0.01836735,  0.03673469,  0.05510204,  0.07346939,  0.09183673,
        0.11020408,  0.12857143,  0.14693878,  0.16530612,  0.18367347,
        0.20204082,  0.22040816,  0.23877551,  0.25714286,  0.2755102 ,
        0.29387755,  0.3122449 ,  0.33061224,  0.34897959,  0.36734694,
        0.38571429,  0.40408163,  0.42244898,  0.44081633,  0.45918367,
        0.47755102,  0.49591837,  0.51428571,  0.53265306,  0.55102041,
        0.56938776,  0.5877551 ,  0.60612245,  0.6244898 ,  0.64285714,
        0.66122449,  0.67959184,  0.69795918,  0.71632653,  0.73469388,
        0.75306122,  0.77142857,  0.78979592,  0.80816327,  0.82653061,
        0.84489796,  0.86326531,  0.88163265,  0.9       ,  0.9146101 ,
        0.92606848,  0.93633897,  0.94555515,  0.95382419,  0.96123311,
        0.96785322,  0.97374342,  0.97895253,  0.98352113,  0.98748293,
        0.99086579,  0.99369256,  0.99598168,  0.99774766,  0.99900147,
        0.99975074,  1.        ,  0.99975074,  0.99900147,  0.99774766,
        0.99598168,  0.99369256,  0.99086579,  0.98748293,  0.98352113,
        0.97895253,  0.97374342,  0.96785322,  0.96123311,  0.95382419,
        0.94555515,  0.93633897,  0.92606848,  0.9146101 ,  0.9       ,
        0.88163265,  0.86326531,  0.84489796,  0.82653061,  0.80816327,
        0.78979592,  0.77142857,  0.75306122,  0.73469388,  0.71632653,
        0.69795918,  0.67959184,  0.66122449,  0.64285714,  0.6244898 ,
        0.60612245,  0.5877551 ,  0.56938776,  0.55102041,  0.53265306,
        0.51428571,  0.49591837,  0.47755102,  0.45918367,  0.44081633,
        0.42244898,  0.40408163,  0.38571429,  0.36734694,  0.34897959,
        0.33061224,  0.3122449 ,  0.29387755,  0.2755102 ,  0.25714286,
        0.23877551,  0.22040816,  0.20204082,  0.18367347,  0.16530612,
        0.14693878,  0.12857143,  0.11020408,  0.09183673,  0.07346939,
        0.05510204,  0.03673469,  0.01836735])


#############################################
if platform.system() == "Darwin":
  try:
    lib = ctypes.CDLL('./helpers_mac.so')
  except:
    raise Exception("Can't find .so file; please type ``make`` to compile the code.")
elif platform.system() == "Linux":
  try:
    lib = ctypes.CDLL('./helpers_linux.so')
  except:
    raise Exception("Can't find .so file; please type ``make`` to compile the code.")
else:
  raise Exception("Unknown platform.")


resample_time = lib.resample_time
resample_time.restype = ctypes.c_int
resample_time.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                       ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double), 
                        ctypes.c_int]

dchi_fn = lib.dchi_fn
dchi_fn.restype = ctypes.c_int
dchi_fn.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_long), ctypes.c_long]

dchi_fn2 = lib.dchi_fn2
dchi_fn2.restype = ctypes.c_int
dchi_fn2.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_long), ctypes.c_long]

dchi_fn_mask = lib.dchi_fn_mask
dchi_fn_mask.restype = ctypes.c_int
dchi_fn_mask.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_long), ctypes.c_long]


dchi_fn_gs = lib.dchi_fn_gs
dchi_fn_gs.restype = ctypes.c_int
dchi_fn_gs.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ndpointer(dtype=ctypes.c_long), ndpointer(dtype=ctypes.c_long), ctypes.c_long]


dchiChoosePC = lib.dchiChoosePC
dchiChoosePC.restype = ctypes.c_int
dchiChoosePC.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                         ndpointer(dtype=ctypes.c_int), ndpointer(dtype=ctypes.c_double),
                         ndpointer(dtype=ctypes.c_double), ctypes.c_double, ndpointer(dtype=ctypes.c_int),
                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

call_qats = lib.get_qats_likelihood
call_qats.restype = ctypes.c_int
call_qats.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(dtype=ctypes.c_double),
                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]

call_qats_indices = lib.get_qats_indices
call_qats_indices.restype = ctypes.c_int
call_qats_indices.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_int), ctypes.POINTER(ctypes.c_double)]

poly_lc = lib.poly_lc
poly_lc.restype = ctypes.c_int
poly_lc.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_long),
                    ctypes.c_int, ctypes.c_int]

poly_lc_ooe = lib.poly_lc_ooe
poly_lc_ooe.restype = ctypes.c_int
poly_lc_ooe.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_long),
                    ctypes.c_int, ctypes.c_int]

rsky_c = lib.rsky
rsky_c.restype = ctypes.c_int
rsky_c.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double), ctypes.c_double, 
		  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
		 
occultquad_c = lib.occultquad
occultquad_c.restype = ctypes.c_int
occultquad_c.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double), 
			ndpointer(dtype=ctypes.c_double), ctypes.c_double, ctypes.c_double, 
			ctypes.c_double, ctypes.c_int]

def rsky(e, P, t0, eps, t):
	maf = np.zeros(len(t))
	rsky_c(t.copy(), maf, e, P, t0, eps, len(t))
	return maf

def occultquad(znorm, u1, u2, rrat, Npts):
	mu = np.zeros(Npts)
	mu0 = mu*0.
	occultquad_c(mu0, mu, znorm, u1, u2, rrat, Npts)
	return mu

class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, junk1, junk2, junk3, *args, **kwargs):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            #warnings.warn("Terminating optimization: time limit reached")
            print "Terminating optimization: time limit reached"
            return True
        # else:
        #     # you might want to report other stuff here
        #     if elapsed > 10:
        #         print("Elapsed: %.3f sec" % elapsed)
        return False

def call_qats_c(dmin, dmax, y, q=1):
    N = len(y)
    #d = np.zeros(N)
    Sbest = ctypes.c_double(0.0)
    Mbest = ctypes.c_int(0)
    call_qats(dmin, dmax, q, N, y, ctypes.byref(Sbest), ctypes.byref(Mbest))
    return Sbest.value, Mbest.value

def load_kplr(kic, lc=True):
    star = kplr.API().star(kic).get_light_curves(short_cadence=not lc)
    jd, flux, dflux = np.array([], dtype='float64'), np.array([], dtype='float64'), np.array([], dtype='float64')
    cadnum, quarter = np.array([], dtype='float64'), np.array([], dtype='float64')
    quality, crowd = np.array([], dtype='float64'), np.array([], dtype='float64')
    for ii in range(len(star)):
        _s = star[ii].open(clobber=False)
        _Npts = _s[1].data.shape[0]
        jd = np.append(jd, _s[1].data['TIME'])
        flux = np.append(flux, _s[1].data['SAP_FLUX'])
        dflux = np.append(dflux, _s[1].data['SAP_FLUX_ERR'])
        cadnum = np.append(cadnum, _s[1].data['CADENCENO'])
        quarter = np.append(quarter, np.zeros(_Npts, dtype=int) + _s[0].header['QUARTER'])
        quality = np.append(quality, _s[1].data['SAP_QUALITY'])
        crowd = np.append(crowd, np.zeros(_Npts) + _s[1].header['CROWDSAP'])
        naninds = np.isnan(flux)
        jd = jd[~naninds]
        dflux = dflux[~naninds]
        cadnum = cadnum[~naninds]
        quarter = quarter[~naninds]
        quality = quality[~naninds]
        crowd = crowd[~naninds]
        flux = flux[~naninds]

    a = np.nanmedian(flux)
    if a>1:
        flux = flux/a
        dflux = abs(dflux/a)
        
    return jd, flux, dflux, cadnum.astype(int), quarter.astype(int), quality, crowd

def call_qats_indices_c(dmin, dmax, y, M, q=1):
    N = len(y)
    Sbest = ctypes.c_double(0.0)
    indices = np.zeros(M, dtype=np.int32)
    call_qats_indices(M, dmin, dmax, q, N, y, indices, ctypes.byref(Sbest))
    return indices, Sbest.value

def call_qats_cpp(dmin, dmax, fname, q=1):
    call_sequence = """/astro/users/windemut/keblat/qats/call_qats {0} {1} {2} {3} >> {4}\n""".format(
                                                fname, dmin, dmax, q, fname+".qatsout")
    subprocess.call(call_sequence, shell=True)
    return

def times_of_transits_pub(kic):
    if kic == 12644769:
        result = np.array([140.4298, 370.6949, 592.203, 822.449, 1044., 1274.23, 1495.76])
        result_shifted = np.array([143.204, 369.12, 594.967, 820.866, 1046.77, 1272.63, 1498.53])
        result_shifted = np.array([137.681, 372.27, 589.423, 824.06, 1041.21, 1275.86, 1492.99])

    return result

def poly_lc_cwrapper(t, f, ef, model, chunks, porder=2, ooe=True):
    totpol = np.ones(len(t), dtype=np.float64)
    f, ef = f.astype(np.float64), ef.astype(np.float64)
    if ooe:
        poly_lc(totpol, t.copy(), f.copy(), ef.copy(), model.copy(), chunks.copy(), porder, len(chunks))
    else:
        poly_lc_ooe(totpol, t.copy(), f.copy(), ef.copy(), model.copy(), chunks.copy(), porder, len(chunks))
    return totpol

def poly_lc_python(jd, flux, dflux, totmod, chunk, porder=2):
    totpol = np.ones(len(t), dtype=np.float64)

    for i in range(len(chunk)-1):
        #print i, chunk[i], chunk[i+1], self.jd[chunk[i]:chunk[i+1]]
        t = jd[chunk[i]:chunk[i+1]]
        f = flux[chunk[i]:chunk[i+1]]
        ef = dflux[chunk[i]:chunk[i+1]]
        model = totmod[chunk[i]:chunk[i+1]]
        bad = (model<1)
        tt = t[~bad]
        tnew = tt-np.mean(tt)
        mmodel = model[~bad]
        ff = f[~bad]
        eef = ef[~bad]
        try:
            poly_instance = np.poly1d(np.polyfit(tnew, ff/mmodel, polyorder,
                   w=1./(eef/mmodel)))
            totpol[chunk[i]:chunk[i+1]] = poly_instance(t-np.mean(t))
        except:
            print "polyfit not good"
        if i==0:
            print t, f, ef, model, tt, tnew
    return totpol


def fdrule(x, ret_binwidth=False):
    binwidth = 2.7*np.std(x) / len(x)**(1./3.)
    if ret_binwidth:
        return binwidth
    nbins = int(np.ceil((x.max()-x.min())/binwidth))
    return nbins

def dchiPC_c(t, rinds, f, ef, depth, duration, ndur, Nrinds, porder, cwidth):
    dchi_all = np.zeros((len(porder), len(cwidth), ndur, Nrinds))
    for ii, jj in itertools.product(range(len(porder)), range(len(cwidth))):
        print porder[ii], cwidth[jj]
        dchi = np.zeros((ndur, Nrinds)).flatten()
        dchiChoosePC(dchi, t, rinds, f, ef, depth, duration, ndur, Nrinds, porder[ii], cwidth[jj])
        dchi_all[ii,jj,:,:] = dchi.reshape((ndur, Nrinds))
    return dchi_all

# double *dchi, double *t, int *rinds, double *f, double *ef, double depth, int *duration,
#                  int ndur, int Nrinds, int porder, int cwidth


#def dchigrid_c(t, jumps, f, ef, depth, duration, ndep, ndur, fname, porder, cwidth, dchi=None):
#    if os.path.isfile(fname):
#        print "File already exists; loading."
#        dchi = np.loadtxt(fname)
#        return dchi.reshape((ndep, ndur, len(t)))
#    assert len(porder) == ndur, "Porder array must have size = # of durations"
#    Ntot = len(t)
#    t, f, ef, depth = t.astype(np.float64), f.astype(np.float64), ef.astype(np.float64), depth.astype(np.float64)
#    if dchi is None:
#        dchi = np.zeros((ndep, ndur, Ntot)).flatten()
#    else:
#        if len(dchi.shape)>1:
#            dchi = dchi.flatten()
#    dchi_fn(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
#    dchi = dchi.reshape((ndep, ndur, Ntot))
#    # baseline = np.ones((ndep, ndur, 1))
#    # for ii in range(ndep):
#    #     for jj in range(ndur):
#    #         for ll in range(len(jumps)-1):
#    #             zeros = (dchi[ii, jj, jumps[ll]:jumps[ll+1]]==0.) | (np.isnan(dchi[ii, jj, jumps[ll]:jumps[ll+1]]))
#    #             baseline[ii, jj, 0] = np.nanmedian(np.sort(dchi[ii, jj, jumps[ll]:jumps[ll+1]][~zeros])[int(0.16*np.sum(~zeros)):int(0.84*np.sum(~zeros))])
#    #             dchi[ii, jj, jumps[ll]:jumps[ll+1]][zeros] = baseline[ii, jj, 0]
#            #dchi[ii, jj, :][ecl_mask] = baseline[ii, jj, 0]
#    print "Creating file:", fname
#    with file(fname, 'w') as outfile:
#        outfile.write('# Array shape: {0}\n'.format(dchi.shape))
#        for data_slice in dchi:
#            np.savetxt(outfile, data_slice, fmt='%.6e')
#    return dchi

def dchigrid_c2(t, jumps, f, ef, depth, duration, ndep, ndur, fname, porder, cwidth, dchi=None,
                gauss_weighting=False, strict_start=True, clobber=False):
    if os.path.isfile(fname+'.npz') and not clobber:
        print "NPZ file already exists; loading."
#        dchi = np.loadtxt(fname)
#        return dchi.reshape((ndep, ndur, len(t)))
        dchi = np.load(fname+'.npz')['dchi']
        return dchi
    assert len(porder) == ndur, "Porder array must have size = # of durations"
    Ntot = len(t)
    t, f, ef, depth = t.astype(np.float64), f.astype(np.float64), ef.astype(np.float64), depth.astype(np.float64)
    if dchi is None:
        dchi = np.zeros((ndep, ndur, Ntot)).flatten()
    else:
        if len(dchi.shape)>1:
            dchi = dchi.flatten()
    if gauss_weighting:
        dchi_fn_gs(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
    else:
        if strict_start:
            #changed this from dchi_fn2 to dchi_fn_mask
            dchi_fn_mask(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
        else:
            dchi_fn(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
    dchi = dchi.reshape((ndep, ndur, Ntot))
    # baseline = np.ones((ndep, ndur, 1))
    # for ii in range(ndep):
    #     for jj in range(ndur):
    #         for ll in range(len(jumps)-1):
    #             zeros = (dchi[ii, jj, jumps[ll]:jumps[ll+1]]==0.) | (np.isnan(dchi[ii, jj, jumps[ll]:jumps[ll+1]]))
    #             baseline[ii, jj, 0] = np.nanmedian(np.sort(dchi[ii, jj, jumps[ll]:jumps[ll+1]][~zeros])[int(0.16*np.sum(~zeros)):int(0.84*np.sum(~zeros))])
    #             dchi[ii, jj, jumps[ll]:jumps[ll+1]][zeros] = baseline[ii, jj, 0]
            #dchi[ii, jj, :][ecl_mask] = baseline[ii, jj, 0]
    print("Creating file {0}.npz...".format(fname))
    np.savez(fname+'.npz', dchi=dchi, depth=depth, duration=duration, p_choose=porder, 
             c_choose=cwidth, gw=gauss_weighting, st=strict_start)
#    with file(fname, 'w') as outfile:
#        outfile.write('# Array shape: {0}\n'.format(dchi.shape))
#        for data_slice in dchi:
#            np.savetxt(outfile, data_slice, fmt='%.6e')
    return dchi

def dchigrid_c_mask(t, jumps, f, ef, depth, duration, ndep, ndur, fname, porder, cwidth, dchi=None,
                gauss_weighting=False, strict_start=True, clobber=False):
    if os.path.isfile(fname) and not clobber:
        print "File already exists; loading."
        dchi = np.loadtxt(fname)
        return dchi.reshape((ndep, ndur, len(t)))
    assert len(porder) == ndur, "Porder array must have size = # of durations"
    Ntot = len(t)
    t, f, ef, depth = t.astype(np.float64), f.astype(np.float64), ef.astype(np.float64), depth.astype(np.float64)
    if dchi is None:
        dchi = np.zeros((ndep, ndur, Ntot)).flatten()
    else:
        if len(dchi.shape)>1:
            dchi = dchi.flatten()
    if gauss_weighting:
        dchi_fn_gs(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
    else:
        if strict_start:
            dchi_fn_mask(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
        else:
            dchi_fn(dchi, t, jumps, f, ef, depth, duration, ndep, ndur, len(jumps), porder, cwidth, Ntot)
    dchi = dchi.reshape((ndep, ndur, Ntot))
    # baseline = np.ones((ndep, ndur, 1))
    # for ii in range(ndep):
    #     for jj in range(ndur):
    #         for ll in range(len(jumps)-1):
    #             zeros = (dchi[ii, jj, jumps[ll]:jumps[ll+1]]==0.) | (np.isnan(dchi[ii, jj, jumps[ll]:jumps[ll+1]]))
    #             baseline[ii, jj, 0] = np.nanmedian(np.sort(dchi[ii, jj, jumps[ll]:jumps[ll+1]][~zeros])[int(0.16*np.sum(~zeros)):int(0.84*np.sum(~zeros))])
    #             dchi[ii, jj, jumps[ll]:jumps[ll+1]][zeros] = baseline[ii, jj, 0]
            #dchi[ii, jj, :][ecl_mask] = baseline[ii, jj, 0]
    print "Creating file:", fname
    with file(fname, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(dchi.shape))
        for data_slice in dchi:
            np.savetxt(outfile, data_slice, fmt='%.6e')
    return dchi

def reinterpret_x_py(t, x, tbary):
    # need delta chi2s so that transit = dip
    xshift = np.ones(len(x))*np.max(x)
    for ii in range(len(t)-1):#1, len(t)-1):
        tnew = t[ii:ii+2]-tbary[ii:ii+2]
        indx_start = np.searchsorted(t, np.min(tnew), 'left')
        indx_end = np.searchsorted(t, np.max(tnew), 'right')
        indx = np.arange(indx_start, indx_end)
        #print tnew, indx_start, indx_end
        if len(indx) > 0:
            xnew = np.interp(t[indx], tnew, x[ii:ii+2])
            xnew0 = xnew.copy()
            if (xnew <= xshift[indx]).all():
                xshift[indx] = xnew0
            #print indx, t[indx], xnew, xshift[indx]
    bad = (xshift == np.max(x))
    xshift[bad] = np.median(xshift[~bad])
    return xshift
    
def reinterpret_x_c(t, x, tbary):
    # assumes delchisq at transits are PEAKS
    baselevel = x.min() #change to x.max() if delchisq at transits are DIPS
    xshift=np.ones(len(t))*baselevel
    resample_time(t, x, tbary, xshift, len(x))
    bad = (xshift == baselevel)
    xshift[bad] = np.median(xshift[~bad])
    return xshift

def user_rc(lw=1.5, fontsize=10):
    """Set plotting RC parameters"""
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    plt.rc('lines', linewidth=lw)
    import matplotlib
    if matplotlib.__version__[0]=='2':
        from cycler import cycler
#        plt.rc('axes', prop_cycle=cycler(c=tableau20), lw=1, labelsize=18, titlesize=22)
        plt.rc('axes', lw=1, labelsize=18, titlesize=22)

    else:
#        plt.rc('axes', axes_cycle=tableau20, lw=1, labelsize=18, titlesize=22)
        plt.rc('axes', lw=1, labelsize=18, titlesize=22)
    plt.rc('figure', titlesize=22, figsize=(10,8))
    return
    #plt.rc('font', size=7)



def get_excludelist(fname='data/sed_flag_file_0328'):
    # cols are kic#, ubv, sdss, wise, 2mass
    flaglist = np.loadtxt(fname, dtype=int)
    flag_headers = ['Umag','Bmag','Vmag','gmag','rmag','imag',
                                 'zmag','w1', 'w2', 'Jmag','Hmag','Kmag']
    klist = flaglist[:,0]
    excludelist = [[item for item, flag in zip(flag_headers, flaglist[j, 1:])
                    if flag == 1] for j in range(flaglist.shape[0])]

    ######### empirical exclude list here ################
    klist = np.append(klist, 3327980)
    excludelist += ['gmag','rmag','imag','zmag']
    return dict(zip(klist, excludelist))


def get3dmap(kic):
    """Get 3D Panstarrs dust map information; returns ebv_arr, ebv_sig,
    ebv_dist_bounds, ebv_bounds"""
    red3d = np.load('data/ebv3d_0216.npz')
    indx = np.where(red3d['kic'] == kic)[0]
    if indx is None:
        print "Sorry no 3D dust map information available. Using 2D instead."
        return
#        return ebv, keblat.sed[good, -1], (10., 10000.), (0.0, 1.0)
    ebv_arr, ebv_sig = red3d['ebv_arr'][indx].ravel(), red3d['ebv_sig'][indx].ravel()
    ebv_dist_bounds = (red3d['dmin'][indx][0], red3d['dmax'][indx][0])
    ebv_bounds = (ebv_arr[np.searchsorted(ebv_dist, ebv_dist_bounds[0])],
                          ebv_arr[np.searchsorted(ebv_dist, ebv_dist_bounds[1])-1])
    return ebv_arr, ebv_sig, ebv_dist_bounds, ebv_bounds
   
def during_eclipse(t, tpe, period, pwidth, swidth, sep, clip_tol=1.5, both=True):
    t = np.atleast_1d(t)
    phase = ((t-tpe)%period/period)
    phase[phase<-np.clip(pwidth*2., 0., 0.2)] += 1
    phase[phase>1.-np.clip(pwidth*2., 0., 0.2)] -= 1
    phase[phase>np.clip(sep+swidth*2., sep, 1.0)] -= 1
    if both:
        return (abs(phase)<clip_tol*pwidth) | (abs(phase-1.0)<clip_tol*pwidth) | (abs(phase-sep)<clip_tol*swidth)
    return (abs(phase)<clip_tol*pwidth) | (abs(phase-1.0)<clip_tol*pwidth)
    
def eclipse_mask(phase, sep, pwidth, swidth, tol=1.2, mask=False):
    """Returns indices of eclipses"""
    if mask:
        return (abs(phase)<tol*pwidth) | (abs(phase-sep)<tol*swidth) | (abs(1.-phase)<tol*pwidth)
    return np.where((abs(phase)<tol*pwidth) | (abs(phase-sep)<tol*swidth) | (abs(1.-phase)<tol*pwidth))[0]

def flagdata(t, f, ef, quality, ftol=0.003, ttol=1., qual_tol = 8, wn=0.,
             eclipses=None, retinds=False, daytol=0.2):
    """Identifies data with bad quality (default quality flag tolerance = 16) and
     data +/- 0.1 d around manual bad regions list; option to interpolate over
     eclipse regions too if eclipse indices are provided; returns array with
     interpolation over flagged data points"""
    flagged = np.where(quality > qual_tol)[0]
    # diff_f = np.insert(abs(np.diff(f)), 0, 0)
    # fluxjump_min = np.where((f < np.roll(f, -1)) & (f < np.roll(f, 1)) &
    #                         (diff_f > ftol) & (np.roll(diff_f, -1) > ftol))[0]
    # fluxjump_max = np.where((f > np.roll(f, -1)) & (f > np.roll(f, 1)) &
    #                         (diff_f > ftol) & (np.roll(diff_f, -1) > ftol))[0]
    # fluxjump = np.sort(np.concatenate((fluxjump_min, fluxjump_max)))
    # flagged = np.unique(np.sort(np.concatenate((bad, fluxjump))))
    for ii in range(len(badregions)):
        flagged = np.append(flagged, np.where((t<badregions[ii]+daytol) & (t>badregions[ii]-daytol))[0])
    if eclipses is not None:
        flagged = np.append(flagged, eclipses)
    flagged = np.unique(np.sort(flagged))
    if retinds:
        return flagged
    if np.sum(flagged) == 0:
        return f, ef
    allinds = np.arange(len(f))
    #print allinds
    nonflagged = allinds[~np.in1d(allinds, flagged)]       # ~np.array([ii in flagged for ii in allinds])
    if wn > 0:
        wn = np.random.normal(0, wn, len(flagged))
    try:
        f[flagged] = np.interp(t[flagged], t[nonflagged], f[nonflagged]) + wn
    except:
        print flagged, np.sum(flagged)
    # ef[flagged] *= 1.5
    # # loop through bad/flagged data points, interpolate using data points +/- 'ttol' day
    # # from flagged time and replace flagged data with interpolated values.
    # for ii in range(len(flagged)):
    #     time_chunk = ((t<t[flagged[ii]]+ttol) & (t>t[flagged[ii]]-ttol)) & (t != t[flagged[ii]])
    #     f[flagged[ii]] = np.median(f[time_chunk])
    # if plot:
    #     plt.figure()
    #     plt.plot(t, f0, 'k.')
    #     plt.plot(t[fluxjump_min], f0[fluxjump_min], 'r.')
    #     plt.plot(t[fluxjump_max], f0[fluxjump_max], 'g.')
    #     plt.plot(t[bad], f0[bad], 'c.')
    #     #plt.plot(keblat.jd, f, 'm.')
    #     plt.show()
    return f, ef

def find_fluxjumps(t, f, tol, eclipses):
    jumps = np.where(abs(np.diff(f[~eclipses]))>tol)[0]
    jumps +=1
    f_right = np.zeros((len(jumps), 30))
    f_left = f_right * 1.0
    for ii in range(len(jumps)):
        f_right[ii, :] = f[~eclipses][jumps[ii]+np.arange(30)]
        f_left[ii, :] = f[~eclipses][jumps[ii]-np.arange(30)-1]
    meanjumps = (abs(np.median(f_left, axis=1)-np.median(f_right, axis=1))>tol) * \
                (np.std(f_left, axis=1)<tol) * (np.std(f_right, axis=1)<tol)
    # jump_right = jumps[:, np.newaxis] + np.arange(15)
    # jump_left = jumps[:, np.newaxis] - (np.arange(15)+1)
    #
    # meanjumps = (abs(np.mean(jump_left, axis=1)-np.mean(jump_right, axis=1))>tol) * \
    #             (np.std(jump_left, axis=1) < tol) * (np.std(jump_right, axis=1) < tol)
    return jumps[meanjumps]

def padcads(t0, cadnum0, f0, ef0, chunks, flagged, tolmin = 1, tolmax = 7, cad = 0.0204340278):
    """Pads missing cadences within tolerance range (default, 1 < gaps < 7 cadences),
    by filling them with interpolation from 2 nearby neighbours."""
    fillcad = np.where((np.diff(cadnum0)>tolmin) & (np.diff(cadnum0)<=tolmax))[0]
    fillcad = fillcad[~np.in1d(fillcad+1,chunks)]
    t = t0.copy()
    ef = ef0.copy()
    cadnum = cadnum0.copy()
    f = f0.copy()
    for ii in range(len(fillcad)-1, -1, -1):
        nfillers = cadnum0[fillcad[ii]+1] - cadnum0[fillcad[ii]]
        t_filler = (np.arange(nfillers-1)+1)*cad + t[fillcad[ii]]
        t = np.insert(t, fillcad[ii]+1, t_filler)
        f_filler = np.interp(t_filler, t0[fillcad[ii]:fillcad[ii]+2], f0[fillcad[ii]:fillcad[ii]+2])
        f = np.insert(f, fillcad[ii]+1, f_filler)
        ef_filler = np.interp(t_filler, t0[fillcad[ii]:fillcad[ii]+2], ef0[fillcad[ii]:fillcad[ii]+2])
        ef = np.insert(ef, fillcad[ii]+1, ef_filler)
        cadnum = np.insert(cadnum, fillcad[ii]+1, (np.arange(nfillers-1)+1) + cadnum[fillcad[ii]])
    return t, cadnum, f, ef

def roll_np(x, N, threshold=1.0):
    if threshold is None:
        res = (np.roll(x, N))
    else:
        res = (np.roll(x, N) < threshold)
    if N < 0:
        res[N:] = [False] * abs(N)
    elif N > 0:
        res[:N] = [False] * N
    else:
        print("N={0}".format(N))
    return res
    
def identify_gaps(tcad, tol = 15, quarts=None, retbounds_inds=True):
    """Returns indices of gaps where no data for > 7 Kepler cadences; if
    retbounds_inds is True, returns inds of beginning and end of condition"""
    jumps = np.where((np.diff(tcad)>tol))[0]
    if retbounds_inds:
        jumps+=1
        jumps = np.insert(jumps, 0, 0)
        jumps = np.append(jumps, len(tcad))
    if quarts is not None:
        qdiff = np.where(np.diff(quarts)>0)[0]+1
        jumps = np.unique(np.append(jumps, qdiff))
    return jumps

def prep_lc(t0, cadnum0, phase0, f0, ef0, quality0, sep, pwidth, swidth, tpe, period,
            tol, clip, no_padding=False, jumps_on=False):
    """Preps lightcurve for uniform time-series analysis by identifying time gaps and flux jumps (chunks), interpolate
    over eclipses and bad data points (qual flag > 16, in bad regions list), and padding missing cadences btw.
    lc chunks"""
    t, cadnum, phase, f, ef, quality = t0.copy(), cadnum0.copy(), phase0.copy(), f0.copy(), ef0.copy(), quality0.copy()
#    eclipses = eclipse_mask(phase, sep, pwidth, swidth, tol=tol)
    chunks = identify_gaps(cadnum, retbounds_inds=True)
    # ooe = np.arange(len(t))[~np.in1d(np.arange(len(t)), eclipses)]

    #ef_tol = 10.*np.median(ef[~clip])
    ef_tol = 10. * np.median(np.sort(abs(np.diff(f[~clip])))[int(0.16*np.sum(~clip)):int(0.84*np.sum(~clip))])
    #print ef_tol, 10.*np.median(ef[~clip])
    if jumps_on:
        fjumps = find_fluxjumps(t, f, ef_tol, clip)
        fjumps = np.arange(len(t))[np.in1d(t, t[~clip][fjumps])]
        chunks = np.unique(np.concatenate((chunks, fjumps)))

    for ii in range(len(chunks)-1):
        eclipse_chunk = eclipse_mask(phase[chunks[ii]:chunks[ii+1]], sep, pwidth, swidth, tol=tol)
        #print ii, len(t[chunks[ii]:chunks[ii+1]]), len(f[chunks[ii]:chunks[ii+1]])
        ef[chunks[ii]:chunks[ii+1]] = np.median(ef0[chunks[ii]:chunks[ii+1]])
        f[chunks[ii]:chunks[ii+1]], ef[chunks[ii]:chunks[ii+1]] = flagdata(t[chunks[ii]:chunks[ii+1]],
                                                                           f[chunks[ii]:chunks[ii+1]],
                                                                           ef[chunks[ii]:chunks[ii+1]],
                                                                           quality[chunks[ii]:chunks[ii+1]],
                                                                           wn=1.2*ef[chunks[ii]], eclipses=eclipse_chunk)

    if no_padding:
        ecl_mask = eclipse_mask(phase, sep, pwidth, swidth, tol=tol, mask=True)
        return t, cadnum, phase, f, ef, chunks, ecl_mask

    t, cadnum, f, ef = padcads(t, cadnum, f, ef, chunks, tolmin=1, tolmax=7, cad=0.0204340278)

    phase = ((t-tpe) % period)/period

#    eclipses = eclipse_mask(phase, sep, pwidth, swidth, tol=tol)
    ecl_mask = eclipse_mask(phase, sep, pwidth, swidth, tol=tol, mask=True)

    time_gaps = identify_gaps(cadnum, retbounds_inds=True)

    fjumps = find_fluxjumps(t, f, ef_tol, ecl_mask)
    fjumps = np.arange(len(t))[np.in1d(t, t[~ecl_mask][fjumps])]
    chunks = np.unique(np.concatenate((time_gaps, fjumps)))
    return t, cadnum, phase, f, ef, chunks, ecl_mask

def flag_outliers_obsolete(t, f, dflux, quality, ftol=10, ttol=1., plot=False):
    """Crude outlier rejection. Interpolates 'bad' data point by taking median of data values
    within some time tolerance (ttol=1 default) around it. 'Bad' data points are single spikes"""
    f0 = f.copy()
    bad = np.where(quality > 10000)[0]
    f_absdif = np.insert(abs(np.diff(f)),0,0)
    fluxjump_min = np.where((f < np.roll(f, -1)) & (f < np.roll(f, 1)) & 
                            (f_absdif>10.*ftol*np.median(dflux)) & 
                            (np.roll(f_absdif, -1)>10.*ftol*np.median(dflux)) & 
                            (np.insert(abs(np.diff(t)),0,1.)<ttol))[0]
    fluxjump_max = np.where((f > np.roll(f, -1)) & (f > np.roll(f, 1)) & 
                            (f_absdif>ftol*np.median(dflux)) & 
                            (np.roll(f_absdif, -1)>ftol*np.median(dflux)) & 
                            (np.insert(abs(np.diff(t)),0,1.)<ttol))[0]
    fluxjump = np.sort(np.concatenate((fluxjump_min, fluxjump_max)))

    flagged = np.unique(np.sort(np.concatenate((bad, fluxjump))))
    # loop through bad/flagged data points, interpolate using data points +/- 'ttol' day 
    # from flagged time and replace flagged data with interpolated values.
    for ii in range(len(flagged)):
        time_chunk = ((t<t[flagged[ii]]+ttol) * (t>t[flagged[ii]]-ttol)) * (t != t[flagged[ii]])
        f[flagged[ii]] = np.median(f[time_chunk])
    if plot:
        plt.figure()
        plt.plot(t, f0, 'k.', label='input')
        plt.plot(t[fluxjump_min], f[fluxjump_min], 'r.', label='max jumps')
        plt.plot(t[fluxjump_max], f[fluxjump_max], 'g.', label='min jumps')
        plt.plot(t[bad], f[bad], 'c.', label='all flagged')
        plt.show()
    return f

def poly_cont(t, f, ef, model, polyorder=2):
    """Computes marginalized polynomial fit to continuum"""
    bad = (model<1)
    tt = t[~bad]
    mmodel = model[~bad]
    ff = f[~bad]
    eef = ef[~bad]
    nnpts = len(ff)
    tnew = tt - np.mean(tt)
    order_pow = np.arange(polyorder+1)
    t_pow = tnew[:,np.newaxis]**order_pow
    Bk = np.ones(shape=(polyorder+1,nnpts))*((ff/mmodel)/(eef/mmodel)**2)
    Bk*=t_pow.T
    Bksum = np.sum(Bk,axis=1)
    Mj = np.ones(shape=(polyorder+1,nnpts))/(eef/mmodel)**2
    Mj*=t_pow.T
    t_pow_3d = tnew[:,np.newaxis,np.newaxis]**order_pow
    Mjk = t_pow_3d.T * Mj[np.newaxis,:,:]
    Mjksum = np.sum(Mjk,axis=2)
    try:
        Aj = np.linalg.lstsq(Mjksum,Bksum)[0]
        pol = np.polyval(Aj[::-1],t-np.mean(t))
    except:
        pol = np.ones(len(t))
    return pol

def interp_eclipses(time, flux, clip, wn=0.):
    """Masks out the eclipses and linearly interpolates across eclipses while adding
    white noise (wn). Returns flux array"""
    continuum = np.arange(len(time))
    continuum = np.delete(continuum, clip)
    if wn > 0:
        wn = np.random.normal(0, wn, len(time[clip]))
    flux[clip] = np.interp(time[clip], time[continuum], flux[continuum]) + wn
    return flux

def interp_poorquality(time, flux, quality, qual_tol = 8, ret_mask = False):
    poorquality = (quality > qual_tol)
    if ret_mask:
        return poorquality
    flux_new = flux.copy()
    flux_new[poorquality] = np.interp(time[poorquality], time[~poorquality], flux[~poorquality])
    return flux_new

def find_jumps(cadnum, t, f, df, cadtol = 20, ftol = 10., plot=False):
    """Find large gaps in data (given some cadence tolerance) and returns
    indices of starting points of each chunk."""
    gaps = np.where(np.diff(cadnum)>cadtol)[0]+1
    gaps = np.insert(gaps, 0, 0)
    gaps = np.append(gaps, len(t))
    fluxjump = np.where((np.diff(f)>np.median(df)*ftol) * \
                (np.diff(t)>1))[0]+1
    gaps = np.unique(np.concatenate((gaps, fluxjump)))
    if plot:
        plt.figure()
        plt.plot(cadnum, f, 'k.')
        plt.plot(cadnum[gaps[:-1]], f[gaps[:-1]], 'co')
        plt.show()
    return gaps

def autocorr(x):
    """Computes the autocorrelation function of input x"""
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def compute_a(period, msum, unit_au = True):
    """Computes the semi-major axis with given period and 
    msum. Returns a in AU if unit_au is True, otherwise 
    returns a in solar radii."""
    a = ((period/d2y)**2 * (msum))**(1./3.)
    if unit_au:
        return a
    return a / r2au 

def chi(model, data, error):
    """Computes the chi-square value"""
    chisq = np.sum(((model-data)/error)**2)
    return chisq
    
def kiclookup(kic, target):
    """Returns the index of corresponding KIC # in target array"""
    idx = np.where(target.astype(int) == kic)[0]
    if len(idx)==0:
        print "No data matched to your input KIC number, double check it!"
        return None
    return idx
    
def checkephem(t, f, ftol, period, plot=False):
    """Need to update this..."""
    pe = np.where((f<np.roll(f,1)) & (f<np.roll(f, -1)) & (f<ftol))[0]
    pe_times = t[pe].copy()
    ws = np.where(np.diff(t[pe])<0.2*period)[0] #w-shaped eclipses (due to starspot crossing?)
    for ii in range(len(ws)):
        pe_times[ws[ii]] = np.mean(pe_times[ws[ii]:ws[ii]+2])
        pe_times = np.delete(pe_times, ws[ii]+1)
    jumps = np.where(np.diff(pe_times)>period*1.2)[0] #fill in gaps in data
    for ii in range(len(jumps)):
        pe_times = np.insert(pe_times, jumps[len(jumps)-1-ii]+1, np.zeros(int(np.round(np.diff(pe_times)/period, 0)[jumps[len(jumps)-1-ii]])-1))
    bad = (pe_times == 0.)
    pe_inds = np.arange(len(pe_times))
    ephem = np.polyfit(pe_inds[~bad], pe_times[~bad], 1)
    if plot:
        plt.figure(figsize=(9, 6))
        plt.suptitle('KIC '+str(kic))
        plt.subplot2grid((2,2), (0,0), colspan=2)
        plt.plot(t, f, 'ko')
        plt.plot(pe_times[~bad], np.ones_like(pe_times[~bad]),
                 marker='v', linewidth=0, color='cyan')
        plt.xlabel('time (d)')
        plt.ylabel('kepler flux')
        plt.xlim(t[0], t[-1])
        plt.subplot2grid((2,2), (1,0))
        plt.plot(pe_inds[~bad], pe_times[~bad], 'ko', pe_inds, ephem[0]*pe_inds+ephem[1], 'r-')
        plt.xlabel('PE event #')
        plt.ylabel('time (d)')
        plt.subplot2grid((2,2), (1,1))
        plt.plot(pe_inds[~bad], ephem[0]*pe_inds[~bad]+ephem[1]-pe_times[~bad], 'go')
        plt.xlabel('PE event #')
        plt.ylabel('ephemeris - observed times of PE (d)')
        plt.show()
    return pe_times, ephem

def sudarsky(theta, e, period):
    tt = (-np.sqrt(1.-e**2) * period / TWOPI) * \
            (e*np.sin(theta)/(1.+e*np.cos(theta)) - 2.*(1.-e**2)**(-0.5) * \
            np.arctan(np.sqrt(1.-e**2) * np.tan((theta)/2.) / (1.+e)))
    return tt


def check_dir_exists(path):
    if os.path.isdir(path):
        print "Directory already exists."
    else:
        os.makedirs(path)
        print "Directory made: ", path
    return

def downsample_cad(*args, **kwargs):
    Nexp = kwargs.pop('exp', 30)
    as_strided = np.lib.stride_tricks.as_strided
    res = []
    for count, thingie in enumerate(args):
        _x = as_strided(thingie, (len(thingie)+1-Nexp, Nexp), (thingie.strides * 2))
        res.append(np.mean(_x, axis=1))
    return res

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')
    
def moving_average(x, N, method='cumsum'):
    if method == 'cumsum':
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 
    elif method == 'convolve':
        window = np.ones(int(N))/float(N)
        return np.convolve(x, window, 'valid')
    
def gr_B(chains):
    """Computes between-chains variance B = N/(M-1) \sum_{m=1}^{M} (T_m-T)^2""" 
    M, N = chains.shape
    theta_m = np.nanmean(chains, axis=1)
    return N/(M-1.) * np.sum((theta_m-np.sum(theta_m)/M)**2)

def gr_W(chains):
    """Computes within-chain variance W = 1/M \sum_{m=1}^{M} \sigma_m^2"""
    M, N = chains.shape
    sigma_m = np.nanvar(chains, axis=1)
    return 1./M * np.sum(sigma_m**2)

def gr_PSRF(B, W, N):
    """Computes potential scale reduction factor (PSRF), the Gelman-Rubin statistic"""
    return (1-1./N)*W + 1/N * B

def gr_R(V, W):
    return np.sqrt(V/W)

def gelman_rubin(chains):
    M, N = chains.shape
    B = gr_B(chains)
    W = gr_W(chains)
    V = gr_PSRF(B, W, N)
    return gr_R(V, W)


def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def get_flicker(x, mad=False, window=16):
    smoothed_x = smooth(x, window_len=window)[window/2:-window/2+1]
    if mad:
        flckr = np.nanmedian(abs(x - smoothed_x))
    else:
        flckr = np.nanstd(x - smoothed_x)
    return smoothed_x, flckr