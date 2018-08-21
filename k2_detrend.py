import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import fits
from everest import Everest
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import scipy.optimize
#import triangle
from helper_funcs import *
import sys
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

def peprof(pars, x, data=None, error=None, porder=1, scalar=True):
    c0, c1, x0, d, tau = pars
    model = c0 - c1 * (1. - (1. - np.exp(1. - np.cosh((x-x0)/d)))**tau)
    ooe = (model==1.)
    if data is not None and ooe.sum() > porder:
        #print x[~(model<c0)].shape, data[~(model<c0)].shape
        coeffs = np.polyfit(x[ooe], data[ooe]/model[ooe], porder)
        poly = np.poly1d(coeffs)(x)
    else:
        poly = np.ones(len(model))
    if scalar:
        return np.sum(((data/model - poly)/(error/model))**2)
    return model, poly

def peprof2(pars, fixed_pars, x, data=None, error=None, porder=1, scalar=True):
    c0, c1, x0 = pars
    d, tau = fixed_pars
    model = c0 - c1 * (1. - (1. - np.exp(1. - np.cosh((x-x0)/d)))**tau)
    ooe = (model==1.)
    if data is not None and ooe.sum() > porder:
        #print x[~(model<c0)].shape, data[~(model<c0)].shape
        coeffs = np.polyfit(x[ooe], data[ooe]/model[ooe], porder)
        poly = np.poly1d(coeffs)(x)
    else:
        poly = np.ones(len(model))
    if scalar:
        return np.sum(((data/model - poly)/(error/model))**2)
    return model, poly

def fit_pe2(prev_fit, ii, tpe, period, pwid, x, y, ey, porder=1):
    pars, seg, bounds = init_fit(ii, tpe, period, pwid, x, y, ey)
    if np.any(np.isinf(pars)):
        return -np.inf, seg
    res = scipy.optimize.minimize(peprof2, pars[:3], args=(prev_fit, x[seg], y[seg], ey[seg]), bounds=bounds[:-2])    
    return res.x, seg

def init_fit(ii, tpe, period, pwid, x, y, ey):
    seg = (abs(x-(tpe+period*ii))<pwid*3)
    if seg.sum()<= 5:
        return -np.inf, seg, -np.inf
    baseline = np.nanmedian(y[seg])
    ymax = np.nanmax(y[seg])
    ymin = np.nanmin(y[seg])
    dep = ymax-ymin
    bounds = ((ymin, ymax), (min(0.5*dep, 0.01), min(1.5*dep, 0.9)), (x[seg][0], x[seg][-1]), (pwid*0.1, pwid), (0, 3))
    return np.array([baseline, dep, tpe+period*ii, pwid/4., 1.]), seg, bounds

def fit_pe(ii, tpe, period, pwid, x, y, ey, porder=1):
    pars, seg, bounds = init_fit(ii, tpe, period, pwid, x, y, ey)
    if np.any(np.isinf(pars)):
        return -np.inf, seg
    res = scipy.optimize.minimize(peprof, pars, args=(x[seg], y[seg], ey[seg]), bounds=bounds)    
    if res.success:
        return res.x, seg
    else:
        res = scipy.optimize.minimize(peprof, pars, args=(x[seg], y[seg], ey[seg]), method='Nelder-Mead')
    return res.x, seg

def during_gap(ii, tpe, period, pwid, x, chunks):
    mask = (abs((period*ii+tpe)-x[chunks[:-1]])<pwid/2.)
    mask = mask | (abs((period*ii+tpe)-x[chunks[1:]-1])<pwid/2.)
    ingap = np.array([(((period*ii+tpe)>=x[chunks[1:]-1][:-1][jj]) * ((period*ii+tpe)<=x[chunks[:-1]][1:][jj])) for jj in range(len(chunks)-2)])
    return mask.sum()+ingap.sum()

chunks = identify_gaps(keblat.cadnum, quarts=keblat.quarter, retbounds_inds=True)

t, f, ef = keblat.jd, keblat.flux, keblat.fluxerr
Neclipses = int((t[-1]-t[0])/period)+1
obs_times = np.zeros((Neclipses, 5))
for ii in range(Neclipses):
    res, seg = fit_pe(ii, tpe, period, keblat.pwidth*period, t, f, ef)
    if np.any(np.isinf(res)) or (during_gap(ii, tpe, period, keblat.pwidth*period, t, chunks)>0):
        obs_times[ii,:] = [np.nan]*5
    else:
        obs_times[ii,:] = res

etvs = np.zeros(Neclipses)*np.nan
fixed_pars = np.nanmedian(obs_times, axis=0)[-2:]

for ii in np.arange(Neclipses)[~np.isnan(obs_times[:,2])]:
    res, seg = fit_pe2(fixed_pars, ii, tpe, period, keblat.pwidth*period, t, f, ef)
    etvs[ii] = res[2]

Neclipse = np.arange(len(etvs))
bad = np.isnan(etvs)
Neclipse = Neclipse[~bad]
etvs = etvs[~bad]
ephem = np.poly1d(np.polyfit(Neclipse, etvs, 1))(Neclipse)
ephem2 = np.poly1d(np.polyfit(np.arange(len(times_of_transits_pub)), times_of_transits_pub-_tbary, 1))(np.arange(len(times_of_transits_pub)))


def lcprofile(pars, x=None, data=None, error=None, porder=0, se=True, scalar=True):
    #print pars, x.shape, data.shape, porder, se, scalar
    pe_dur = 0
    se_dur = 0
    # plt.figure()
    # plt.plot(x, data, 'k-x')
    if se:
        c0, c1, x0, d, tau, c2, x2, d2, tau2 = pars
        model = c0 - c1 * (1. - (1. - np.exp(1. - np.cosh((x-x0)/d)))**tau)
        pe = (abs(model-c0) > 1e-5)
        if pe.sum()>1:
            pe_dur = np.nanmax(x[pe]) - np.nanmin(x[pe])
	    # plt.plot(x[pe], data[pe], 'r.')
	    # plt.plot(x[pe][[0,-1]], data[pe][[0,-1]], 'go')
        model = model - c2 * (1. - (1. - np.exp(1. - np.cosh((x-x2)/d2)))**tau2)
        se = (abs(model-c0) > 1e-5) * ~pe
        if se.sum()>1:
            se_dur = np.nanmax(x[se])- np.nanmin(x[se])
    else:
        c0, c1, x0, d, tau = pars
        model = c0 - c1 * (1. - (1. - np.exp(1. - np.cosh((x-x0)/d)))**tau)
    if porder>0:
        #print x[~(model<c0)].shape, data[~(model<c0)].shape
        coeffs = np.polyfit(x, data/model, porder)
        poly = np.poly1d(coeffs)(x)
    else:
        poly = np.ones(len(model))
    if scalar:
        return np.sum(((data/model - poly)/(error/model))**2)
    return model, poly, pe_dur, se_dur


###############################################################################################
################### compute approximate eclipse times, depths, and durations ##################
###############################################################################################

if not os.path.isfile('k2/rod.lcprofs'):
	# tse_arr = np.append(lcprofile_pars[6], tpe_arr[~(abs(tpe_arr-lcprofile_pars[2])<lcprofile_pars[3]*2)])
	# lcprofile_pars0 = lcprofile_pars
	# lcbounds[2] = (np.clip(lcprofile_pars[2]-0.01, 0, 1), np.clip(lcprofile_pars[2]+0.01, 0, 1))
	# for ii in range(len(tse_arr)):
	# 	lcprofile_pars0[6] = tse_arr[ii]
	# 	lcbounds[6] = (np.clip(tse_arr[ii]-0.1, 0, 1), np.clip(tse_arr[ii]+0.1, 0, 1))
	# 	result = scipy.optimize.minimize(lcprofile, lcprofile_pars0, method='L-BFGS-B',
	# 									 args=(keblat.jd%period/period, keblat.flux, keblat.fluxerr),
	# 									 bounds=lcprofile_bounds)
	# 	current_chi2 = lcprofile(result.x, x=keblat.jd%period/period, data=keblat.flux, error=keblat.fluxerr, se=True, scalar=True)
	# 	if current_chi2<bestlcchi2:
	# 		bestlcchi2 = current_chi2
	# 		lcprofile_pars = result.x
	#
	# mod, poly = lcprofile(lcprofile_pars, x=keblat.jd%period/period, data=keblat.flux, porder=0,
	# 					  error=keblat.fluxerr,
	# 					  se=True, scalar=False)
	# plt.figure()
	# plt.plot(keblat.jd%period/period, keblat.flux, 'k.')
	# plt.plot(keblat.jd%period/period, mod*poly, 'r.')
	# plt.plot(keblat.jd%period/period, poly, 'g.')
	# plt.savefig('k2/'+str(kic)+'.png')
	# plt.close()
	# np.savetxt('k2/'+str(kic)+'.lcprof', np.array([kic, period, lcprofile_pars[2]*period, lcprofile_pars[6]*period,
	# 										 lcprofile_pars[1], lcprofile_pars[5],
	# 										 lcprofile_pars[3]*period, lcprofile_pars[7]*period]).reshape((1,-1)))

	fout = open('k2/rod.lcprofs', 'w')
	fout.close()


	rod_in = np.loadtxt('k2/lcprofiles.list').astype(int)

	fout = open('k2/rod.lcprofs', 'a')
	for ii in range(len(rod_in)):
		test = np.loadtxt('k2/may11_2016/'+str(rod_in[ii])+'.lcprof')
		k2lcfname = glob.glob('/astro/users/windemut/keblat/data/k2/*'+str(rod_in[ii])+'*')[0]
		_jd, _f = np.loadtxt(k2lcfname, delimiter=',', usecols=(0,1), unpack=True, skiprows=1)
		period = test[1]
		tmp = np.array([1., test[4], test[2]/period, test[6]/period, 1.0, test[5], test[3]/period, test[7]/period, 1.0])
		if rod_in[ii] == 211489484:
			tmp[3] = 0.0005
			tmp[-3] = 0.7562
			tmp[-2] = 0.0005
		if rod_in[ii] == 211920612:
			tmp[-2] = 0.019
		if rod_in[ii] == 203610780:
			tmp[-2] = 0.0035
		x = _jd%period/period
		x = np.append(x, x+1)
		f = np.append(f, f)
		#_f = _f[np.argsort(x)]
		#x = x[np.argsort(x)]
		result = scipy.optimize.minimize(lcprofile, tmp, args=(x, _f, _f*0 + 1.2*np.median(abs(np.diff(_f)))))
		mod, poly, pe_dur, se_dur = lcprofile(result.x, x=x, data=_f, porder=0,
							  error=_f*0 + 1.2*np.median(abs(np.diff(_f))),
							  se=True, scalar=False)
		plt.figure()
		plt.plot(x, _f, 'k.')
		plt.plot(x, poly, 'g.', alpha=0.1)
		plt.plot(x, mod*poly, 'r.')
		plt.ylim((0, 1.5))
		plt.savefig('k2/'+str(rod_in[ii])+'_DELETE_.png')
		plt.close()
		#plt.show()
		fout.write("""{0}\n""".format(" ".join([str(j) for j in [rod_in[ii], period, result.x[2]*period, result.x[6]*period,
												 result.x[1], result.x[5], pe_dur, se_dur]])))
	fout.close()


###############################################################################################
#################### detrend each EPIC light curve with everest ...  ##########################
###############################################################################################
print blah
epic, period, tpe, tse, pdepth, sdepth, pwidth, swidth = np.loadtxt('k2/rod.lcprofs',
																unpack=True)
#kic;period;bjd0;pdepth;sdepth;pwidth;swidth;sep;morph;RA;DEC;kmag;Teff;Teff_Pin;Teff_Casa;SC

badlist = []
for i in range(len(epic)):
	mult=1.25
	print "Detrending {0}".format(int(epic[i]))
	star = Everest(int(epic[i]))
	fail = True
	niter=0
	while fail and niter<5:
		try:
			star.set_mask(transits = [(period[i], tpe[i], pwidth[i]*period[i]),
										(period[i], tse[i], swidth[i]*period[i])])
			if star.crwdflag == 5:
				break
			if (star.crwdflag <= 2) and (star.satflag <= 2):
				fail = False
			else:
				print("bad crowding values {0}, {1}".format(star.crwdflag, star.satflag))
				fail = True
				mult -= 0.1
		except:
			mult -= 0.1
			fail = True
		niter+=1
	print mult, niter
	star.plot()
	#plt.show()
	#print("Save info for EPIC {0}?".format(int(epic[i])))
	# response = raw_input("Save?")
	# if response == 'y':
	if (star.crwdflag + star.satflag < 5):#<= 2) and (star.satflag <= 2):
		np.savez('k2/'+str(int(epic[i]))+'.npz', time = star.time, flux = star.flux, raw_flux = star.raw_flux, raw_fluxerr = star.raw_ferr)
	else:
		badlist.append(int(epic[i]))
	plt.close()
	# 	print "Saved."
	# else:
	# 	print "Not saved."
