import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lmfit
if lmfit.__version__[2] != '9':
    print "Version >= 0.9 of lmfit required..."
    sys.exit()
from lmfit import minimize, Parameters, report_fit
import scipy.optimize, scipy.special
import numpy as np
try:
    from keblat import *
except:
    print('Exception from keblat import *')
    import datetime
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    print('Current dir: {0}'.format(os.getcwd()))
    print('keblat.py exists? {0}'.format(os.path.isfile('keblat.py')))

from helper_funcs import *
from emcee import autocorr as autoc

#parnames_dict = {'lc': ['msum', 'rsum', 'rrat', 'period', 'tpe', 'esinw', 'ecosw', 'b', 'frat', 'q1',
#                        'q2', 'q3', 'q4'],
#                 'sed': ['m1', 'm2', 'z0', 'age', 'dist', 'ebv', 'h0', 'isoerr'],
#                 'rv': ['msum', 'mrat', 'period', 'tpe', 'esinw', 'ecosw', 'inc', 'k0', 'rverr'],
#                 'lcsed': ['m1', 'm2', 'z0', 'age', 'dist', 'ebv', 'h0', 'period', 'tpe','esinw',
#                           'ecosw', 'b', 'q1', 'q2', 'q3', 'q4', 'lcerr', 'isoerr'],
#                 'lcrv': ['msum', 'mrat', 'rsum', 'rrat', 'period', 'tpe', 'esinw', 'ecosw', 'b', 'frat',
#                          'q1', 'q2', 'q3', 'q4', 'lcerr', 'k0', 'rverr']}



def get_pars2vals(fisopars, partype='lc', qcrow=None):
    try:
        parnames = parnames_dict[partype][:]
    except:
        print("You entered: {0}. Partype options are 'lc', 'sed', 'rv', 'lcsed', 'lcrv'. Try again.".format(partype))
        return
    if qcrow is not None:
        parnames += ['cr' + str(ii) for ii in qcrow]
#    print parnames
    parvals = np.zeros(len(parnames))
    novalue = (len(fisopars) == len(parnames))
    #print fisopars, type(fisopars), parnames
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

def rez(fit_params, keblat, qcrow=None, polyorder=0, ooe=True):
#    print len(fit_params), qcrow
    guess = get_pars2vals(fit_params, partype='lc', qcrow=qcrow)
    if len(guess)>13:
        crowd_fits = keblat.broadcast_crowd(keblat.quarter, guess[-len(np.unique(keblat.quarter)):])
    else:
        crowd_fits = keblat.crowd
    lcmod, lcpol = keblat.lcfit(guess[:13], keblat.jd[keblat.clip],
                                keblat.quarter[keblat.clip], keblat.flux[keblat.clip],
                                keblat.fluxerr[keblat.clip], crowd_fits[keblat.clip],
                                polyorder=polyorder, ooe=ooe)
    if np.any(np.isinf(lcmod)):
        return np.ones(np.sum(keblat.clip))*1e10#(1.-keblat.flux[keblat.clip])/keblat.dflux[keblat.clip]
    if keblat.pe_dur == 0 or keblat.se_dur == 0:
        return (keblat.flux[keblat.clip] - lcmod) / keblat.fluxerr[keblat.clip]
    return (keblat.flux[keblat.clip] - lcmod*lcpol)/keblat.fluxerr[keblat.clip]

def lnlike_rv(fisopars, keblat, residual=True):
    pars = get_pars2vals(fisopars, partype='rv')
    rv1, rv2 = keblat.rvfit(pars, keblat.rv_t)
    res = np.concatenate(((rv1[~keblat.bad1] - keblat.rv1_obs[~keblat.bad1]) /
                          np.sqrt(keblat.rv1_err_obs[~keblat.bad1]**2 + pars[-1]**2),
                          (rv2[~keblat.bad2] - keblat.rv2_obs[~keblat.bad2]) /
                          np.sqrt(keblat.rv2_err_obs[~keblat.bad2]**2 + pars[-1]**2)))
    if residual:
        if np.any(np.isinf(res)) or np.sum(np.isnan(res)) > 0.05 * len(res):
            return np.ones(np.sum(~keblat.bad1)+np.sum(~keblat.bad2)) * 1e20
        return res
    return np.sum(res**2)

def lnlike_lcrv(fisopars, keblat, qua=[1], polyorder=2, residual=True, retro=False):
    pars = get_pars2vals(fisopars, partype='lcrv')
    res = keblat.lnlike_lcrv(pars, qua=qua, polyorder=polyorder, residual=residual, retro=retro)
    if residual:
        if np.any(np.isinf(res)) or np.sum(np.isnan(res)) > 0.05 * len(res):
            return np.ones(np.sum(~keblat.bad1)+np.sum(~keblat.bad2) + np.sum(keblat.clip)) * 1e20
        return res
    return np.sum(res**2)

def lnlike_lmfit(fisopars, keblat, lc_constraints=None, ebv_arr=None, qua=[1], polyorder=2, residual=False):
    allpars = get_pars2vals(fisopars, partype='lcsed')

    if ebv_arr is not None:
        allpars[5] = np.interp(allpars[4], ebv_dist, ebv_arr)
    #print "sum of clips = ", keblat.clip.sum()
    res = keblat.lnlike(allpars, lc_constraints=lc_constraints, qua=qua, polyorder=polyorder,
                         residual=residual)
    if np.any(np.isinf(res)):
        print "Inf res"
        extra = 0 if lc_constraints is None else len(lc_constraints)
        return np.ones(len(keblat.magsobs)+len(keblat.flux[keblat.clip]) + extra)*1e20
    bads = np.isnan(res)
    if np.sum(bads) > 0.05*len(res):
        print "Seems to be a lot of Nans..."
        extra = 0 if lc_constraints is None else len(lc_constraints)
        return np.ones(len(keblat.magsobs)+len(keblat.flux[keblat.clip]) + extra)*1e20
    return res

def ew_search_lmfit(ew_trials, keblat, pars0, argpars, fit_ecosw=True, polyorder=0):
    fit_ew = Parameters()
    fit_ew.add('esinw', value=pars0[5], min=-0.9, max=0.9, vary=True)
    fit_ew.add('ecosw', value=pars0[6], min=-0.9, max=0.9, vary=fit_ecosw)
    chisq = 1e18
    for ii in range(len(ew_trials)):
        fit_ew['esinw'].value=ew_trials[ii][0]
        fit_ew['ecosw'].value=ew_trials[ii][1]
        result = minimize(ew_search, fit_ew, args=(keblat, pars0), 
                          kws={'argpars':argpars, 'polyorder':polyorder})
        if result.redchi < chisq or ii==0:
            chisq = result.redchi * 1.0
            ew_best = result.params['esinw'].value, result.params['ecosw'].value
            print "Better redchi: ", chisq, result.redchi, ew_best
    return ew_best

def ew_search(ew, keblat, pars0, argpars=None, polyorder=0, retmod=False):
    pars = np.array(pars0).copy()
    #esinw, ecosw = ew
    try:
        pars[5], pars[6] = ew['esinw'].value, ew['ecosw'].value
    except:
        pars[5], pars[6] = ew[0], ew[1]
    mod, poly = keblat.lcfit(pars, keblat.jd, keblat.quarter, keblat.flux, keblat.dflux, keblat.crowd, polyorder=polyorder)
    # _phasesort = np.argsort(keblat.phase)
    # _phase = keblat.phase[_phasesort]
    # _flux = keblat.flux[_phasesort]
    # _dflux = keblat.dflux[_phasesort]
    # Nbins = int(1./min(keblat.pwidth, keblat.swidth))*4
    # bins = np.linspace(_phase[0], _phase[-1], Nbins)
    # digitized = np.digitize(_phase, bins)
    # x_means = [_phase[digitized == i].mean() for i in range(1, len(bins))]
    # y_means = [_flux[digitized == i].mean() for i in range(1, len(bins))]
    # y_means /= y_means
    #
    if np.any(np.isinf(mod)):
        return np.ones_like(keblat.jd+1)*1e10
    if retmod:
        return mod, poly
    return np.append((keblat.flux - mod*poly)/keblat.dflux, tse_residuals((pars[5],pars[6]), *argpars))


def get_age_trials(mass):
    if np.log10(mass) <= 0.02:
        return np.log10(np.exp(np.linspace(np.log(3e6), np.log(7e9), 5)))
    else:
        agelim = -2.7 * np.log10(mass) + 9.9
        return np.log10(np.exp(np.linspace(np.log(3e6), np.log(10**agelim), 5)))

def sed_chisq(sedpars0, keblat, lc_constraints):
    # ll,_ = keblat.ilnlike(sedpars0, lc_constraints=lc_constraints)
    # return ll/(-0.5)
    residuals = keblat.ilnlike(sedpars0, lc_constraints=lc_constraints, residual=True)
    if np.all(residuals)==1e12:
        return residuals
    residuals[:-3] *= np.sqrt(keblat.emagsobs**2+np.exp(sedpars0[-1])**2)
    lc_inputs = np.array([(keblat.r1+keblat.r2)/(sedpars0[0]+sedpars0[1])**(1./3.),
                          keblat.r2/keblat.r1, keblat.frat])

    residuals[-3:] = np.log(lc_constraints) - np.log(lc_inputs)
    #residuals = np.append(residuals, lc_inputs)
    #residuals = np.append(residuals, lc_inputs)
    return residuals

def opt_sed(keblat, **kwargs):
    fit_ebv = kwargs.pop('fit_ebv', True)
    ret_lcpars = kwargs.pop('ret_lcpars', True)
    vary_z0 = kwargs.pop('vary_z0', True)
    lc_constraints = kwargs.pop('lc_constraints', None)
    ebv_arr = kwargs.pop('ebv_arr', None)
    ebv_dist = kwargs.pop('ebv_dist', None)
    vary_age = kwargs.pop('vary_age', True)
    vary_msum = kwargs.pop('vary_msum', True)
    vary_dist = kwargs.pop('vary_dist', True)
    xtol = kwargs.pop('xtol', 1e-8)
    ftol = kwargs.pop('ftol', 1e-8)

    keblat.updatepars(**kwargs)
    
    fit_params2 = Parameters()

    for name, val in kwargs.items():
        if name in keblat.parbounds.keys():
            fit_params2.add(name, value=val, min=keblat.parbounds[name][0],
                            max=keblat.parbounds[name][1], vary=False)
        else:
            fit_params2.add(name, value=val, vary=False)
    
    fit_params2['msum'].vary=vary_msum
    fit_params2['mrat'].vary=True
    fit_params2['z0'].vary=vary_z0
    fit_params2['age'].vary=vary_age
    fit_params2['dist'].vary=vary_dist
    fit_params2['ebv'].vary=fit_ebv
    kws = {'lc_constraints': lc_constraints, 'residual': True, 
           'ebv_arr': ebv_arr, 'ebv_dist': ebv_dist}

    fit_kws={'maxfev':2000*(len(fit_params2)+1), 'xtol':xtol, 'ftol':ftol}
    if len(keblat.magsobs)<6:
        fit_params2['ebv'].vary=False

    print "=========================================================================="
    print "======================== Starting SED ONLY fit... ========================"
    print "=========================================================================="

    result2 = minimize(keblat.ilnlike, fit_params2, kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
    isores = keblat.ilnlike(result2.params, lc_constraints=lc_constraints, residual=True)
    redchi2 = np.sum(isores**2) / len(isores)
    print redchi2, result2.redchi
    report_fit(result2)
    fit_params2 = result2.params.copy()

    # fit_params2['z0'].vary=True
    # result2 = minimize(keblat.ilnlike, fit_params2, kws=kws, **fit_kws)
    # isores = keblat.ilnlike(result2.params, lc_constraints=lc_constraints, residual=True)
    # current_redchi2 = np.sum(isores**2) / len(isores)
    # if current_redchi2 < redchi2:
    #     print current_redchi2, result2.redchi
    #     report_fit(result2)
    #     print result2.message
    #     fit_params2 = result2.params.copy()
    #     redchi2 = current_redchi2*1.

    niter=0
    while (redchi2>1.) and (niter<3):
        result2 = minimize(keblat.ilnlike, fit_params2, kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
        #redchi2_0 = keblat.ilnlike(result2.params, lc_constraints=lc_constraints)
        isores = keblat.ilnlike(result2.params, lc_constraints=lc_constraints, residual=True)
        current_redchi2 = np.sum(isores**2) / len(isores)
        print "Iteration: ", niter, current_redchi2, result2.redchi, result2.nfev
        if current_redchi2 < redchi2:
            print "Saving the following results:"
            report_fit(result2)
            redchi2 = current_redchi2*1.0
            fit_params2 = result2.params.copy()
        niter+=1

    #print result2.params, fit_params2
    if ret_lcpars:
        return fit_params2, keblat.r1, keblat.r2, keblat.frat
    return fit_params2


def opt_sed_old(sedpars0, keblat, lc_constraints, ebv_dist, ebv_arr, fit_ebv=True, ret_lcpars=True, vary_z0=True):
    m1, m2, z0, age, dist, ebv, h0, isoerr = sedpars0
    fit_params2 = Parameters()
    fit_params2.add('m1', value=m1, min=0.1, max=12.)
    fit_params2.add('m2', value=m2, min=0.1, max=12.)
    fit_params2.add('z0', value=z0, min=0.001, max=0.06, vary=vary_z0)
    fit_params2.add('age', value=age, min=6.0, max=10.1, vary=True)
    fit_params2.add('dist', value=dist, min=10., max=15000.)
    kws = {'lc_constraints': lc_constraints, 'residual': True, 'ebv_arr': ebv_arr, 'ebv_dist': ebv_dist}
    if fit_ebv:
        fit_params2.add('ebv', value=ebv, min=0.0, max=1.0)#ebv[0], vary=False)
        kws['ebv_arr'] = None
        kws['ebv_dist'] = None
    fit_params2.add('h0', value=h0, vary=False)
    fit_params2.add('isoerr', value=isoerr, min=-8, max=0.,
                    vary=False)
#    fit_params2.add('mrat', expr='m2/m1', max=1.0)
#    fit_params2.add('mbound', expr='m1-m2>0')

#    if isoerr>0.1:
#    fit_params2['isoerr'].vary=True

    fit_kws={'maxfev':2000*(len(fit_params2)+1)}
    if len(keblat.magsobs)<6:
        fit_params2['ebv'].vary=False

    print "=========================================================================="
    print "======================== Starting SED ONLY fit... ========================"
    print "=========================================================================="

    result2 = minimize(keblat.ilnlike, fit_params2, kws=kws, iter_cb=MinimizeStopper(30), **fit_kws)
    isores = keblat.ilnlike(result2.params, lc_constraints=lc_constraints, residual=True)
    redchi2 = np.sum(isores**2) / len(isores)
    print redchi2, result2.redchi
    report_fit(result2)
    fit_params2 = result2.params.copy()

    # fit_params2['z0'].vary=True
    # result2 = minimize(keblat.ilnlike, fit_params2, kws=kws, **fit_kws)
    # isores = keblat.ilnlike(result2.params, lc_constraints=lc_constraints, residual=True)
    # current_redchi2 = np.sum(isores**2) / len(isores)
    # if current_redchi2 < redchi2:
    #     print current_redchi2, result2.redchi
    #     report_fit(result2)
    #     print result2.message
    #     fit_params2 = result2.params.copy()
    #     redchi2 = current_redchi2*1.

    niter=0
    while (redchi2>1.) and (niter<3):
        result2 = minimize(keblat.ilnlike, fit_params2, kws=kws, iter_cb=MinimizeStopper(30), **fit_kws)
        #redchi2_0 = keblat.ilnlike(result2.params, lc_constraints=lc_constraints)
        isores = keblat.ilnlike(result2.params, lc_constraints=lc_constraints, residual=True)
        current_redchi2 = np.sum(isores**2) / len(isores)
        print "Iteration: ", niter, current_redchi2, result2.redchi, result2.nfev
        if current_redchi2 < redchi2:
            print "Saving the following results:"
            report_fit(result2)
            redchi2 = current_redchi2*1.0
            fit_params2 = result2.params.copy()
        niter+=1

    #print result2.params, fit_params2
    if ret_lcpars:
        return fit_params2, keblat.r1, keblat.r2, keblat.frat
    return fit_params2


def opt_lc_old(lcpars0, jd, phase, flux, dflux, crowd, clip, set_upperb=2., fit_crowd=False, fit_se=False,
           vary_msum=True):
    msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4 = lcpars0

    fit_params = Parameters()
    fit_params.add('esinw', value=esinw, min=-.999, max=0.999, vary=False)
    fit_params.add('ecosw', value=ecosw, min=-.999, max=0.999, vary=False)#ecosw-0.05, max=ecosw+0.05, vary=False)
    fit_params.add('rsum', value=rsum, min=0.1, max=10000., vary=False)
    fit_params.add('rrat', value=rrat, min=1e-4, max=1e3, vary=False)
    fit_params.add('b', value=b, min=0., max=set_upperb, vary=False)
    fit_params.add('frat', value=frat, min=1e-6, max=1e2, vary=False)
    fit_params.add('msum', value=msum, min=0.2, max=24., vary=False)

    fit_params.add('period', value=period, min=period-0.005, max=period+0.005, vary=False)
    fit_params.add('tpe', value=tpe, min=tpe-10., max=tpe+10., vary=False)
    fit_params.add('q1', value=q1, min=0., max=1., vary=False)
    fit_params.add('q2', value=q2, min=0., max=1., vary=False)
    fit_params.add('q3', value=q3, min=0., max=1., vary=False)
    fit_params.add('q4', value=q4, min=0., max=1., vary=False)

    fit_params['rrat'].vary=True
    fit_params['rsum'].vary=True
    fit_params['b'].vary=True
    fit_params['frat'].vary=True
    fit_params['esinw'].vary=True
    fit_params['ecosw'].vary=True
    fit_params['tpe'].vary=True

    fit_kws={'maxfev':100*(len(fit_params)+1)}

    if fit_crowd:
        print "Fitting crowding parameters..."
        for ii in fit_params.keys():
            fit_params[ii].vary=True
        for ii in range(len(np.unique(keblat.quarter))):
            fit_params.add('cr'+str(np.unique(keblat.quarter)[ii]), value=keblat._crowdsap[ii], min=0.1, max=1.0)
        result0 = minimize(rez, fit_params, kws={'polyorder':2}, iter_cb=MinimizeStopper(10), **fit_kws)
        report_fit(result0)

        for ii in result0.params.keys():
            result0.params[ii].vary=True
        niter=0
        redchi2 = np.sum((rez(get_pars2vals(result0.params, partype='lc'), polyorder=2))**2) / np.sum(keblat.clip)
        guess=get_pars2vals(result0.params, partype='lc')
        while (redchi2>1.) and (niter<5):
            result0 = minimize(rez, result0.params, kws={'polyorder':2}, iter_cb=MinimizeStopper(10), **fit_kws)
            current_chi = np.sum((rez(get_pars2vals(result0.params, partype='lc'), polyorder=2))**2) / np.sum(keblat.clip)
            if current_chi < redchi2:
                redchi2=current_chi*1.0
                report_fit(result0)
                guess=get_pars2vals(result0.params, partype='lc')
            niter+=1

        return guess

    print "=========================================================================="
    print "==================== Starting LIGHTCURVE ONLY fit... ====================="
    print "=========================================================================="
    if fit_se:
        if ecosw == 0.:
            ecosw=1e-5
        if esinw == 0.:
            esinw=1e-5
        fit_params['esinw'].min=esinw-0.1*abs(esinw)
        fit_params['ecosw'].min=ecosw-0.05*abs(ecosw)
        fit_params['esinw'].max=esinw+0.1*abs(esinw)
        fit_params['ecosw'].max=ecosw+0.05*abs(ecosw)

        result0 = minimize(rez, fit_params, kws={'polyorder': 0}, iter_cb=MinimizeStopper(10), **fit_kws)
        report_fit(result0)
        fit_params = result0.params

    result0 = minimize(rez, fit_params, kws={'polyorder': 1}, iter_cb=MinimizeStopper(10), **fit_kws)
    report_fit(result0)


#    redchi2 = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
    #guess = get_lcvals(result0.params)
    fit_params = result0.params
    redchi2 = np.sum((rez(get_pars2vals(result0.params, partype='lc'), polyorder=1))**2) / np.sum(keblat.clip)

    fit_params['msum'].vary=vary_msum
    fit_params['tpe'].vary=True
    fit_params['period'].vary=True
    fit_params['b'].vary=True
    fit_params['frat'].vary=True
    fit_params['esinw'].vary=True
    fit_params['ecosw'].vary=True
    fit_params['rsum'].vary=True
    fit_params['rrat'].vary=True
    fit_params['q1'].vary=True
    fit_params['q2'].vary=True
    fit_params['q3'].vary=True
    fit_params['q4'].vary=True

    result0 = minimize(rez, fit_params, kws={'polyorder': 1}, iter_cb=MinimizeStopper(10), **fit_kws)
#    current_redchi = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
    current_redchi = np.sum((rez(get_pars2vals(result0.params, partype='lc'), polyorder=1))**2) / np.sum(keblat.clip)

    if current_redchi < redchi2:
        redchi2 = current_redchi * 1.
        #guess = get_lcvals(result0.params)
        fit_params = result0.params
        print "polyorder = 1: ", current_redchi, result0.redchi
        report_fit(result0)

    result0 = minimize(rez, fit_params, kws={'polyorder': 2}, iter_cb=MinimizeStopper(10), **fit_kws)
#    current_redchi = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
    current_redchi = np.sum((rez(get_pars2vals(result0.params, partype='lc'), polyorder=2))**2) / np.sum(keblat.clip)
    if current_redchi < redchi2:
        redchi2 = current_redchi * 1.
        #guess = get_lcvals(result0.params)
        fit_params = result0.params
        print "polyorder = 2: ", current_redchi, result0.redchi
        report_fit(result0)


#     fit_params['rsum'].vary=True
#     fit_params['rrat'].vary=True
#     niter=0
#
#     while (redchi2>1.) and (niter<5):
#         result0 = minimize(rez, fit_params, kws={'polyorder': 2}, iter_cb=MinimizeStopper(10), **fit_kws)
#         current_redchi = np.sum((rez(get_lcvals(result0.params), polyorder=2))**2) / np.sum(keblat.clip)
#         print "Iteration: ", niter, redchi2, current_redchi, result0.redchi, result0.nfev#, get_lcvals(result0.params)
# #        current_redchi = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
#         if current_redchi < redchi2:
#             print "Saving the following results:"
#             report_fit(result0)
#             redchi2 = current_redchi * 1.
#             #guess = get_lcvals(result0.params)
#             fit_params = result0.params
#         niter+=1
    guess = get_pars2vals(fit_params, partype='lc')
    return guess

def plot_fourth(keblat, lcpars, crow=None):
    pos1 = np.array([0, 4, 8, 12, 16])
    pos2 = np.array([1, 5, 9, 13, 17])
    pos3 = np.array([2, 6, 10, 14, 18])
    pos4 = np.array([3, 7, 11, 15, 19])
    if crow is None:
        crow=keblat.crowd
    fig=plt.figure()
    fig2=plt.figure()
    _ucrow = np.zeros(4)
    ii=0
    for pos in [pos1, pos2, pos3, pos4]:
        good = keblat.clip * np.array([(keblat.quarter == qrt) for qrt in pos]).sum(axis=0).astype(bool)
        if good.sum() > 0:
            lcmod, lcpol = keblat.lcfit(lcpars[:13], keblat.jd[good], keblat.quarter[good],
                                        keblat.flux[good], keblat.dflux[good], crow[good])
            ax = fig.add_subplot(2,2,ii+1)
            ax.plot(keblat.phase[good], keblat.flux[good]/lcpol, 'k.')
            ax.plot(keblat.phase[good], lcmod, 'r.')
            ax2 = fig2.add_subplot(2,2,ii+1)
            ax2.plot(keblat.quarter[good], keblat.crowd[good], '.')
            _ucrow[ii] = np.nanmedian(keblat.crowd[good])
        ii+=1
    return _ucrow

def crowd_fourth(keblat, crow):
    pos1 = np.array([0, 4, 8, 12, 16])
    pos2 = np.array([1, 5, 9, 13, 17])
    pos3 = np.array([2, 6, 10, 14, 18])
    pos4 = np.array([3, 7, 11, 15, 19])
    crowd_fits = keblat.crowd.copy()
    ii=0
    for pos in [pos1, pos2, pos3, pos4]:
        good = keblat.clip * np.array([(keblat.quarter == qrt) for qrt in pos]).sum(axis=0).astype(bool)
        if good.sum() > 0:
            crowd_fits[good] = crow[ii]
        ii+=1
    return crowd_fits

def opt_lc(keblat, **kwargs):
    #msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4 = lcpars0
    #set_upperb = kwargs.pop('set_upperb', 2.0)
    vary_msum = kwargs.pop('vary_msum', True)
    fit_crowd = kwargs.pop('fit_crowd', False)
    vary_timing = kwargs.pop('vary_timing', True)
    vary_frat = kwargs.pop('vary_frat', True)
    try:
        crowd_pars = kwargs.pop('crowd_pars')
    except:
        crowd_pars = keblat._crowdsap
    ooe = kwargs.pop('ooe', True)
    keblat.updatepars(**kwargs)

    fit_params = Parameters()
    # fit_params.add('esinw', value=esinw, min=-.999, max=0.999, vary=False)
    # fit_params.add('ecosw', value=ecosw, min=-.999, max=0.999, vary=False)#ecosw-0.05, max=ecosw+0.05, vary=False)
    # fit_params.add('rsum', value=rsum, min=0.1, max=10000., vary=False)
    # fit_params.add('rrat', value=rrat, min=1e-4, max=1e3, vary=False)
    # fit_params.add('b', value=b, min=0., max=set_upperb, vary=False)
    # fit_params.add('frat', value=frat, min=1e-6, max=1e2, vary=False)
    # fit_params.add('msum', value=msum, min=0.2, max=24., vary=False)
    #
    # fit_params.add('period', value=period, min=period-0.005, max=period+0.005, vary=False)
    # fit_params.add('tpe', value=tpe, min=tpe-10., max=tpe+10., vary=False)
    # fit_params.add('q1', value=q1, min=0., max=1., vary=False)
    # fit_params.add('q2', value=q2, min=0., max=1., vary=False)
    # fit_params.add('q3', value=q3, min=0., max=1., vary=False)
    # fit_params.add('q4', value=q4, min=0., max=1., vary=False)
    for name, val in kwargs.items():
        if name in keblat.parbounds.keys():
            fit_params.add(name, value=val, min=keblat.parbounds[name][0], max=keblat.parbounds[name][1], vary=False)
        else:
            fit_params.add(name, value=val, vary=False)

    fit_params['rrat'].vary=True
    fit_params['rsum'].vary=True
    fit_params['b'].vary=True
    fit_params['frat'].vary=vary_frat
    fit_params['esinw'].vary=True
    fit_params['ecosw'].vary=True
    fit_params['tpe'].vary=vary_timing

    fit_kws={'maxfev':2000*(len(fit_params)+1)}
    if fit_crowd:
        print "Fitting crowding parameters..."
#        for ii in fit_params.keys():
#            fit_params[ii].vary=True
        fit_params['ecosw'].vary=False
        fit_params['esinw'].vary=False
        fit_params['tpe'].vary=False
        for ii in range(len(np.unique(keblat.quarter))):
            fit_params.add('cr'+str(np.unique(keblat.quarter)[ii]), 
                           value=crowd_pars[ii], min=0.1, max=1.0)
        result0 = minimize(rez, fit_params, args=(keblat,), 
                           kws={'polyorder':2, 'qcrow':np.unique(keblat.quarter)}, 
                           iter_cb=MinimizeStopper(10), **fit_kws)
        report_fit(result0)

        for ii in result0.params.keys():
            result0.params[ii].vary=True
        niter=0
        redchi2 = np.sum((rez(get_pars2vals(result0.params, partype='lc', qcrow=np.unique(keblat.quarter)), 
                              keblat, qcrow=np.unique(keblat.quarter), polyorder=2))**2) / np.sum(keblat.clip)
        guess=get_pars2vals(result0.params, partype='lc', qcrow=np.unique(keblat.quarter))
        while (redchi2>1.) and (niter<5):
            result0 = minimize(rez, result0.params, args=(keblat, ), 
                               kws={'polyorder':2, 'qcrow':np.unique(keblat.quarter)}, 
                               iter_cb=MinimizeStopper(10), **fit_kws)
            current_chi = np.sum((rez(get_pars2vals(result0.params, partype='lc', 
                                                    qcrow=np.unique(keblat.quarter)), 
                                      keblat, polyorder=2))**2) / np.sum(keblat.clip)
            if current_chi < redchi2:
                redchi2=current_chi*1.0
                report_fit(result0)
                guess=get_pars2vals(result0.params, partype='lc', qcrow=np.unique(keblat.quarter))
            niter+=1

        return guess

    print "=========================================================================="
    print "==================== Starting LIGHTCURVE ONLY fit... ====================="
    print "=========================================================================="

    result0 = minimize(rez, fit_params, args=(keblat, ), 
                       kws={'polyorder': 1}, iter_cb=MinimizeStopper(10), **fit_kws)
    report_fit(result0)


#    redchi2 = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
    #guess = get_lcvals(result0.params)
    fit_params = result0.params
    redchi2 = np.sum((rez(get_pars2vals(result0.params, partype='lc'), 
                          keblat, polyorder=1))**2) / np.sum(keblat.clip)

    fit_params['msum'].vary=vary_msum
    fit_params['tpe'].vary=vary_timing
    fit_params['period'].vary=vary_timing
    fit_params['b'].vary=True
    fit_params['frat'].vary=vary_frat
    fit_params['esinw'].vary=True
    fit_params['ecosw'].vary=True
    fit_params['rsum'].vary=True
    fit_params['rrat'].vary=True
    fit_params['q1'].vary=True
    fit_params['q2'].vary=True
    fit_params['q3'].vary=True
    fit_params['q4'].vary=True

    result0 = minimize(rez, fit_params, args=(keblat, ),
                       kws={'polyorder': 1}, iter_cb=MinimizeStopper(10), **fit_kws)
#    current_redchi = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
    current_redchi = np.sum((rez(get_pars2vals(result0.params, partype='lc'), 
                                 keblat, polyorder=1))**2) / np.sum(keblat.clip)

    if current_redchi < redchi2:
        redchi2 = current_redchi * 1.
        #guess = get_lcvals(result0.params)
        fit_params = result0.params
        print "polyorder = 1: ", current_redchi, result0.redchi
        report_fit(result0)

    result0 = minimize(rez, fit_params, args=(keblat, ), 
                       kws={'polyorder': 2}, iter_cb=MinimizeStopper(10), **fit_kws)
#    current_redchi = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
    current_redchi = np.sum((rez(get_pars2vals(result0.params, partype='lc'), 
                                 keblat, polyorder=2))**2) / np.sum(keblat.clip)
    if current_redchi < redchi2:
        redchi2 = current_redchi * 1.
        #guess = get_lcvals(result0.params)
        fit_params = result0.params
        print "polyorder = 2: ", current_redchi, result0.redchi
        report_fit(result0)


#     fit_params['rsum'].vary=True
#     fit_params['rrat'].vary=True
#     niter=0
#
#     while (redchi2>1.) and (niter<5):
#         result0 = minimize(rez, fit_params, kws={'polyorder': 2}, iter_cb=MinimizeStopper(10), **fit_kws)
#         current_redchi = np.sum((rez(get_lcvals(result0.params), polyorder=2))**2) / np.sum(keblat.clip)
#         print "Iteration: ", niter, redchi2, current_redchi, result0.redchi, result0.nfev#, get_lcvals(result0.params)
# #        current_redchi = np.sum((result0.residual)**2) / (len(result0.residual)-result0.nfev)
#         if current_redchi < redchi2:
#             print "Saving the following results:"
#             report_fit(result0)
#             redchi2 = current_redchi * 1.
#             #guess = get_lcvals(result0.params)
#             fit_params = result0.params
#         niter+=1
    guess = get_pars2vals(fit_params, partype='lc')
    return guess

def opt_lc_crowd(keblat, **kwargs):
    #msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4 = lcpars0
    #set_upperb = kwargs.pop('set_upperb', 2.0)
    vary_msum = kwargs.pop('vary_msum', False)
    vary_ew = kwargs.pop('vary_ew', False)
    vary_frat = kwargs.pop('vary_frat', False)
    vary_b = kwargs.pop('vary_b', False)
    vary_rrat = kwargs.pop('vary_rrat', False)
    vary_rsum = kwargs.pop('vary_rsum', False)
    vary_tpe = kwargs.pop('vary_tpe', False)
    try:
        crowd_pars = kwargs.pop('crowd_pars')
    except:
        crowd_pars = keblat._crowdsap
    ooe = kwargs.pop('ooe', True)
    keblat.updatepars(**kwargs)

    fit_params = Parameters()
    # fit_params.add('esinw', value=esinw, min=-.999, max=0.999, vary=False)
    # fit_params.add('ecosw', value=ecosw, min=-.999, max=0.999, vary=False)#ecosw-0.05, max=ecosw+0.05, vary=False)
    # fit_params.add('rsum', value=rsum, min=0.1, max=10000., vary=False)
    # fit_params.add('rrat', value=rrat, min=1e-4, max=1e3, vary=False)
    # fit_params.add('b', value=b, min=0., max=set_upperb, vary=False)
    # fit_params.add('frat', value=frat, min=1e-6, max=1e2, vary=False)
    # fit_params.add('msum', value=msum, min=0.2, max=24., vary=False)
    #
    # fit_params.add('period', value=period, min=period-0.005, max=period+0.005, vary=False)
    # fit_params.add('tpe', value=tpe, min=tpe-10., max=tpe+10., vary=False)
    # fit_params.add('q1', value=q1, min=0., max=1., vary=False)
    # fit_params.add('q2', value=q2, min=0., max=1., vary=False)
    # fit_params.add('q3', value=q3, min=0., max=1., vary=False)
    # fit_params.add('q4', value=q4, min=0., max=1., vary=False)
    for name, val in kwargs.items():
        if name in keblat.parbounds.keys():
            fit_params.add(name, value=val, min=keblat.parbounds[name][0], max=keblat.parbounds[name][1], vary=False)
        else:
            fit_params.add(name, value=val, vary=False)

    fit_params['rrat'].vary=vary_rrat
    fit_params['rsum'].vary=vary_rsum
    fit_params['b'].vary=vary_b
    fit_params['frat'].vary=vary_frat
    fit_params['tpe'].vary=vary_tpe
    fit_params['msum'].vary=vary_msum

    fit_params['q1'].vary=True
    fit_params['q2'].vary=True
    fit_params['q3'].vary=True
    fit_params['q4'].vary=True
    
    fit_kws={'maxfev':100*(len(fit_params)+1)}
    print "Fitting crowding parameters..."
#        for ii in fit_params.keys():
#            fit_params[ii].vary=True
    fit_params['ecosw'].vary=vary_ew
    fit_params['esinw'].vary=vary_ew
    for ii in range(len(np.unique(keblat.quarter))):
        fit_params.add('cr'+str(np.unique(keblat.quarter)[ii]), 
                       value=crowd_pars[ii], min=0.1, max=1.0)
    result0 = minimize(rez, fit_params, args=(keblat,), 
                       kws={'polyorder':2, 'qcrow':np.unique(keblat.quarter)}, 
                       iter_cb=MinimizeStopper(10), **fit_kws)
    report_fit(result0)

    for ii in result0.params.keys():
        result0.params[ii].vary=True
    niter=0
    redchi2 = np.sum((rez(get_pars2vals(result0.params, partype='lc', qcrow=np.unique(keblat.quarter)), 
                          keblat, qcrow=np.unique(keblat.quarter), polyorder=2))**2) / np.sum(keblat.clip)
    guess=get_pars2vals(result0.params, partype='lc', qcrow=np.unique(keblat.quarter))
    while (redchi2>1.) and (niter<5):
        result0 = minimize(rez, result0.params, args=(keblat, ), 
                           kws={'polyorder':2, 'qcrow':np.unique(keblat.quarter)}, 
                           iter_cb=MinimizeStopper(10), **fit_kws)
        current_chi = np.sum((rez(get_pars2vals(result0.params, partype='lc', 
                                                qcrow=np.unique(keblat.quarter)), 
                                  keblat, polyorder=2))**2) / np.sum(keblat.clip)
        if current_chi < redchi2:
            redchi2=current_chi*1.0
            report_fit(result0)
            guess=get_pars2vals(result0.params, partype='lc', qcrow=np.unique(keblat.quarter))
        niter+=1

    return guess

def opt_sedlc(keblat, **kwargs):
    #mciso = kwargs.pop('mciso', None)
    # fit_ebv = kwargs.pop('fit_ebv', True)
    # set_upperb = kwargs.pop('set_upperb', None)
    init_porder = kwargs.pop('init_porder', 1)
    init_varyb = kwargs.pop('init_varyb', False)
    init_varyew = kwargs.pop('init_varyew', False)
    init_varyza = kwargs.pop('init_varyza', False)
    lc_constraints = kwargs.pop('lc_constraints', False)
    freeze_iso = kwargs.pop('freeze_iso', False)
    vary_ephem = kwargs.pop('vary_ephem', True)
    vary_dist = kwargs.pop('vary_dist', False)
    #isonames = ['m1', 'm2', 'z0', 'age', 'dist', 'ebv', 'h0', 'isoerr']

    fit_params = Parameters()

    for name, val in kwargs.items():
        if name in keblat.parbounds.keys():
            if (name == 'isoerr') or (name == 'lcerr'):
                fit_params.add(name, value=val, min=np.exp(keblat.parbounds[name][0]), 
                           max=np.exp(keblat.parbounds[name][1]), vary=False)
                print fit_params[name]
            else:
                fit_params.add(name, value=val, min=keblat.parbounds[name][0], 
                           max=keblat.parbounds[name][1], vary=False)
        else:
            fit_params.add(name, value=val, vary=False)

    kws = {'lc_constraints': lc_constraints,
           'qua': np.unique(keblat.quarter), 'polyorder': init_porder, 'residual': True, 'ebv_arr': None}

    fit_params['age'].vary=True
    fit_params['msum'].vary=True
    fit_params['mrat'].vary=True
    fit_params['z0'].vary=True
    fit_params['ebv'].vary=True
    fit_params['dist'].vary=True
    fit_params['isoerr'].vary=False

    # fit_params.add('esinw', value=guess[5], min=-.999, max=0.999, vary=False)
    # fit_params.add('ecosw', value=guess[6], min=guess[6]-0.05, max=guess[6]+0.05, vary=False)
    # fit_params.add('tpe', value=guess[4], min=tpe-10., max=tpe+10., vary=False)
    # fit_params.add('period', value=guess[3], min=period-0.005, max=period+0.005, vary=False)
    # fit_params.add('q1', value=guess[-4], min=0., max=1., vary=False)
    # fit_params.add('q2', value=guess[-3], min=0., max=1., vary=False)
    # fit_params.add('q3', value=guess[-2], min=0., max=1., vary=False)
    # fit_params.add('q4', value=guess[-1], min=0., max=1., vary=False)
    #fit_params.add('lcerr', value=1e-6, min=0., max=1e-3, vary=False)
    # if set_upperb is None:
    #     fit_params.add('b', value=guess[7], min=0., vary=False)
    # else:
    #     fit_params.add('b', value=guess[7], min=0., max=set_upperb, vary=False)

    # fit_params['esinw'].vary=False
    # fit_params['ecosw'].vary=False
    if init_varyew:
        fit_params['esinw'].vary=True
        fit_params['ecosw'].vary=True
    if init_varyb:
        fit_params['b'].vary=True
    if init_varyza:
        fit_params['z0'].vary=False
        fit_params['age'].vary=False
    fit_params['dist'].vary=vary_dist
    if freeze_iso:
        fit_params['msum'].vary=False
        fit_params['mrat'].vary=False
        fit_params['age'].vary=False
        fit_params['z0'].vary=False
    fit_kws={'maxfev':2000*(len(fit_params)+1)}

    print "=========================================================================="
    print "================= Starting SED + LC simultaneous fit... =================="
    print "=========================================================================="

    # print fit_params
    result3 = minimize(lnlike_lmfit, fit_params, args=(keblat, ), 
                       kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)


    fit_params = result3.params.copy()
    _allres = lnlike_lmfit(result3.params, keblat, lc_constraints=lc_constraints, 
                           qua=np.unique(keblat.quarter), polyorder=2, residual=True)
    redchi2 = np.sum(_allres**2) / len(_allres)
    print redchi2, result3.redchi
    report_fit(result3)
    print result3.message

    fit_params['age'].vary=True
    fit_params['msum'].vary=True
    fit_params['mrat'].vary=True
    fit_params['z0'].vary=True
    fit_params['dist'].vary=True
    fit_params['esinw'].vary=True
    fit_params['ecosw'].vary=True
    fit_params['b'].vary=True
    fit_params['q1'].vary=True
    fit_params['q2'].vary=True
    fit_params['q3'].vary=True
    fit_params['q4'].vary=True

    result3 = minimize(lnlike_lmfit, fit_params, args=(keblat, ), 
                       kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
    _allres = lnlike_lmfit(result3.params, keblat, lc_constraints=lc_constraints, 
                           qua=np.unique(keblat.quarter), polyorder=2, residual=True)
    current_redchi2 = np.sum(_allres**2) / len(_allres)
    if current_redchi2 < redchi2:
        print "The following results are saved:", current_redchi2, result3.redchi
        report_fit(result3)
        print result3.message
        fit_params = result3.params.copy()
        redchi2 = current_redchi2

    kws['polyorder'] = 2

    fit_params['tpe'].vary=vary_ephem
    fit_params['period'].vary=vary_ephem

    niter=0
    while (niter<3): #(redchi2>1.) and (niter<10):
        result3 = minimize(lnlike_lmfit, fit_params, args=(keblat, ), 
                           kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
        _allres = lnlike_lmfit(result3.params, keblat, lc_constraints=lc_constraints, 
                               qua=np.unique(keblat.quarter), polyorder=2, residual=True)
        current_redchi2 = np.sum(_allres**2) / len(_allres)
        print "Iteration: ", niter, current_redchi2, result3.redchi

        if current_redchi2 < redchi2:
            print "The following results are saved:"
            report_fit(result3)
            fit_params = result3.params.copy()
            redchi2 = current_redchi2
        niter+=1
    #print "logL of best allpars = ", keblat.lnlike(allpars, lc_constraints=None, qua=np.unique(keblat.quarter), polyorder=2)
    allpars = get_pars2vals(fit_params, partype='lcsed')
    #print fit_params
    return allpars


def opt_sedlc_fix1par(keblat, **kwargs):
    init_porder = kwargs.pop('init_porder', 2)
    lc_constraints = kwargs.pop('lc_constraints', None)
    vary_timing = kwargs.pop('vary_timing',False)
    par2fix = kwargs.pop('par2fix', None)
    niter_max = kwargs.pop('niter_max', 1)
    xtol = kwargs.pop('xtol', 1e-8)
    ftol = kwargs.pop('ftol', 1e-8)


    fit_params = Parameters()

    for name, val in kwargs.items():
        if name in keblat.parbounds.keys():
            if (name == 'isoerr') or (name == 'lcerr'):
                fit_params.add(name, value=val, min=np.exp(keblat.parbounds[name][0]), 
                           max=np.exp(keblat.parbounds[name][1]), vary=False)
                print fit_params[name]
            else:
                fit_params.add(name, value=val, min=keblat.parbounds[name][0], 
                           max=keblat.parbounds[name][1], vary=True)
        else:
            fit_params.add(name, value=val, vary=True)
    fit_params['h0'].vary=False
    fit_params['period'].vary=vary_timing
    fit_params['tpe'].vary=vary_timing
    kws = {'lc_constraints': lc_constraints,
           'qua': np.unique(keblat.quarter), 'polyorder': init_porder, 
           'residual': True, 'ebv_arr': None}
    if par2fix is not None:
        fit_params[par2fix].vary=False
    fit_kws={'maxfev':2000*(len(fit_params)+1), 'xtol':xtol, 'ftol':ftol}

    print "=========================================================================="
    print "================= Starting SED + LC simultaneous fit... =================="
    try:
        print "=============== Fixing {}={} while all other pars fixed ==================".format(par2fix, fit_params[par2fix].value)
    except:
        print "================~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~===================" 
    print "=========================================================================="

    redchi2 = 1e25
    niter=0
    while (niter<niter_max): #(redchi2>1.) and (niter<10):
        result3 = minimize(lnlike_lmfit, fit_params, args=(keblat, ), 
                           kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
#        print(result3.params)
        _allres = lnlike_lmfit(result3.params, keblat, lc_constraints=lc_constraints, 
                               qua=np.unique(keblat.quarter), polyorder=2, residual=True)
        current_redchi2 = np.sum(_allres**2) / len(_allres)
        print "Iteration: ", niter, current_redchi2, result3.redchi

        if current_redchi2 < redchi2:
            print "The following results are saved:"
            report_fit(result3)
            fit_params = result3.params.copy()
#            print(result3.params, fit_params)
            redchi2 = current_redchi2*1.0
        niter+=1
    #print "logL of best allpars = ", keblat.lnlike(allpars, lc_constraints=None, qua=np.unique(keblat.quarter), polyorder=2)
    allpars = get_pars2vals(fit_params, partype='lcsed')
    #print fit_params
    return allpars


def opt_sedlc_old(fit_params2, guess, ebv_dist, ebv_arr, jd, phase, flux, dflux, crowd, clip, mciso=None, fit_ebv=True,
              set_upperb = None, init_porder=1, init_varyb=False, init_varyew=False, init_varyza=False, lc_constraints=False):
    isonames = ['m1', 'm2', 'z0', 'age', 'dist', 'ebv', 'h0', 'isoerr']

    fit_params = Parameters()
    """
    for ii in range(len(keblat.pars)):
        fit_params.add(keblat.pars.keys()[ii], value=initvals[ii], min=keblat.parbounds.values()[ii][0],
                        max=keblat.parbounds.values()[ii][1])
    fit_params.pop('lcerr')
    fit_params['isoerr'].vary=False
    fit_params['h0'].vary=False
    kws = {'qua': np.unique(keblat.quarter), 'polyorder': 2, 'residual': True, 'ebv_arr': None}

    if not fit_ebv:
        fit_params.pop['ebv']
        kws = {'qua': np.unique(keblat.quarter), 'polyorder': 2, 'residual': True, 'ebv_arr': ebv_arr}
    """
    fit_params = fit_params2
    if mciso is not None:
        for ii in range(len(isonames)):
            fit_params[isonames[ii]].value = mciso[ii]
    fit_params['ebv'].max = 1.0
    fit_params['age'].vary=True
    fit_params['m1'].vary=True
    fit_params['m2'].vary=True
    fit_params['z0'].vary=True
    fit_params['ebv'].vary=True
    fit_params['dist'].vary=True
    fit_params['isoerr'].vary=False

    kws = {'lc_constraints': lc_constraints,
           'qua': np.unique(keblat.quarter), 'polyorder': init_porder, 'residual': True, 'ebv_arr': None}

    fit_params.add('esinw', value=guess[5], min=-.999, max=0.999, vary=False)
    fit_params.add('ecosw', value=guess[6], min=guess[6]-0.05, max=guess[6]+0.05, vary=False)
    fit_params.add('tpe', value=guess[4], min=tpe-10., max=tpe+10., vary=False)
    fit_params.add('period', value=guess[3], min=period-0.005, max=period+0.005, vary=False)
    fit_params.add('q1', value=guess[-4], min=0., max=1., vary=False)
    fit_params.add('q2', value=guess[-3], min=0., max=1., vary=False)
    fit_params.add('q3', value=guess[-2], min=0., max=1., vary=False)
    fit_params.add('q4', value=guess[-1], min=0., max=1., vary=False)
    fit_params.add('lcerr', value=1e-6, min=0., max=1e-3, vary=False)
    if set_upperb is None:
        fit_params.add('b', value=guess[7], min=0., vary=False)
    else:
        fit_params.add('b', value=guess[7], min=0., max=set_upperb, vary=False)

    fit_params['esinw'].vary=False
    fit_params['ecosw'].vary=False
    if init_varyew:
        fit_params['esinw'].vary=True
    if init_varyb:
        fit_params['b'].vary=True
    if init_varyza:
        fit_params['z0'].vary=False
        fit_params['age'].vary=False

    fit_kws={'maxfev':2000*(len(fit_params)+1)}

    print "=========================================================================="
    print "================= Starting SED + LC simultaneous fit... =================="
    print "=========================================================================="

    print fit_params
    result3 = minimize(lnlike_lmfit, fit_params, kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)


    fit_params = result3.params.copy()
    _allres = lnlike_lmfit(result3.params, lc_constraints=lc_constraints, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
    redchi2 = np.sum(_allres**2) / len(_allres)
    print redchi2, result3.redchi
    report_fit(result3)
    print result3.message

    fit_params['age'].vary=True
    fit_params['m1'].vary=True
    fit_params['m2'].vary=True
    fit_params['z0'].vary=True
    fit_params['dist'].vary=True
    fit_params['esinw'].vary=True
    fit_params['ecosw'].vary=True
    fit_params['b'].vary=True
    fit_params['q1'].vary=True
    fit_params['q2'].vary=True
    fit_params['q3'].vary=True
    fit_params['q4'].vary=True

    result3 = minimize(lnlike_lmfit, fit_params, kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
    _allres = lnlike_lmfit(result3.params, lc_constraints=lc_constraints, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
    current_redchi2 = np.sum(_allres**2) / len(_allres)
    if current_redchi2 < redchi2:
        print "The following results are saved:", current_redchi2, result3.redchi
        report_fit(result3)
        print result3.message
        fit_params = result3.params.copy()
        redchi2 = current_redchi2

    kws['polyorder'] = 2

    fit_params['tpe'].vary=True
    fit_params['period'].vary=True

    niter=0
    while (niter<10): #(redchi2>1.) and (niter<10):
        result3 = minimize(lnlike_lmfit, fit_params, kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
        _allres = lnlike_lmfit(result3.params, lc_constraints=lc_constraints, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
        current_redchi2 = np.sum(_allres**2) / len(_allres)
        print "Iteration: ", niter, current_redchi2, result3.redchi

        if current_redchi2 < redchi2:
            print "The following results are saved:"
            report_fit(result3)
            fit_params = result3.params.copy()
            redchi2 = current_redchi2
        niter+=1
    #print "logL of best allpars = ", keblat.lnlike(allpars, lc_constraints=None, qua=np.unique(keblat.quarter), polyorder=2)
    allpars = get_pars2vals(fit_params, partype='lcsed')
    #print fit_params
    return allpars

def opt_rv(keblat, **kwargs):
    vary_ecosw=kwargs.pop('vary_ecosw', False)
    vary_esinw=kwargs.pop('vary_esinw', False)
    vary_inc=kwargs.pop('vary_inc', False)
    vary_msum=kwargs.pop('vary_msum', False)
    vary_mrat=kwargs.pop('vary_mrat', False)

    fit_pars = Parameters()
    for name, val in kwargs.items():
        print name, val
        fit_pars.add(name, value=val, min=keblat.parbounds[name][0], max=keblat.parbounds[name][1])
    fit_pars['rverr'].vary=False
    fit_pars['period'].vary=False
    fit_pars['tpe'].vary=False
    fit_pars['esinw'].vary=vary_esinw
    fit_pars['ecosw'].vary=vary_ecosw
    fit_pars['inc'].vary=vary_inc
    fit_pars['msum'].vary=vary_msum
    fit_pars['mrat'].vary=vary_mrat

    # fit_pars['m1'].min=fit_pars['m1'].value*0.5
    # fit_pars['m2'].min=fit_pars['m2'].value*0.5
    # fit_pars['m1'].max=fit_pars['m1'].value*1.5
    # fit_pars['m2'].max=fit_pars['m2'].value*1.5
    # fit_pars['k0'].min=fit_pars['k0'].value - abs(fit_pars['k0'].value)
    # fit_pars['k0'].min=fit_pars['k0'].value + abs(fit_pars['k0'].value)

    fit_kws = {'maxfev': 2000 * (len(fit_pars) + 1)}

    print "=========================================================================="
    print "========================= Starting RV ONLY fit... ========================"
    print "=========================================================================="
    print fit_pars
    result3 = minimize(lnlike_rv, fit_pars, args=(keblat, ), 
                       iter_cb=MinimizeStopper(60), **fit_kws)
    fit_params = result3.params.copy()
    _allres = lnlike_rv(result3.params, keblat)
    redchi2 = np.sum(_allres ** 2) / len(_allres)
    print redchi2, result3.redchi
    report_fit(result3)
    print result3.message
    niter = 0
    while (niter < 10):  # (redchi2>1.) and (niter<10):
        result3 = minimize(lnlike_rv, fit_params, args=(keblat, ),
                           iter_cb=MinimizeStopper(60), **fit_kws)
        _allres = lnlike_rv(result3.params, keblat)
        current_redchi2 = np.sum(_allres ** 2) / len(_allres)
        print "Iteration: ", niter, current_redchi2, result3.redchi
        if current_redchi2 < redchi2:
            print "The following results are saved:"
            report_fit(result3)
            fit_params = result3.params.copy()
            redchi2 = current_redchi2
        niter += 1
    # print "logL of best allpars = ", keblat.lnlike(allpars, lc_constraints=None, qua=np.unique(keblat.quarter), polyorder=2)
    rvpars = get_pars2vals(fit_params, partype='rv')
    # print fit_params
    return rvpars

def opt_lcrv(keblat, **kwargs):
    vary_tpe = kwargs.pop('vary_tpe', False)
    vary_period=kwargs.pop('vary_period', False)
    vary_err = kwargs.pop('vary_err', False)
    retro = kwargs.pop('retro', False)
    fit_pars = Parameters()
    for name, val in kwargs.items():
        if name in keblat.parbounds.keys():
            fit_pars.add(name, value=val, min=keblat.parbounds[name][0], max=keblat.parbounds[name][1])
        else:
            fit_pars.add(name, value=val)
    kws = {'qua': np.unique(keblat.quarter), 'polyorder': 2, 'residual': True, 'retro': retro}
    fit_kws = {'maxfev': 2000 * (len(fit_pars) + 1)}
    fit_pars['lcerr'].vary=vary_err
    fit_pars['rverr'].vary=vary_err
    fit_pars['tpe'].vary=vary_tpe
    fit_pars['period'].vary=vary_period

    print "=========================================================================="
    print "================= Starting LC + RV simultaneous fit... ==================="
    print "=========================================================================="
    result3 = minimize(lnlike_lcrv, fit_pars, args=(keblat, ),
                       kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
    fit_params = result3.params.copy()
    _allres = lnlike_lcrv(result3.params, keblat, qua=np.unique(keblat.quarter), 
                          polyorder=2, residual=True, retro=retro)
    redchi2 = np.sum(_allres ** 2) / len(_allres)
    print redchi2, result3.redchi
    report_fit(result3)
    print result3.message
    niter = 0
    while (niter < 5):  # (redchi2>1.) and (niter<10):
        result3 = minimize(lnlike_lcrv, fit_params, args=(keblat, ), 
                           kws=kws, iter_cb=MinimizeStopper(60), **fit_kws)
        _allres = lnlike_lcrv(result3.params, keblat, qua=np.unique(keblat.quarter), 
                              polyorder=2, residual=True, retro=retro)
        current_redchi2 = np.sum(_allres ** 2) / len(_allres)
        print "Iteration: ", niter, current_redchi2, result3.redchi
        if current_redchi2 < redchi2:
            print "The following results are saved:"
            report_fit(result3)
            fit_params = result3.params.copy()
            redchi2 = current_redchi2
        niter += 1
    # print "logL of best allpars = ", keblat.lnlike(allpars, lc_constraints=None, qua=np.unique(keblat.quarter), polyorder=2)
    lcrvpars = get_pars2vals(fit_params, partype='lcrv')
    # print fit_params
    return lcrvpars

def estimate_rsum(rsum, period, eclipse_widths, msum=1.0):
    if msum is None:
        msum=rsum
    res = abs(rsum/compute_a(period, msum, unit_au=False) - eclipse_widths)
    return res

def compute_tse(e, w, period, tpe_obs):
    t0 = tpe_obs - sudarsky(np.pi/2. - w, e, period)
    tse = t0 + sudarsky(-np.pi/2. - w, e, period)
    return tse

def estimate_ew(ew, *args):
    period, tpe_obs, tse_obs = args[0], args[1], args[2]
    e, w = ew
    if (e<0) or (e>1):
        return 1e8
    tse = compute_tse(e, w, period, tpe_obs)
    return ((tse % period - tse_obs % period)/0.001)**2

def tse_residuals(ew, *args):
    esinw, ecosw = ew
    e = np.sqrt(esinw**2 + ecosw**2)
    w = np.arctan2(esinw, ecosw)
    if e>1:
        return 1e3
    period, tpe, tse0 = args[0], args[1], args[2]
    tse = tpe - sudarsky(np.pi/2.-w, e, period) + sudarsky(-np.pi/2.-w, e, period)
    return (tse % period - tse0 % period) / 0.01#**2

def flatbottom(x, y, sep, swidth):
    check = (x<sep+swidth/3.) * (x>sep-swidth/3.)
    grady = np.gradient(y)
    grady_m = np.polyfit(x[check], grady[check], 1)[0]
    if abs(grady_m)<0.1:
        return 0.01
    elif abs(grady_m)>10.:
        return 0.4
    else:
        return 0.1

def guess_rrat(sdep, pdep):
    if (pdep>0.2):
        val = sdep/pdep*1.4
        if val>1.:
            return 0.95
        elif val<0.5:
            return 0.7
        return val
    else:
        return np.clip(sdep/pdep, 0.1, 0.95)

def check_lcresiduals(x, y, ey):
    degrees = [2, 5, 9, 13]
    bic = np.zeros(len(degrees))
    for i in range(len(degrees)):
        z = np.poly1d(np.polyfit(x, y, degrees[i]))
        bic[i] = np.sum(((z(x) - y)/ey)**2) + degrees[i]*np.log(len(y))
    bic_slope = np.median(np.diff(bic))/bic[0]
    if bic_slope < -0.1:
        return bic, bic_slope, True
    return bic, bic_slope, False

def ilnprior(isopars, keblat):

    bounds = np.array([keblat.parbounds['msum'], keblat.parbounds['mrat'], 
                       keblat.parbounds['z0'], keblat.parbounds['age'],
                       keblat.parbounds['dist'], 
                       keblat.parbounds['ebv'], keblat.parbounds['h0'], 
                       keblat.parbounds['isoerr']])
    pcheck = np.all((np.array(isopars) >= bounds[:, 0]) & \
                    (np.array(isopars) <= bounds[:, 1]))
    if pcheck:
        return 0.0 + isopars[3]*np.log(10.) + np.log(np.log(10.))
    else:
        return -np.inf

def ilnprob(isopars, keblat, lc_constraints=None):
    lp = ilnprior(isopars, keblat)
    if np.isinf(lp):
        return -np.inf, str((0, 0, 0))
    ll, blobs = keblat.ilnlike(isopars, lc_constraints=lc_constraints)
    if (np.isnan(ll) or np.isinf(ll)):
        return -np.inf, str((0, 0, 0))
    return lp + ll, blobs

def lnprior(allpars, keblat, crowd_fit=False):       
    m1, m2, z0, age, dist, ebv, h0, period, tpe, esinw, ecosw, \
        b, q1, q2, q3, q4, lcerr, isoerr = allpars[:18]

    lcerr = np.log(lcerr)
    isoerr = np.log(isoerr)
    e = np.sqrt(esinw**2 + ecosw**2)
    pars2check = np.array([m1, m2, z0, age, dist, ebv, h0, \
        period, tpe, e, b, q1, q2, q3, q4, lcerr, isoerr])
    bounds = np.array([keblat.parbounds['m1'], keblat.parbounds['m2'], 
                       keblat.parbounds['z0'], keblat.parbounds['age'],
                       keblat.parbounds['dist'], keblat.parbounds['ebv'], 
                       keblat.parbounds['h0'], keblat.parbounds['period'], 
                       keblat.parbounds['tpe'], (0., 0.99), keblat.parbounds['b'], 
                       keblat.parbounds['q1'], keblat.parbounds['q2'],
                       keblat.parbounds['q3'], keblat.parbounds['q4'], 
                       (-16, -4), (-16, -1)])
    if crowd_fit:
        bounds = np.vstack((bounds, [[0, 1]]*len(allpars[18:])))
        pars2check = np.append(pars2check, allpars[18:])
    pcheck = np.all((pars2check >= bounds[:,0]) & \
                    (pars2check <= bounds[:,1]))

    if pcheck:
        return 0.0 + age*np.log(10.) + np.log(np.log(10.))
    else:
        return -np.inf

def lnlike(allpars, keblat, polyorder=2, crowd_fit=None):
    m1, m2, z0, age, dist, ebv, h0, period, tpe, esinw, \
        ecosw, b, q1, q2, q3, q4, lcerr, isoerr = allpars[:18]
    ldcoeffs = np.array([q1, q2, q3, q4])
    keblat.updatepars(m1=m1, m2=m2, z0=z0, age=age, dist=dist, ebv=ebv,
                            h0=h0, period=period, tpe=tpe, esinw=esinw, 
                            ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4, 
                            lcerr=lcerr, isoerr=isoerr, msum=m1+m2, mrat=m2/m1)
    isopars = [m1, m2, z0, age, dist, ebv, h0, isoerr]
    magsmod = keblat.isofit(isopars)
    if np.any(np.isinf(magsmod)):
        return -np.inf
    lc_inputs = np.array([(keblat.r1+keblat.r2)/(m1+m2)**(1./3.), keblat.r2/keblat.r1, keblat.frat])

    if np.any(np.isinf(lc_inputs)):
        return -np.inf
    isores = (magsmod - keblat.magsobs) / np.sqrt(keblat.emagsobs**2 + isoerr**2) #/ np.sqrt(self.emagsobs**2 + isoerr**2)
#    for ii, dii, jj in zip([keblat.armstrongT1, keblat.armstrongT2],
#                           [keblat.armstrongdT1, keblat.armstrongdT2],
#                           [10**keblat.temp1, 10**keblat.temp2]):
#        if ii is not None:
#            if dii is None:
#                dii=0.05*ii
#            isores = np.append(isores, (ii-jj)/dii)
    sed_like = np.sum(isores**2) + np.sum(np.log((keblat.emagsobs**2 + isoerr**2)))
    sed_like += ((isopars[5] - keblat.ebv)/(keblat.debv))**2
    sed_like += ((isopars[2] - keblat.z0)/(0.2 * np.log(10) * keblat.z0))**2
    sed_like *= -0.5

    #now the light curve fitting part
    lcpars = np.concatenate((np.array([m1+m2, keblat.r1+keblat.r2, 
                                       keblat.r2/keblat.r1, period, 
                                       tpe, esinw, ecosw, b, keblat.frat]),
                                       ldcoeffs))
    clip = keblat.clip

    lcmod, lcpol = keblat.lcfit(lcpars, keblat.jd[clip],
                              keblat.quarter[clip], keblat.flux[clip],
                            keblat.fluxerr[clip], np.ones(clip.sum()),
                            polyorder=polyorder)

    if np.any(np.isinf(lcmod)):
        return -np.inf
    a = keblat.flux[clip] - 1.
    b = lcmod*lcpol - 1.
    bad = (b == 0)
    c2 = 2.*(keblat.fluxerr[clip]**2 + lcerr**2)
    A = np.sum((b**2/c2)[~bad])
    lc_like = -0.5*np.sum(np.log(c2[~bad])) - 0.5 * np.log(A) + \
                np.sum((a*b/c2)[~bad])**2 / A - np.sum((a**2/c2)[~bad])
    erf_sum = scipy.special.erf(np.sum(((b**2 - a*b)/c2)[~bad]) / np.sqrt(A)) + \
                       scipy.special.erf(np.sum((a*b/c2)[~bad]) / (np.sqrt(A)))
    if erf_sum == 0: erf_sum = 1e-16
    lc_like += np.log(erf_sum)
#    if np.isinf(lc_like):
#        print A, -0.5*np.sum(np.log(c2[~bad])), np.sum((a*b/c2)[~bad])**2 / A, \
#            np.sum((a**2/c2)[~bad]), scipy.special.erf(np.sum(((b**2 - a*b)/c2)[~bad]) / np.sqrt(A)), \
#            scipy.special.erf(np.sum((a*b/c2)[~bad]) / (np.sqrt(A)))
#    print -0.5*np.sum(np.log(c2)), -0.5 * np.log(A)
#    print np.sum(a*b/c2)**2 / A, -np.sum(a**2/c2)
#    print scipy.special.erf(np.sum((b**2 - a*b)/c2) / np.sqrt(A)), scipy.special.erf(np.sum(a*b/c2) / (np.sqrt(A)))
#    lc_like = -0.5 * np.sum(np.log(keblat.fluxerr[clip]**2 + lcerr**2)) + \
#                      0.5 * np.sum(mmod**2 / (keblat.fluxerr[clip]**2 + lcerr**2)) + \
#                      np.log(scipy.special.erf(np.sum((lcmod*lcpol - keblat.flux[clip])/mmod)) + \
#                              scipy.special.erf(np.sum((keblat.flux[clip]-1)/mmod)))

#    if np.isinf(lc_like):
#        print np.log(scipy.special.erf((lcmod*lcpol - keblat.flux[clip])/mmod) +
#                              scipy.special.erf((keblat.flux[clip]-1)/mmod)), np.sum(np.log(scipy.special.erf((lcmod*lcpol - keblat.flux[clip])/mmod) +
#                              scipy.special.erf((keblat.flux[clip]-1)/mmod)))
    #chisq += ((isopars[6] - 119.)/(15.))**2        

    lili = lc_like + sed_like
    crowd_star = ((keblat.flux[clip]-1)/b)[~bad]
    chunk = np.unique(np.append([0, len(b[~bad])], np.where(np.diff(keblat.quarter[clip][~bad])>0)[0]+1))
#    bad = np.isinf(crowd_star)
#    if bad.sum() > 0:
#        return lcmod, lcpol
#    crowd_star[bad] = 1.0
#    if bad.sum()>0:
#        print keblat.jd[clip][bad], lcmod[bad], lcpol[bad], keblat.flux[clip][bad], mmod[bad], crowd_star[bad]
    crowding = [np.sum(crowd_star[chunk[ii]:chunk[ii+1]])/(chunk[ii+1]-chunk[ii]) for ii in range(len(chunk)-1)]
#    if np.isinf(lili):
#        print lc_like, sed_like, bad.sum(), A, -0.5*np.sum(np.log(c2[~bad])), np.sum((a*b/c2)[~bad])**2 / A, \
#            np.sum((a**2/c2)[~bad]), scipy.special.erf(np.sum(((b**2 - a*b)/c2)[~bad]) / np.sqrt(A)), \
#            scipy.special.erf(np.sum((a*b/c2)[~bad]) / (np.sqrt(A)))
    return lili, crowding
    

def lnprob(allpars, keblat, lc_constraints=None, crowd_fit=False):
    lp = lnprior(allpars, keblat, crowd_fit=crowd_fit)
    if np.isinf(lp):
        return -np.inf, str([-np.inf]*(3+len(np.unique(keblat.quarter[keblat.clip]))))
#    allpars[-2:] = np.exp(allpars[-2:])
#    ll = keblat.lnlike(allpars[:18], lc_constraints=lc_constraints, 
#                       qua=np.unique(keblat.quarter),
#                       crowd_fit=allpars[18:] if crowd_fit else None)
    ll, crowd = lnlike(allpars[:18], keblat)
    if (np.isnan(ll) or np.isinf(ll)):
        return -np.inf, str([-np.inf]*(3+len(crowd)))
    return lp + ll, str([keblat.r1, keblat.r2, keblat.frat]+crowd)

def mix_lnprior(allpars, keblat):
    m1, m2, z0, age, dist, ebv, h0, period, tpe, esinw, ecosw, \
        b, q1, q2, q3, q4, lnlcerr = allpars[:17]
    Pb, Yb, lnisoerr = allpars[17:]
    e = np.sqrt(esinw**2 + ecosw**2)
    pars2check = np.array([m1, m2, z0, age, dist, ebv, h0, \
        period, tpe, e, b, q1, q2, q3, q4, lnlcerr, Pb, Yb, lnisoerr])
    bounds = np.array([(.1, 12.), (.1, 12.), (0.001, 0.06), (6., 10.1),
                       (10., 15000.), (0.0, 1.0), (119-20., 119+20.), #(10., 15000.), (0.0, 1.0), (119-20., 119+20.),
                        (5., 3000.), (0., 1e8), (0., 0.99), (0., 10.), (0.,1.),
                        (0.,1.), (0.,1.), (0.,1.), (-14, -4.5),
                       (0., 1.), (np.min(keblat.magsobs-2.), np.max(keblat.magsobs)+2.), (-8, 0.)])

    pcheck = np.all((pars2check >= bounds[:,0]) & \
                    (pars2check <= bounds[:,1]))

    if pcheck:
        return 0.0 + age*np.log(10.) + np.log(np.log(10.))
    else:
        return -np.inf

def mix_lnlike2(allpars, keblat, polyorder=2, ooe=True):
    m1, m2, z0, age, dist, ebv, h0, period, tpe, esinw, \
        ecosw, b, q1, q2, q3, q4, lcerr = allpars[:17]
    #dist = np.exp(dist)
    Pb, Yb = allpars[17:-1]
    isoerr = np.exp(allpars[-1])
    lcerr = np.exp(lcerr)
    ldcoeffs = np.array([q1, q2, q3, q4])
    keblat.updatepars(m1=m1, m2=m2, z0=z0, age=age, dist=dist, ebv=ebv,
                    h0=h0, period=period, tpe=tpe, esinw=esinw,
                    ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4,
                    lcerr=lcerr, isoerr=isoerr)
    isopars = [m1, m2, z0, age, dist, ebv, h0, isoerr]
    magsmod = keblat.isofit(isopars)


    isores = (magsmod - keblat.magsobs) / keblat.emagsobs

    Lin = 1./(np.sqrt(TWOPI)*keblat.emagsobs) * np.exp(-0.5 * isores**2)
    Lout = 1./np.sqrt(TWOPI * (isoerr**2 + keblat.emagsobs**2)) * \
        np.exp(-0.5 * (keblat.magsobs-Yb)**2 / (isoerr**2+keblat.emagsobs**2))
    lnll = np.sum(np.log((1.-Pb) * Lin + Pb * Lout))
    print lnll
    lcpars = np.concatenate((np.array([m1+m2, keblat.r1+keblat.r2,
                                       keblat.r2/keblat.r1, period,
                                       tpe, esinw, ecosw, b, keblat.frat]),
                                       ldcoeffs))

    clip = keblat.clip
    lcmod, lcpol = keblat.lcfit(lcpars, keblat.jd[clip],
                              keblat.quarter[clip], keblat.flux[clip],
                            keblat.dflux[clip], keblat.crowd[clip],
                            polyorder=polyorder, ooe=ooe)

    lcres = (lcmod*lcpol - keblat.flux[clip]) / np.sqrt(keblat.dflux[clip]**2 + lcerr**2)

    if np.any(np.isinf(lcmod)):
        return -np.inf

    lnll += -0.5 * (np.sum(lcres**2) + np.sum(np.log((keblat.dflux[clip]**2 + lcerr**2))))
    lnll += -0.5 * ((isopars[5] - keblat.ebv)/(keblat.debv))**2
    #lnll += -0.5 * ((isopars[2] - keblat.z0)/(0.2 * np.log(10) * keblat.z0))**2

    return lnll

def mix_lnlike(allpars, keblat, polyorder=2, split=False, ooe=True):
    m1, m2, z0, age, dist, ebv, h0, period, tpe, esinw, \
        ecosw, b, q1, q2, q3, q4, lcerr = allpars[:17]
    #dist = np.exp(dist)
    Pb, Yb = allpars[17:-1]
    isoerr = np.exp(allpars[-1])
    lcerr = np.exp(lcerr)

    ldcoeffs = np.array([q1, q2, q3, q4])
    keblat.updatepars(m1=m1, m2=m2, z0=z0, age=age, dist=dist, ebv=ebv,
                    h0=h0, period=period, tpe=tpe, esinw=esinw,
                    ecosw=ecosw, b=b, q1=q1, q2=q2, q3=q3, q4=q4,
                    lcerr=lcerr, isoerr=isoerr)
    isopars = [m1, m2, z0, age, dist, ebv, h0, isoerr]
    magsmod = keblat.isofit(isopars)
    if np.any(np.isinf(magsmod)) or np.isinf(keblat.r1) or np.isinf(keblat.r2):
        return -np.inf #/ np.sqrt(self.emagsobs**2 + isoerr**2)

    isores = (magsmod - keblat.magsobs) / keblat.emagsobs

    # lnll = (1.-Pb)/(np.sqrt(TWOPI)*keblat.emagsobs) * np.exp(-0.5 * isores**2) + \
    #     Pb/np.sqrt(TWOPI * (isoerr**2 + keblat.emagsobs**2)) * \
    #     np.exp(-0.5 * (keblat.magsobs-Yb)**2 / (isoerr**2+keblat.emagsobs**2))
    # lnll = np.sum(np.log(lnll))

    # in case of numerical instabilities with small log sums...
    Lin = -0.5 * isores**2 + np.log((1.-Pb)/(np.sqrt(TWOPI)*keblat.emagsobs))
    Lout = -0.5 * (keblat.magsobs-Yb)**2 / (isoerr**2 + keblat.emagsobs**2) + \
                    np.log(Pb/np.sqrt(TWOPI * (isoerr**2 + keblat.emagsobs**2)))

    lnll = np.logaddexp(Lin, Lout)
    lnll = np.sum(lnll)

    #now the light curve fitting part
    lcpars = np.concatenate((np.array([m1+m2, keblat.r1+keblat.r2,
                                       keblat.r2/keblat.r1, period,
                                       tpe, esinw, ecosw, b, keblat.frat]),
                                       ldcoeffs))

    clip = keblat.clip
    lcmod, lcpol = keblat.lcfit(lcpars, keblat.jd[clip],
                              keblat.quarter[clip], keblat.flux[clip],
                            keblat.dflux[clip], keblat.crowd[clip],
                            polyorder=polyorder, ooe=ooe)

    lcres = (lcmod*lcpol - keblat.flux[clip]) / np.sqrt(keblat.dflux[clip]**2 + lcerr**2)

    if np.any(np.isinf(lcmod)):
        return -np.inf

    lnll += -0.5 * (np.sum(lcres**2) + np.sum(np.log((keblat.dflux[clip]**2 + lcerr**2))))
    lnll += -0.5 * ((isopars[5] - keblat.ebv)/(keblat.debv))**2
    #lnll += -0.5 * ((isopars[2] - keblat.z0)/(0.2 * np.log(10) * keblat.z0))**2

    if split:
        return lnll, -0.5 * (np.sum(lcres**2) + np.sum(np.log((keblat.dflux[clip]**2 + lcerr**2)))) + \
                -0.5 * ((isopars[5] - keblat.ebv)/(keblat.debv))**2
    return lnll

def mix_lnprob(allpars, keblat, polyorder=2):
    lp = mix_lnprior(allpars, keblat)
    if np.isinf(lp):
        return -np.inf, str((0,0,0))
    ll = mix_lnlike(allpars, keblat, polyorder=polyorder)
    if np.isinf(ll) or np.isnan(ll):
        return -np.inf, str((0,0,0))
    return lp+ll, str((keblat.r1, keblat.r2, keblat.frat))


def k_lnprior(lcpars, keblat):
    msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4, lcerr = lcpars
    e = np.sqrt(esinw**2 + ecosw**2)
    pars2check = np.array([msum, rsum, rrat, period, tpe, e, b, frat, q1, q2, q3, q4, lcerr])
    lcbounds = [keblat.parbounds[par] for par in parnames_dict['lc']] + \
                    [keblat.parbounds['lcerr']]
    lcbounds.pop(5)
    lcbounds[5] = [0, .99]
    lcbounds = np.array(lcbounds)
    pcheck = np.all((pars2check >= lcbounds[:,0]) & (pars2check <= lcbounds[:,1]))

    if pcheck:
        return 0.0
    else:
        return -np.inf


def k_lnlike(lcpars, keblat, polyorder=2, ooe=True):
    lcmod, lcpol = keblat.lcfit(lcpars[:13], keblat.jd[keblat.clip],
                                  keblat.quarter[keblat.clip], keblat.flux[keblat.clip],
                                keblat.dflux[keblat.clip], keblat.crowd[keblat.clip],
                                polyorder=polyorder, ooe=ooe)
    lcerr = np.exp(lcpars[-1])
    lcres = (lcmod*lcpol - keblat.flux[keblat.clip]) / np.sqrt(keblat.dflux[keblat.clip]**2 + lcerr**2)

    if np.any(np.isinf(lcmod)):
        return -np.inf

    chisq = np.sum(lcres**2) + np.sum(np.log((keblat.dflux[keblat.clip]**2 + lcerr**2)))
    #chisq += ((lcpars[3] - keblat.period)/(0.003*keblat.period))**2
    #chisq += ((lcpars[4] - keblat.tpe)/(0.03*keblat.tpe))**2
    return -0.5 * chisq

def k_lnprob(lcpars, keblat, polyorder=2):
    lp = k_lnprior(lcpars, keblat)
    if np.isinf(lp):
        return -np.inf, str((0,0,0))
    ll = k_lnlike(lcpars, keblat, polyorder=polyorder)
    if np.isnan(ll) or np.isinf(ll):
        return -np.inf, str((0,0,0))
    return lp + ll, str((0,0,0))

def get_ac_time(chains, ndim):
    """chains: (niter, nwalkers, ndim) ndarray"""
    tau = np.zeros(ndim)
    print(chains.shape)
    for ii in range(ndim):
#        try:
#        tau[ii] = autoc.integrated_time(np.nanmean(chains[:,:,ii].T, axis=0), 
#           axis=0, low=1, high=None, step=1, c=c0, fast=False)
        tau[ii] = autoc.integrated_time(np.nanmean(chains[:,:,ii].T, axis=0), axis=0, low=1, high=None, step=1, c=1, fast=False)

#        except:
#            print("Autocorrelation time couldn't be computed for {}".format(parnames_dict['lcsed'][ii]))
    return tau

def test_convergence(filename, keblat, header, footer, nwalkers, ndim, niter, 
            burnin=None, isonames=None, blob_names=None, c=10.):
    iwalker = np.arange(nwalkers)
    data = np.loadtxt(filename)
    if data.shape[0]/nwalkers < niter/20:
        print "MC file not complete... returning the last ball of walkers (1, nwalkers, ndim)"
        return  None,  None, None, None, None, None, False
    afrac = np.empty((data.shape[0]/nwalkers, nwalkers))
    logli = afrac*0.
    params = np.empty((data.shape[0]/nwalkers, nwalkers, len(isonames)))
    if blob_names is None:
        blob_names = ['r1', 'r2', 'frat'] + ['cr'+str(crq) for crq in np.unique(keblat.quarter)]
    blobs = np.empty((data.shape[0]/nwalkers, nwalkers, len(blob_names)))
#    strays = []
    for jj in iwalker:
        afrac[:,jj] = data[jj::nwalkers,2]
        logli[:,jj] = data[jj::nwalkers,3]
        for ii in range(len(blob_names)):
            blobs[:,jj,ii] = data[jj::nwalkers,ii+4+len(isonames)]
#        if len(afrac[:,jj][(afrac[:,jj]<0.1)])>=0.66*len(afrac[:,jj]):
#            strays.append(jj)

        for ii in range(len(isonames)):
            params[:, jj, ii] = data[jj::nwalkers, ii+4]#7]
#    params0 = params.copy()

    mostlike = np.where(logli == np.nanmax(logli))
    mlpars = params[:,:,:][mostlike][0]
    mlcrowd = blobs[:,:,:][mostlike][0]
    print "Max likelihood out of all samples: ", logli[:,:][mostlike]
    for kk in range(len(isonames)):
        print("""{0} = {1}""".format(str(isonames[kk]), mlpars[kk]))

    if burnin is None:
        print "Burn-in = when logli first crosses median value"
        burnin = np.where(np.nanmedian(logli, axis=1) >= np.nanmean(logli))[0][0] * 2

        
    _, bad_walkers, walker_percentiles = get_stray_walkers(params, nwalkers, ndim, burnin=burnin/10)
    strays = iwalker[bad_walkers>1]
    print("{} bad/stray walkers = {}".format(len(strays), strays))
    
    keep = iwalker[~np.in1d(iwalker, strays)]
    if len(strays)>=0.33*nwalkers:
        keep = iwalker

    print "Making plots now."
    fig = plt.figure(figsize=(16, 16))
    for ii in range(len(isonames)):
        ax = fig.add_subplot(int(len(isonames)/2)+1, 2, ii+1)
        ax.plot(params[:, :, ii])
        ax.plot(np.nanmean(params[:,:,ii].T, axis=0), 'k-', lw=2, alpha=0.2)
        ax.plot([burnin, burnin], plt.ylim(), 'y-', lw=2.0)
        ax.set_xlabel('N/10 iteration')
        ax.set_ylabel(isonames[ii])
        divider = make_axes_locatable(ax)
        axhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
        axhist.hist(params[:,:,ii], 100, histtype='step', alpha=0.6, normed=True,
                    orientation='horizontal')
        axhist.hist(params[:,:,ii].ravel(), 100, histtype='step', color='k',
                    normed=True, orientation='horizontal')
        plt.setp(axhist.get_yticklabels(), visible=False)
    ax = fig.add_subplot(int(len(isonames)/2)+1, 2, len(isonames)+1)
    ax.plot(logli)
    ax.axvline(burnin, color='y', lw=2.0)
    ax.set_xlabel('N/10 iteration')
    ax.set_ylabel('logL')
    plt.savefig(header+footer+'_parameters.png')

    tau = np.zeros((ndim))
#    tau = np.zeros((ndim, nwalkers))

    print "Making ACF plots now."
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=0.0)
    x = np.arange(params.shape[0])
    for ii in range(ndim):
        ax = fig.add_subplot(ndim/3+1, 3, ii+1)            

        print("{}".format(isonames[ii]))
        tau[ii], mean_of_acfs, acf = get_acf_tau(params[burnin:, keep, ii], c=c)
        ax.plot(acf, alpha=0.5)
        ax.plot(mean_of_acfs, 'k-', lw=1.5, alpha=0.8, label='mean(acfs)')
        ax.axvline(tau[ii], ls='--', lw=1.5)
        ax.text(tau[ii]*1.1, 0.8, '{}'.format(int(tau[ii])))
        ax.plot(x, np.exp(-x/tau[ii]), '-', lw=1.5, alpha=0.8)
        ax.set_xlabel('N/10 iteration lag')
        ax.set_ylabel('{}'.format(isonames[ii]))
        #plt.legend()

    plt.suptitle("ACF")
    plt.savefig(header+footer+'_ACF.png')
    return bad_walkers, params, blobs, mlpars, tau, afrac, logli

def tail(filename, n, ret_array=True):
    stdin, stdout = os.popen2("tail -n {} {}".format(n, filename))
    stdin.close()
    lines = stdout.read().split()
    stdout.close()
    if ret_array:
        return np.array(lines).astype(float).reshape((n, -1))
    return lines

def wc(filename):
    stdin, stdout = os.popen2("wc -l {}".format(filename))
    stdin.close()
    lines = stdout.read().split()
    stdout.close()
    return int(lines[0])

def plot_mc(filename, keblat, header, footer, nwalkers, ndim, niter, 
            burnin=None, plot=True, posteriors=False, huber_truths=[],
            isonames=None, iso_extras=False, blob_names=None, acf=True, write_mc=False,
            c=10.):
    iwalker = np.arange(nwalkers)
    data = np.loadtxt(filename)
    afrac = np.empty((data.shape[0]/nwalkers, nwalkers))
    logli = afrac*0.
    params = np.empty((data.shape[0]/nwalkers, nwalkers, len(isonames)))
    if blob_names is None:
        blob_names = ['r1', 'r2', 'frat'] + ['cr'+str(crq) for crq in np.unique(keblat.quarter)]
    blobs = np.empty((data.shape[0]/nwalkers, nwalkers, len(blob_names)))
#    strays = []
    for jj in iwalker:
        afrac[:,jj] = data[jj::nwalkers,2]
        logli[:,jj] = data[jj::nwalkers,3]
        for ii in range(len(blob_names)):
            blobs[:,jj,ii] = data[jj::nwalkers,ii+4+len(isonames)]
#        if len(afrac[:,jj][(afrac[:,jj]<0.1)])>=0.66*len(afrac[:,jj]):
#            strays.append(jj)

        for ii in range(len(isonames)):
            params[:, jj, ii] = data[jj::nwalkers, ii+4]#7]
#    params0 = params.copy()

    mostlike = np.where(logli == np.nanmax(logli))
    mlpars = params[:,:,:][mostlike][0]
    mlcrowd = blobs[:,:,:][mostlike][0]
    print "Max likelihood out of all samples: ", logli[:,:][mostlike]
    for kk in range(len(isonames)):
        print("""{0} = {1}""".format(str(isonames[kk]), mlpars[kk]))
    if burnin is None:
        _cross_ind = np.where(np.nanmedian(logli, axis=1) >= np.nanmean(logli))[0]
        if len(_cross_ind) == 0:
            _cross_ind = 0
        else:
            _cross_ind = _cross_ind[0]
        burnin = min(max(_cross_ind * 5, params.shape[0]/5), params.shape[0]/2)
    print "Burn-in = {}".format(burnin)

    _, bad_walkers, walker_percentiles = get_stray_walkers(params, nwalkers, ndim, burnin=burnin)
    strays = iwalker[bad_walkers>1]
    print("{} bad/stray walkers = {}".format(len(strays), strays))
    
    keep = iwalker[~np.in1d(iwalker, strays)]
    if len(strays)>=0.33*nwalkers:
        keep = iwalker


#    afrac, logli = afrac[burnin/10:,:], logli[burnin/10:,:]
#    params = params[burnin/10:,:,:]
#    blobs = blobs[burnin/10:,:,:]
    
    if plot:
#        from mpl_toolkits.axes_grid1 import make_axes_locatable
        print("Making trace plots now.")
        fig = plt.figure(figsize=(16, 16))
        for ii in range(len(isonames)):
            ax = fig.add_subplot(int(len(isonames)/2)+1, 2, ii+1)
            ax.plot(params[:, keep, ii])
            ax.plot(np.nanmean(params[:,keep,ii].T, axis=0), 'k-', lw=2, alpha=0.2)
            if len(strays)>0:
                ax.plot(params[:, strays, ii], alpha=0.5, linestyle='--')
            ax.plot([burnin, burnin], plt.ylim(), 'y-', lw=2.0)
            ax.set_xlabel('N/10 iteration')
            ax.set_ylabel(isonames[ii])
            divider = make_axes_locatable(ax)
            axhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
            axhist.hist(params[:,keep,ii], 100, histtype='step', alpha=0.6, normed=True,
                        orientation='horizontal')
            axhist.hist(params[:,keep,ii].ravel(), 100, histtype='step', color='k',
                        normed=True, orientation='horizontal')
            plt.setp(axhist.get_yticklabels(), visible=False)
        ax = fig.add_subplot(int(len(isonames)/2)+1, 2, len(isonames)+1)
        ax.plot(logli)
        ax.axvline(burnin, color='y', lw=2.0)
        ax.set_xlabel('N/10 iteration')
        ax.set_ylabel('logL')
        plt.suptitle("KIC {} Parameter Trace".format(keblat.kic))
        plt.savefig(header+footer+'_parameters.png')

        print("Making ml sed+lc plot now.")
        if len(mlpars) == 8:
            iso_pars = mlpars.copy()
            iso_pars[-1] = np.exp(iso_pars[-1])
            keblat.plot_sed(iso_pars, header+footer, suffix='', savefig=True)
        elif len(mlpars) == 14:
            keblat.plot_lc(mlpars, header+footer, suffix='', savefig=True, polyorder=2)
        elif len(mlpars) == 18:
            if blob_names is None:
                ll = keblat.lnlike(mlpars, qua=np.unique(keblat.quarter), crowd_fit=mlcrowd)
            keblat.plot_sedlc(mlpars, header+footer, suffix='', savefig=True, polyorder=2)
        elif len(mlpars) >=19:
            mlpars_sedlc = mlpars[:18]
            keblat.pars['sigma_ebv'] = mlpars[18]
            if ndim>19:
                keblat.pars['sigma_d'] = mlpars[19]
            #mlpars_sedlc = np.append(mlpars_sedlc, mlpars[-1])
            keblat.plot_sedlc(mlpars_sedlc, header+footer, suffix='', savefig=True, polyorder=2)
#        elif len(mlpars) == 18+len(np.unique(keblat.quarter)):
#            mlpars_sedlc = mlpars[:18]
#            mlpars_crowd = mlpars[18:]
#            ll = keblat.lnlike(mlpars_sedlc, qua=np.unique(keblat.quarter), 
#                               crowd_fit=mlpars_crowd)
#            keblat.plot_sedlc(mlpars_sedlc, header+footer, suffix='', savefig=True, polyorder=2)
        else:
            print "Not recognized mcmc run"

    bfpars = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                 zip(*np.nanpercentile(params[burnin:,keep,:].reshape((-1, ndim)),
                                    [16, 50, 84], axis=0)))
    try:
        blobpars = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.nanpercentile(blobs[burnin:,keep,:].reshape((-1, len(blob_names))),
                                      [16, 50, 84], axis=0)))
    except:
        print("No blobs.")
    print "MCMC result: "
    print "Accep. Frac = ", np.nanmean(afrac[burnin:, keep])
    for kk in range(len(isonames)):
        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(isonames[kk]),
              bfpars[kk]))
    for kk in range(len(blob_names)):
        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(blob_names[kk]),
              blobpars[kk]))
    # mostlike = np.where(logli[:,keep] == np.max(logli[:,keep]))
    # mlpars = params[:,keep,:][mostlike][0]
    # print "Max likelihood: ", logli[:,keep][mostlike]
    # for kk in range(len(isonames)):
    #     print("""{0} = {1}""".format(str(isonames[kk]), mlpars[kk]))
    if write_mc:
#        _mlpars = mlpars.copy()
#        _mlpars[[4,16,17]][_mlpars[[4,16,17]]<0] = np.exp(_mlpars[[4,16,17]][_mlpars[[4,16,17]]<0])
        residuals = abs(keblat.lnlike(mlpars[:18], lc_constraints=None, 
                              qua=np.unique(keblat.quarter), polyorder=2, residual=True))

        changeofvar_names = ['m1', 'm2', 'inc', 'e']
        params_changeofvar = np.zeros((params.shape[0], params.shape[1], len(changeofvar_names)))
        params_changeofvar[:,:,0], params_changeofvar[:,:,1] = keblat.sumrat_to_12(params[:,:,0], params[:,:,1])
        params_changeofvar[:,:,2] = keblat.get_inc(params[:,:,11], blobs[:,:,0], keblat.get_a(params[:,:,7], params[:,:,0]))
        params_changeofvar[:,:,3] = np.sqrt(params[:,:,9]**2 + params[:,:,10]**2)
        bfpars_changeofvar = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.nanpercentile(params_changeofvar[burnin:, keep, :].reshape((-1, len(changeofvar_names))),
                                        [16, 50, 84], axis=0)))
        
        bffile = open(header+footer+'.mcpars', "w")
        mlpars_write = mlpars.copy()
        if len(mlpars)==19:
            mlpars_write = np.append(mlpars_write, 0.0)
        bffile.write("""{} {}\n""".format(" ".join([str(mp) for mp in mlpars_write]), " ".join([str(bp) for bp in keblat.get_blobs()])))
        for kk in range(len(isonames)):
    #        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(isonames[kk]),
    #              bfpars[kk]))
            bffile.write("""#{0} = {1[0]} +{1[1]} -{1[2]}\n""".format(str(isonames[kk]),
                  bfpars[kk]))
        for kk in range(len(blob_names)):
    #        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(blob_names[kk]),
    #              blobpars[kk]))
            bffile.write("""#{0} = {1[0]} +{1[1]} -{1[2]}\n""".format(str(blob_names[kk]),
                  blobpars[kk]))
        for kk in range(len(changeofvar_names)):
    #        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(changeofvar_names[kk]),
    #              bfpars_changeofvar[kk]))
            bffile.write("""#{0} = {1[0]} +{1[1]} -{1[2]}\n""".format(str(changeofvar_names[kk]),
                  bfpars_changeofvar[kk]))
        bffile.close()

        ##### making median residuals output ######
        chunks = identify_gaps(keblat.cadnum, retbounds_inds=True)
        chunks = np.delete(chunks, np.where(np.diff(chunks)<2)[0])
        x = keblat.jd.copy()
        y = keblat.flux.copy()
        x[keblat.clip] = np.nan
        y[keblat.clip] = np.nan
        flckr = np.zeros((len(chunks)-2))
        for ii in range(len(chunks)-2):                                  
            _smoo, _flick = get_flicker(y[chunks[ii]:chunks[ii+1]], mad=False, window=24)
            flckr[ii] = _flick  
        residuals *= np.concatenate((np.sqrt(keblat.emagsobs**2+mlpars[:18][-1]**2), 
                                     np.sqrt(keblat.fluxerr[keblat.clip]**2+mlpars[:18][-2]**2)))
    
        sed_res = np.nanpercentile(residuals[:len(keblat.magsobs)], [16, 50, 84])
    
        lcmod, lcpol = keblat.lcfit(keblat.getpars('lc')[:13], keblat.jd[keblat.clip], keblat.quarter[keblat.clip], 
                                 keblat.flux[keblat.clip], keblat.dflux[keblat.clip], keblat.crowd[keblat.clip], 
                                 polyorder=2)
        eclipse = (lcmod<1)
        ooe = abs(keblat.flux[keblat.clip][~eclipse] - lcmod[~eclipse]*lcpol[~eclipse])
        ine = abs(keblat.flux[keblat.clip][eclipse] - lcmod[eclipse]*lcpol[eclipse])
        ooe_res = np.nanpercentile(ooe, [16, 50, 84])
        ine_res = np.nanpercentile(ine, [16, 50, 84])
        mrfile = open(header+footer+'.medres', 'w')
        mrfile.write("{} {} {} {} {} {}".format(keblat.kic, " ".join([str(zz) for zz in ine_res]), 
               " ".join([str(zz) for zz in ooe_res]), 
               " ".join([str(zz) for zz in sed_res]), np.nanmedian(abs(np.diff(keblat.flux[~keblat.clip]))), 
               np.nanmedian(flckr)))
        mrfile.close()

    if posteriors:
        import corner
        plt.figure(figsize=(16, 16))
        thin_by = np.clip((params.shape[0]-burnin)*params.shape[1]/50000, 1, 50000)
        print("burned-in param matrix is {}; thinning by {}".format(params[burnin:, :, :].shape, thin_by))
#        samples = np.concatenate((params[burnin/10::thin_by, :, :], blobs[burnin/10::thin_by, :, :]), axis=2)
#        post_inds = np.arange(len(isonames)+len(blob_names))
#        post_inds = np.delete(post_inds, np.where(np.std(samples, axis=(0,1)) == 0)[0])
#        try:
#            corner.corner(samples[:,keep,:][:,:,post_inds].reshape((-1, len(post_inds))), 
#                          labels=np.append(isonames, blob_names)[post_inds], quantiles=[0.16, 0.5, 0.84],
#                          show_titles=True, title_kwargs={"fontsize": 11})
#            plt.savefig(header+footer+'_posteriors.png')
        post_inds = np.arange(len(isonames))
        post_inds = np.delete(post_inds, 
                              np.where(np.nanmedian(abs(np.diff(params[burnin::thin_by, :, :], axis=0)), axis=(0,1)) < 1e-12)[0])
        try:
            corner.corner(params[burnin::thin_by, keep, :][:,:,post_inds].reshape((-1, len(post_inds))), 
                          labels=np.array(isonames)[post_inds], quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_kwargs={"fontsize": 11})
            plt.suptitle("KIC {}".format(keblat.kic))
            plt.savefig(header+footer+'_posteriors.png')
        except Exception, e:
            print(str(e), post_inds)
    tau = np.zeros((ndim))
#    tau = np.zeros((ndim, nwalkers))
    if acf:
        print "Making ACF plots now."
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(hspace=0.0)
        x = np.arange(params.shape[0])
        for ii in range(ndim):
            ax = fig.add_subplot(ndim/3+1, 3, ii+1)            

#            try:
#                tau[ii] = autoc.integrated_time(np.nanmean(params[:,keep,ii].T, 
#                   axis=0), axis=0, low=1, high=None, step=1, c=1, fast=False)
#            except:
#                print("Autocorr time could not be computed for {}".format(isonames[ii]))
#            for jj in range(nwalkers):
#                acf = autocorr(params[:,jj,ii]-np.nanmedian(params[:,jj,ii]))
#                acf /= np.nanmax(acf)
#                ax.plot(acf, alpha=0.5)
            print("{}".format(isonames[ii]))
            tau[ii], mean_of_acfs, acf = get_acf_tau(params[:, keep, ii], c=c)
            ax.plot(acf, alpha=0.5)
            ax.plot(mean_of_acfs, 'k-', lw=1.5, alpha=0.8, label='mean(acfs)')

#            mean_chain = np.nanmean(params[:,keep,ii].T, axis=0)
#            acf_of_mean = autocorr(mean_chain-np.nanmedian(mean_chain))
#            acf_of_mean /= np.nanmax(acf_of_mean)
#            ax.plot(acf_of_mean, 'k--', lw=1.5, alpha=0.8, label='acf(mean chain)')
            ax.text(tau[ii]*1.1, 0.8, '{}'.format(int(tau[ii])))
            ax.plot(x, np.exp(-x/tau[ii]), '-', lw=1.5, alpha=0.8)
            ax.set_xlabel('N/10 iteration lag')
            ax.set_ylabel('{}'.format(isonames[ii]))
#            if ii == ndim-1:
#                plt.legend()

#                ax.plot([burnin/10, burnin/10], plt.ylim(), 'y-', lw=2.0)
#                ax.set_xlabel('N/10 iteration')
#                ax.set_ylabel(isonames[ii])
#                divider = make_axes_locatable(ax)
#                axhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
#                axhist.hist(params[:,:,ii], 100, histtype='step', alpha=0.6, normed=True,
#                            orientation='horizontal')
#                axhist.hist(params[:,:,ii].ravel(), 100, histtype='step', color='k',
#                            normed=True, orientation='horizontal')
#                plt.setp(axhist.get_yticklabels(), visible=False)
        plt.suptitle("KIC {} ACF".format(keblat.kic))
        plt.savefig(header+footer+'_ACF.png')
    return bad_walkers, params, blobs, mlpars, tau, afrac, logli

def plot_mc_pandas(filename, keblat, header, footer, nwalkers, ndim, niter, 
            burnin=40000, plot=True, posteriors=False, huber_truths=[],
            isonames=None, iso_extras=False, blob_names=None, acf=True, write_mc=False):
    iwalker = np.arange(nwalkers)
    if blob_names is None:
        blob_names = ['r1', 'r2', 'frat'] + ['cr'+str(crq) for crq in np.unique(keblat.quarter)]

    data = pd.read_csv(filename, delim_whitespace=True, header=None, 
                       names=['iter', 'iwalker', 'accep_frac', 'logli']+isonames+blob_names)

    afrac = np.empty((data.shape[0]/nwalkers, nwalkers))
    logli = afrac*0.
    params = np.empty((data.shape[0]/nwalkers, nwalkers, len(isonames)))
    blobs = np.empty((data.shape[0]/nwalkers, nwalkers, len(blob_names)))
#    strays = []
    for jj in iwalker:
        afrac[:,jj] = data['accep_frac'].values[jj::nwalkers]
        logli[:,jj] = data['logli'].values[jj::nwalkers]
        for ii in range(len(blob_names)):
            blobs[:,jj,ii] = data[blob_names[ii]].values[jj::nwalkers]
#        if len(afrac[:,jj][(afrac[:,jj]<0.1)])>=0.66*len(afrac[:,jj]):
#            strays.append(jj)

        for ii in range(len(isonames)):
            params[:, jj, ii] = data[isonames[ii]].values[jj::nwalkers]
#    params0 = params.copy()

    mostlike = np.where(data['logli'].values == np.nanmax(data['logli']))[0]
    mlpars = data[isonames].values[mostlike][0]
    mlcrowd = data[blob_names].values[mostlike][0]
    print "Max likelihood out of all samples: ", data['logli'].values[mostlike]
    for kk in range(len(isonames)):
        print("""{0} = {1}""".format(str(isonames[kk]), mlpars[kk]))
    if burnin/10>=params.shape[0]:
        print "Burn-in shorter than length of MCMC run, adjusting..."
        burnin = params.shape[0]*3/4*10
        
        
    _, bad_walkers, walker_percentiles = get_stray_walkers(data[isonames].values.reshape((-1, nwalkers, ndim)), nwalkers, ndim, burnin=burnin/10)
    strays = iwalker[bad_walkers>1]
    print("{} bad/stray walkers = {}".format(len(strays), strays))
    
    keep = iwalker[~np.in1d(iwalker, strays)]
    if len(strays)>=0.33*nwalkers:
        keep = iwalker


#    afrac, logli = afrac[burnin/10:,:], logli[burnin/10:,:]
#    params = params[burnin/10:,:,:]
#    blobs = blobs[burnin/10:,:,:]
    
    if plot:
#        from mpl_toolkits.axes_grid1 import make_axes_locatable
        print "Making plots now."
        fig = plt.figure(figsize=(16, 16))
        for ii in range(len(isonames)):
            ax = fig.add_subplot(int(len(isonames)/2)+1, 2, ii+1)
            ax.plot(data[isonames[ii]].values.reshape((-1, nwalkers)))
            #ax.plot(np.nanmean(params[:,:,ii].T, axis=0), 'k-', lw=2, alpha=0.2)
            ax.plot([burnin/10, burnin/10], plt.ylim(), 'y-', lw=2.0)
            ax.set_xlabel('N/10 iteration')
            ax.set_ylabel(isonames[ii])
            divider = make_axes_locatable(ax)
            axhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
            axhist.hist(data[isonames[ii]].values.reshape((-1, nwalkers)), 100, histtype='step', alpha=0.6, normed=True,
                        orientation='horizontal')
            axhist.hist(data[isonames[ii]].values, 100, histtype='step', color='k',
                        normed=True, orientation='horizontal')
            plt.setp(axhist.get_yticklabels(), visible=False)
        plt.savefig(header+footer+'_parameters.png')


    bfpars = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                 zip(*np.nanpercentile(data[isonames].values.reshape((-1, nwalkers, ndim))[burnin/10:,keep,:].reshape((-1, ndim)),
                                    [16, 50, 84], axis=0)))
    blobpars = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.nanpercentile(data[blob_names].values.reshape((-1, nwalkers, len(blob_names)))[burnin/10:,keep,:].reshape((-1, len(blob_names))),
                                      [16, 50, 84], axis=0)))
    print "MCMC result: "
    print "Accep. Frac = ", np.nanmean(data['accept_frac'].values.reshape((-1, nwalkers))[burnin/10:, keep])
    for kk in range(len(isonames)):
        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(isonames[kk]),
              bfpars[kk]))
    for kk in range(len(blob_names)):
        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(blob_names[kk]),
              blobpars[kk]))
    # mostlike = np.where(logli[:,keep] == np.max(logli[:,keep]))
    # mlpars = params[:,keep,:][mostlike][0]
    # print "Max likelihood: ", logli[:,keep][mostlike]
    # for kk in range(len(isonames)):
    #     print("""{0} = {1}""".format(str(isonames[kk]), mlpars[kk]))
    if write_mc:
        _mlpars = mlpars.copy()
        _mlpars[[4,16,17]][_mlpars[[4,16,17]]<0] = np.exp(_mlpars[[4,16,17]][_mlpars[[4,16,17]]<0])
        _ = keblat.lnlike(_mlpars[:18], lc_constraints=None, qua=np.unique(keblat.quarter), residual=False)
    
        changeofvar_names = ['m1', 'm2', 'inc', 'e']
        params_changeofvar = np.zeros((params.shape[0], params.shape[1], len(changeofvar_names)))
        params_changeofvar[:,:,0], params_changeofvar[:,:,1] = keblat.sumrat_to_12(params[:,:,0], params[:,:,1])
        params_changeofvar[:,:,2] = keblat.get_inc(params[:,:,11], blobs[:,:,0]+blobs[:,:,1], keblat.get_a(params[:,:,7], params[:,:,0]))
        params_changeofvar[:,:,3] = np.sqrt(params[:,:,9]**2 + params[:,:,10]**2)
        bfpars_changeofvar = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(params_changeofvar[burnin/10:, keep, :].reshape((-1, len(changeofvar_names))),
                                        [16, 50, 84], axis=0)))
        
        bffile = open(header+footer+'.mcpars', "w")
        bffile.write("""{}\n""".format(" ".join([str(mp) for mp in mlpars])))
        for kk in range(len(isonames)):
    #        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(isonames[kk]),
    #              bfpars[kk]))
            bffile.write("""#{0} = {1[0]} +{1[1]} -{1[2]}\n""".format(str(isonames[kk]),
                  bfpars[kk]))
        for kk in range(len(blob_names)):
    #        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(blob_names[kk]),
    #              blobpars[kk]))
            bffile.write("""#{0} = {1[0]} +{1[1]} -{1[2]}\n""".format(str(blob_names[kk]),
                  blobpars[kk]))
        for kk in range(len(changeofvar_names)):
    #        print("""{0} = {1[0]} +{1[1]} -{1[2]}""".format(str(changeofvar_names[kk]),
    #              bfpars_changeofvar[kk]))
            bffile.write("""#{0} = {1[0]} +{1[1]} -{1[2]}\n""".format(str(changeofvar_names[kk]),
                  bfpars_changeofvar[kk]))
        bffile.close()


    if plot:
        if len(mlpars) == 8:
            iso_pars = mlpars.copy()
            iso_pars[-1] = np.exp(iso_pars[-1])
            keblat.plot_sed(iso_pars, header+footer, suffix='', savefig=True)
        elif len(mlpars) == 14:
            keblat.plot_lc(mlpars, header+footer, suffix='', savefig=True, polyorder=2)
        elif len(mlpars) == 18:
            if blob_names is None:
                ll = keblat.lnlike(mlpars, qua=np.unique(keblat.quarter), crowd_fit=mlcrowd)
            keblat.plot_sedlc(mlpars, header+footer, suffix='', savefig=True, polyorder=2)
        elif len(mlpars) == 20:
            mlpars_sedlc = mlpars[:18]
            keblat.pars['sigma_ebv'] = mlpars[18]
            keblat.pars['sigma_d'] = mlpars[19]
            #mlpars_sedlc = np.append(mlpars_sedlc, mlpars[-1])
            keblat.plot_sedlc(mlpars_sedlc, header+footer, suffix='', savefig=True, polyorder=2)
#        elif len(mlpars) == 18+len(np.unique(keblat.quarter)):
#            mlpars_sedlc = mlpars[:18]
#            mlpars_crowd = mlpars[18:]
#            ll = keblat.lnlike(mlpars_sedlc, qua=np.unique(keblat.quarter), 
#                               crowd_fit=mlpars_crowd)
#            keblat.plot_sedlc(mlpars_sedlc, header+footer, suffix='', savefig=True, polyorder=2)
        else:
            print "Not recognized mcmc run"
    if posteriors:
        import corner
        plt.figure(figsize=(16, 16))
        thin_by = np.clip((params.shape[0]-burnin/10)*params.shape[1]/50000, 1, 50000)
        print("burned-in param matrix is {}; thinning by {}".format(params[burnin/10:, :, :].shape, thin_by))
#        samples = np.concatenate((params[burnin/10::thin_by, :, :], blobs[burnin/10::thin_by, :, :]), axis=2)
#        post_inds = np.arange(len(isonames)+len(blob_names))
#        post_inds = np.delete(post_inds, np.where(np.std(samples, axis=(0,1)) == 0)[0])
#        try:
#            corner.corner(samples[:,keep,:][:,:,post_inds].reshape((-1, len(post_inds))), 
#                          labels=np.append(isonames, blob_names)[post_inds], quantiles=[0.16, 0.5, 0.84],
#                          show_titles=True, title_kwargs={"fontsize": 11})
#            plt.savefig(header+footer+'_posteriors.png')
        post_inds = np.arange(len(isonames))
        post_inds = np.delete(post_inds, np.where(np.nanstd(params[burnin/10::thin_by, :, :], axis=(0,1)) == 0)[0])
        try:
            corner.corner(params[burnin/10::thin_by, keep, :][:,:,post_inds].reshape((-1, len(post_inds))), 
                          labels=isonames[post_inds], quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_kwargs={"fontsize": 11})
            plt.savefig(header+footer+'_posteriors.png')
        except Exception, e:
            print str(e)
    tau = np.zeros((ndim))
#    tau = np.zeros((ndim, nwalkers))
    if acf:
        print "Making ACF plots now."
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(hspace=0.0)
        x = np.arange(params.shape[0])
        for ii in range(ndim):
            ax = fig.add_subplot(ndim/3+1, 3, ii+1)            

#            try:
#                tau[ii] = autoc.integrated_time(np.nanmean(params[:,keep,ii].T, 
#                   axis=0), axis=0, low=1, high=None, step=1, c=1, fast=False)
#            except:
#                print("Autocorr time could not be computed for {}".format(isonames[ii]))
#            for jj in range(nwalkers):
#                acf = autocorr(params[:,jj,ii]-np.nanmedian(params[:,jj,ii]))
#                acf /= np.nanmax(acf)
#                ax.plot(acf, alpha=0.5)
            print("{}".format(isonames[ii]))
            tau[ii], mean_of_acfs, acf = get_acf_tau(params[:, keep, ii])
            ax.plot(acf, alpha=0.5)
            ax.plot(mean_of_acfs, 'k-', lw=1.5, alpha=0.8, label='mean(acfs)')

#            mean_chain = np.nanmean(params[:,keep,ii].T, axis=0)
#            acf_of_mean = autocorr(mean_chain-np.nanmedian(mean_chain))
#            acf_of_mean /= np.nanmax(acf_of_mean)
#            ax.plot(acf_of_mean, 'k--', lw=1.5, alpha=0.8, label='acf(mean chain)')
            ax.plot(x, np.exp(-x/tau[ii]), 'c-', lw=1.5, alpha=0.8, label='tau exp')
            ax.set_xlabel('N/10 iteration lag')
            ax.set_ylabel('{}'.format(isonames[ii]))
            if ii == ndim-1:
                plt.legend()

#                ax.plot([burnin/10, burnin/10], plt.ylim(), 'y-', lw=2.0)
#                ax.set_xlabel('N/10 iteration')
#                ax.set_ylabel(isonames[ii])
#                divider = make_axes_locatable(ax)
#                axhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
#                axhist.hist(params[:,:,ii], 100, histtype='step', alpha=0.6, normed=True,
#                            orientation='horizontal')
#                axhist.hist(params[:,:,ii].ravel(), 100, histtype='step', color='k',
#                            normed=True, orientation='horizontal')
#                plt.setp(axhist.get_yticklabels(), visible=False)
        plt.suptitle("ACF")
        plt.savefig(header+footer+'_ACF.png')
    return bad_walkers, params, blobs, mlpars, tau, np.mean(afrac[:, keep]), True


def auto_window(taus, c):
    """
    Find the smallest window such that m >= c * tau (Sokal 1989)   
    
    Function directly from dfm.io/posts/autocorr/"""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def get_acf_tau(y, c=7.0):
    """
    Find integrated autocorrelation time by taking the mean of ACFs 
    of a 2d time series y[time, walker] (rather than taking the ACF of the 
    mean of the time series)
    
    Adapted from dfm.io/posts/autocorr/
    """
    if np.nansum(y) == 0 or np.nanstd(y) < 1e-12:
        print("Autocorr time could not be computed. Check your input.")
        return 0, np.zeros(len(y)), np.zeros(len(y))
    acf = y*0.
    for ii in range(y.shape[1]):
        acf[:,ii] = autocorr(y[:,ii] - np.nanmean(y[:,ii]))
        acf[:,ii] /= acf[0,ii] #np.nanmax(acf[ii,:])
    f = np.nansum(acf, axis=1) / y.shape[1]
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window], f, acf

def get_stray_walkers(mcmcchains, nwalkers, ndim, burnin=0, threshold=10., bfpars=None):
    iwalker=np.arange(nwalkers)
    walker_percentiles = np.zeros((ndim, nwalkers, 5))
    if bfpars is None:
            bfpars = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                 zip(*np.nanpercentile(mcmcchains[burnin:,:,:].reshape((-1, ndim)),
                                    [16, 50, 84], axis=0)))

    bad_walkers = np.zeros(nwalkers)
    for ii in range(ndim):
        for jj in range(nwalkers):
            walker_percentiles[ii,jj,:3] = np.nanpercentile(mcmcchains[burnin:, jj, ii], [16, 50, 84])
            walker_percentiles[ii,jj,3] = np.nanmean(mcmcchains[burnin:, jj, ii])
            walker_percentiles[ii,jj,4] = np.nanstd(mcmcchains[burnin:, jj, ii])
#            print(np.nanpercentile(mcmcchains[burnin:, jj, ii], [16, 50, 84]))
            bad_walkers[jj] += ((walker_percentiles[ii,jj,2]-walker_percentiles[ii,jj,1] < bfpars[ii][1]/threshold) * \
                                (walker_percentiles[ii,jj,1]-walker_percentiles[ii,jj,0] < bfpars[ii][2]/threshold)) | \
                               (abs(walker_percentiles[ii,jj,3] - bfpars[ii][0]) > threshold * bfpars[ii][1])
    return iwalker, bad_walkers, walker_percentiles

def write_sedlc_pars(keblat, opt_allpars, crowd, fname=None):
    kepQs = np.arange(18)
    crowd_vals = np.zeros(len(kepQs))
    crowd_vals[np.in1d(kepQs, np.unique(keblat.quarter))] = crowd
    _ = keblat.lnlike(opt_allpars[:18], qua=np.unique(keblat.quarter))
    if fname is not None:
        outf = open(fname, "w")
        outf.write("""{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n""".format(keblat.kic, keblat.morph, 
                                                       " ".join(str(jj) for jj in opt_allpars),
                                                        keblat.r1, keblat.r2, keblat.frat, 
                                                        keblat.temp1, keblat.temp2, 
                                                        keblat.logg1, keblat.logg2,
                                                        keblat.pars['inc'], keblat.mact1, keblat.mact2,
                                                        " ".join(str(zz) for zz in crowd_vals)))
        outf.close()
    else:
        print("""{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n""".format(keblat.kic, keblat.morph, 
                                                   " ".join(str(jj) for jj in opt_allpars),
                                                    keblat.r1, keblat.r2, keblat.frat, 
                                                    keblat.temp1, keblat.temp2, 
                                                    keblat.logg1, keblat.logg2,
                                                    keblat.pars['inc'], keblat.mact1, keblat.mact2,
                                                    " ".join(str(zz) for zz in crowd_vals)))
    return

def make_p0_ball(p_init, ndim, nwalkers, scale=1e-4, period_scale=1e-7, mass_scale=1e-4, age_scale=1e-5):
    p0_scale = np.ones(ndim)*scale
    p0_scale[[0,1]] = mass_scale
    p0_scale[3] = age_scale
    p0_scale[[7,8]] = period_scale
    p0 = [p_init + p0_scale * p_init * np.random.randn(ndim) for ii in range(nwalkers)]
    p0 = np.array(p0)
#    p0[:,6] = 119.
    #p0[:,16] = 1e-4
    p0[:,12:16] = np.clip(p0[:,12:16], 0., 1.0)
#    p0[:,18:] = np.clip(p0[:,18:], 0, 1)
#    p0[:,18] = np.clip(p0[:,18], -12, 2)
#    if ndim>19:
#        p0[:,19] = np.clip(p0[:,19], -1, 7)
    return p0

def estimate_tpe_sep(jd, flux, period):
    phase = jd % period / period
    breaks = np.where(np.diff(phase)>0.1)[0]
    breaks = np.append(0, breaks)
    breaks = np.append(breaks, len(phase))
    ii=0
    _sorted = np.argsort(phase[breaks[ii]:breaks[ii+1]])
    smoothed = scipy.signal.savgol_filter(keblat.flux[breaks[ii]:breaks[ii+1]][_sorted], 45, polyorder=2)
    return

