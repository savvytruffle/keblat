import matplotlib
matplotlib.use('agg')
import sys, itertools, time, os
print(os.uname())
import emcee
from emcee.utils import MPIPool
#from helper_funcs import *
from eb_fitting import *

kic = int(sys.argv[1])
try:
    prefix = sys.argv[2]
except:
    prefix = '/astro/store/gradscratch/tmp/windemut/eb_fitting/'

prefix = prefix + str(kic)+'/'
#kic = int(float(sys.argv[1]))
#period, tpe, esinw, ecosw, rsum, rrat, b, frat, q1, q2, q3, q4 = np.array(sys.argv[2:-2], dtype=float)
#nwalkers, niter = int(sys.argv[-2]), int(sys.argv[-1])

clobber_lc=False #overwrite LC only fits?
clobber_sed=True #overwrite SED only fits?
kiclist, perlist, pdeplist, sdeplist, morphlist = np.loadtxt('data/kebproperties_0216.dat',
                                          usecols=(0, 1, 3, 4, 8), unpack=True, delimiter=';')

#goodlist = (morphlist<0.6) & (pdeplist>0.1) & (sdeplist>0.01) & (perlist > 1.)
goodlist = (perlist>0)

excludelist = get_excludelist(fname='data/sed_flag_file_0328')

keblat = Keblat(preload=False)
#keblat.loadiso2()
keblat.loadiso2(isoname='isodata_jun4.dat')
keblat.loadsed(sedfile='data/kepsedall_0216.dat')
keblat.loadvkeb(filename='data/kebproperties_0216.dat')
goodlist_ind = np.where(kiclist[goodlist].astype(int) == kic)[0]
if len(goodlist_ind)>1:
    goodlist_ind=goodlist_ind[0]

goodv = keblat.kiclookup(kic, target=keblat.vkeb[:, 0])
keblat.loadlc(kic, keblat.vkeb[goodv, [1, 2, 5, 6, 7]], clip_tol=2.0)
magsobs, emagsobs, extinction, glat, z0 = keblat.getmags(kic)
ebv = extinction[0]
if np.isnan(z0):
    z0 = keblat.zsun
#print "Loading SED data, excluding ", excludelist[kic]
try:
    exclusive = excludelist[kic]
except:
    exclusive = []
keblat.isoprep(magsobs, emagsobs, extinction, glat, z0, exclude=exclusive)#exclude=[]) #excludelist[goodlist_ind])#'gmag','rmag','imag','zmag'])
period, tpe = keblat.vkeb[goodv, 1], keblat.vkeb[goodv, 2]
ecosw, esinw = keblat.vkeb[goodv, -2], keblat.vkeb[goodv, -1]
if ecosw == 0: ecosw = 1e-5
if esinw == 0: esinw = 1e-5
frat = (keblat.vkeb[goodv, 4]/keblat.vkeb[goodv, 3])

ebv_arr, ebv_sig, ebv_dist_bounds, ebv_bounds = None, None, None, None#get3dmap(kic)

if keblat.swidth < 0.:
    print "No secondary eclipses detected. Exiting."
    sys.exit()

#if np.median(np.unique(keblat.crowd)) < 0.5:
#    print "Crowding > 0.5. Exiting."
#    sys.exit()

if (max(keblat.pwidth, keblat.swidth) > 0.04) and (keblat.pwidth+keblat.swidth > 0.091):
    clip_tol = 1.4
#elif ((keblat.pw3idth > 0.01) and (keblat.swidth > 0.01)) or (keblat.pwidth+keblat.swidth>:
#    clip_tol = 1.5
else:
    clip_tol = 1.7

print("Clip tolerance = {0}".format(clip_tol))
keblat.updatephase(tpe, period, clip_tol=clip_tol)


check_dir_exists(prefix)
keblat.start_errf(prefix+'lcfit.err')

# rvdata = np.loadtxt('data/{0}.rv'.format(kic), delimiter=';')

# uncomment the code segment below if want to fit RV
# //load rv data
# //make init guess for masses + K offset
# m1, m2, k0 = keblat.rvprep(rvdata[:,0], rvdata[:,1], rvdata[:,3], rvdata[:,2], rvdata[:,4])
# //run light-curve opt first
# //make sure keblat.pars are updated...
# lcmod, lcpol = keblat.lcfit(opt_lcpars, keblat.jd[keblat.clip].....)
# //update the bounds to make them stricter
# keblat.updatebounds('period', 'tpe', 'esinw', 'ecosw')
# rvpars = [m1+m2, m2/m1, opt_lcpars[3], opt_lcpars[4], opt_lcpars[5], opt_lcpars[6], keblat.pars['inc'], k0, 0]
# //optimize rvparameters using opt_lc + init rv guesses
# opt_rvpars = opt_rv(msum=m1+m2, mrat=m2/m1, period=opt_lcpars[3], tpe=opt_lcpars[4], esinw=opt_lcpars[5],
#                       ecosw=opt_lcpars[6], inc=keblat.pars['inc'], k0=k0, rverr=0)
# //fix msum from rv fit to lc fit
# opt_lcpars[0] = opt_rvpars[0]
# lcpars2 = opt_lc(opt_lcpars, keblat.jd, keblat.phase, keblat.flux, keblat.fluxerr, keblat.crowd, keblat.clip, set_upperb = 2.0, vary_msum=False)
# //then optimize both simultaneously
#opt_lcrvpars = opt_lcrv(keblat,msum=opt_rvpars[0], mrat=opt_rvpars[1],
#                         rsum=lcpars2[1], rrat=lcpars2[2], period=lcpars2[3],
#                         tpe=lcpars2[4], esinw=lcpars2[5], ecosw=lcpars2[6],
#                         b=lcpars2[7], frat=lcpars2[8], q1=lcpars2[-4],
#                         q2=lcpars2[-3], q3=lcpars2[-2], q4=lcpars2[-1],
#                         lcerr=0.0, k0=opt_rvpars[-2], rverr=0.)

q1, q2, q3, q4 = 0.01, 0.01, 0.01, 0.01
age, h0, dist = 9.2, 119., 850.
chunks = identify_gaps(keblat.cadnum, retbounds_inds=True)
chunks = np.delete(chunks, np.where(np.diff(chunks)<2)[0])
lcchi2_threshold = 5./np.nanmedian(np.array([np.nanmedian(abs(keblat.flux[chunks[ii]:chunks[ii+1]] -
                                                          np.nanmedian(keblat.flux[chunks[ii]:chunks[ii+1]])))
                                          for ii in range(len(chunks)-1)]))

print(blah)
#if os.path.isfile(prefix+'allpars.lmfit') or os.path.isfile(prefix+'allpars_wc.lmfit'):
#    print("opt lc/sed allpars.lmfit exists.")
#    sys.exit()

#finalkics = np.loadtxt('goodEBs_nov2016.list')
#final_ind = finalkics[:,0] == kic
#opt_allpars = finalkics[final_ind,1:].ravel()[:18]
#_opt_allpars = opt_allpars.copy()
#opt_allpars[0] = _opt_allpars[1]+_opt_allpars[0]
#opt_allpars[1] = _opt_allpars[1]/_opt_allpars[0]
##if opt_allpars[1]>1.:
#    #opt_allpars[1] = 0.95
#
#keblat.plot_sedlc(opt_allpars, prefix, savefig=False)
#if keblat.temp1<keblat.temp2:
#    keblat.parbounds['mrat'] = [0.0085, 1.0]
#    if opt_allpars[1]>1.:
#        opt_allpars[1] = 0.99
#else:
#    keblat.parbounds['mrat'] = [0.0085, 1.5]
#if abs(keblat.pars['ecosw'])<0.01:
#    keblat.parbounds['ecosw'] = [-0.01, 0.01]
#elif abs(keblat.pars['ecosw'])<0.03:
#    keblat.parbounds['ecosw'] = [-0.1, 0.1]
#else:
#    keblat.updatebounds('ecosw', partol=0.01)
#if abs(keblat.pars['esinw'])<0.01:
#    keblat.parbounds['esinw'] = [-0.1, 0.1]
#else:
#    #keblat.updatebounds('esinw', partol=0.01)
#    keblat.parbounds['esinw'] =  [-np.sqrt(.9**2-keblat.pars['esinw']**2), 
#                    np.sqrt(.9**2-keblat.pars['esinw']**2)]
#keblat.updatebounds('tpe', 'period', partol=0.0001)
#opt_allpars0 = opt_allpars.copy()
#_kwargs = dict(zip(parnames_dict['lcsed'], opt_allpars0))
#_kwargs.update({'init_porder': 2, 'init_varyb': True, 
#                'init_varyew': True, 'init_varyza': True, 'lc_constraints': None})
#opt_allpars0 = opt_sedlc(keblat, **_kwargs)
#opt_allpars0 = np.asarray(opt_allpars0)
#allres = lnlike_lmfit(opt_allpars, keblat, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
#allres0 = lnlike_lmfit(opt_allpars0, keblat, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
#
#if np.sum(allres0**2) < np.sum(allres**2):
#    print("Improvement")
#    opt_allpars = opt_allpars0.copy()
#    keblat.plot_sedlc(opt_allpars, prefix, savefig=False)
#
#crowd = keblat._crowdsap.ravel()
#kepQs = np.arange(18)
#crowd_vals = np.zeros(len(kepQs))
#crowd_vals[np.in1d(kepQs, np.unique(keblat.quarter))] = crowd
#print("""{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n""".format(kic, 
#                                                   " ".join(str(jj) for jj in opt_allpars),
#                                                    keblat.r1, keblat.r2, keblat.frat, 
#                                                    keblat.temp1, keblat.temp2, 
#                                                    keblat.logg1, keblat.logg2,
#                                                    keblat.pars['inc'], 
#                                                    " ".join(str(zz) for zz in crowd_vals)))
try:
    opt_lcpars = np.loadtxt(prefix+'lcpars_wcrowd.lmfit2')
    crowd = opt_lcpars[13:]
    crowd_fits = keblat.broadcast_crowd(keblat.quarter, opt_lcpars[13:])
    keblat.crowd = crowd_fits
    opt_lcpars = opt_lcpars[:13]  
except:
    opt_lcpars = np.loadtxt(prefix+'lcpars.lmfit2')
    crowd = keblat._crowdsap.ravel()
keblat.pars['lcerr'] = 1e-5

keblat.plot_lc(opt_lcpars, prefix, savefig=False)

print blah
if not os.path.isfile(prefix+'lcpars.lmfit2') or clobber_lc:

    # make initial guesses for rsum and f2/f1, assuming main sequence equal mass binary
    rsum = scipy.optimize.fmin_l_bfgs_b(estimate_rsum, 1.0,
                                        args=(period, 2*(keblat.pwidth+keblat.swidth)),
                                        bounds=[(1e-3, 1e3)], approx_grad=True)[0][0]

    # ew = scipy.optimize.fmin(tse_residuals, np.array([1e-3, ecosw]),
    #                          args=(period, tpe, tpe+keblat.sep*period))
    b = flatbottom(keblat.phase[keblat.clip], keblat.flux[keblat.clip], keblat.sep, keblat.swidth)
    # if sdeplist[goodlist][goodlist_ind] < 0.02 and pdeplist[goodlist][goodlist_ind] < 0.04:
    #     b = 1.0
    rrat = guess_rrat(sdeplist[goodlist][goodlist_ind], pdeplist[goodlist][goodlist_ind])
    frat = rrat**(2.5)
    if rsum > 10:
        msum = 2.0
    else:
        msum = rsum
    
    ew_trials = [[esinw, ecosw], [-esinw, ecosw]]
    for jj in np.linspace(0.01, .5, 5):
        ew_trials = ew_trials + [[jj, ecosw], [-jj, ecosw]]
        #[[esinw, ecosw], [-esinw, ecosw], [-0.521, ecosw], [-0.332, ecosw], [-0.142, ecosw], [0.521, ecosw], [0.332, ecosw], [0.142, ecosw], [-.2]]
    lcpars0 = np.array([msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4])
    ew = ew_search_lmfit(ew_trials, keblat, lcpars0, (period, tpe, tpe+keblat.sep*period), fit_ecosw=False, polyorder=1)

    b_trials = [0.01, 0.1, 0.4, 0.8, 1.2]
    rrat_trials = [0.3, 0.7, 0.95]


    b_trials = [b] + [float(jj) for jj in np.array(b_trials)[~np.in1d(b_trials, b)]]
    rrat_trials = [rrat] + [float(jj) for jj in np.array(rrat_trials)[~np.in1d(rrat_trials, rrat)]]
    lc_search_counts=0
    bestlcchi2 = 1e25

    ###################################################################################
    ########################### LC ONLY OPTIMIZATION FIRST ############################
    ###################################################################################
    keblat.updatebounds('period', 'tpe', partol=0.01)
    if pdeplist[goodlist][goodlist_ind]<0.16:
        keblat.parbounds['rrat'] = [1e-6, 1.]
        keblat.parbounds['frat'] = [1e-8, 1.]
    if abs(ecosw) < 0.015:
        keblat.parbounds['ecosw'] = [-0.02, 0.02]
        keblat.parbounds['esinw'] = [-.9, .9]
    else:
        keblat.updatebounds('ecosw', partol=0.1)
        keblat.parbounds['esinw'] = [-np.sqrt(.9**2-ecosw**2), np.sqrt(.9**2-ecosw**2)]

    for i_b, i_rrat, i_ew in list(itertools.product(b_trials, rrat_trials, [ew, [esinw, ecosw]])):
#    for i_b, i_rrat, i_ew in list(itertools.product(b_trials, rrat_trials, [[esinw, ecosw]])):
        # lcpars0 = np.array([rsum, rsum, i_rrat, period, tpe, ew[0], ew[1], i_b, i_rrat**(2.5),
        #                     q1, q2, q3, q4])
        #upper_b = 2.*i_b if i_b==0.01 else 3.0
        #keblat.parbounds['b'][1] = upper_b
        opt_lcpars0 = opt_lc(keblat, msum=msum, rsum=rsum, rrat=i_rrat, period=period, tpe=tpe, esinw=i_ew[0],
                             ecosw=i_ew[1], b=i_b, frat=i_rrat**2.5, q1=q1, q2=q2, q3=q3, q4=q4)

        lcchi2 = np.sum(rez(opt_lcpars0, keblat, polyorder=2)**2)/(np.sum(keblat.clip) - len(opt_lcpars0) - 1)
        if (lcchi2 < bestlcchi2) or (lc_search_counts < 1):
            print "Saving from this run:", lcchi2, bestlcchi2, lc_search_counts
            bestlcchi2 = lcchi2*1.0
            opt_lcpars = opt_lcpars0.copy()
        lc_search_counts+=1

        if (bestlcchi2 <= 1.5) and opt_lcpars[2]<=1.0:
            print "These init b, rrat, esinw, ecosw lcpars are: ", i_b, i_rrat, ew
            break

        # opt_lcpars0 = opt_lc(lcpars0, keblat.jd, keblat.phase, keblat.flux, keblat.fluxerr, keblat.crowd, \
        #                     keblat.clip, set_upperb=upper_b, prefix=prefix)
        # lcchi2 = np.sum(rez(opt_lcpars0, polyorder=2)**2)/np.sum(keblat.clip)
        # if lcchi2 < bestlcchi2:
        #     bestlcchi2 = lcchi2*1.0
        #     opt_lcpars = opt_lcpars0 * 1.0
        #     make_lc_plots(kic, opt_lcpars0, prefix, polyorder=2, suffix='lc_opt')

    try:
        keblat.plot_lc(opt_lcpars, prefix, polyorder=2, suffix='lc_opt2', savefig=True)
    except Exception, e:
        print str(e)


    if bestlcchi2 < lcchi2_threshold:
        print "Saving lmfit lcpars..."
        np.savetxt(prefix+'lcpars.lmfit2', opt_lcpars)
    else:
        print("Bestlcchi2 = {0}, exiting.".format(bestlcchi2))
        np.savetxt(prefix+'lcpars.lmfit2', opt_lcpars)
        sys.exit()
else:
    print "Loading lcpars lmfit"
    opt_lcpars = np.loadtxt(prefix+'lcpars.lmfit2')

bestlcchi2 = np.sum(rez(opt_lcpars[:13], keblat, polyorder=2)**2)/(np.sum(keblat.clip) - len(opt_lcpars) - 1)

msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4 = opt_lcpars[:13]
#opt_lcpars0 = opt_lc(keblat, msum=msum, rsum=rsum, rrat=rrat, period=period, 
#                     tpe=tpe, esinw=esinw, ecosw=ecosw, b=b, frat=frat, q1=q1, 
#                     q2=q2, q3=q3, q4=q4, vary_msum=False, fit_crowd=True)
opt_lcpars0 = opt_lc_crowd(keblat, msum=msum, rsum=rsum, rrat=rrat, period=period, 
                           tpe=tpe, esinw=esinw, ecosw=ecosw, b=b, frat=frat, 
                           q1=q1, q2=q2, q3=q3, q4=q4, vary_msum=False)
crowd_fits = keblat.broadcast_crowd(keblat.quarter, opt_lcpars0[13:])
keblat.crowd = crowd_fits
lcchi2 = np.sum(rez(opt_lcpars0[:13], keblat, polyorder=2)**2)/(np.sum(keblat.clip) - len(opt_lcpars0) - 1)
if (lcchi2 < bestlcchi2):
    print "Saving from this crowding fit run:", lcchi2, bestlcchi2
    bestlcchi2 = lcchi2*1.0
    opt_lcpars = opt_lcpars0[:13].copy()
    crowd = opt_lcpars0[13:]
    np.savetxt(prefix+'lcpars_wcrowd.lmfit2', opt_lcpars0)
else:
    crowd_fits = keblat.broadcast_crowd(keblat.quarter, keblat._crowdsap)
    keblat.crowd = crowd_fits
    crowd = keblat._crowdsap.ravel()

print blah
_, _ = keblat.lcfit(opt_lcpars[:13], keblat.jd, keblat.quarter, keblat.flux, keblat.dflux, keblat.crowd, polyorder=0)

keblat.updatephase(keblat.pars['tpe'], keblat.pars['period'], clip_tol=keblat.clip_tol)

opt_lcpars = np.append(opt_lcpars, np.log(np.median(abs(np.diff(keblat.flux)))))
if np.isinf(k_lnprior(opt_lcpars, keblat)):
    if (np.sqrt(opt_lcpars[5]**2+opt_lcpars[6]**2) > 1.):
        opt_lcpars[5] = ew[0]
        opt_lcpars[6] = ew[1]
        opt_lcpars[3] = period
        opt_lcpars[4] = tpe
    print "Some stuff seems to be out of bounds... somehow... changing them to mean bound values"

####################################################################################
################################### LC ONLY MCMC ###################################
####################################################################################
def k_lnprob2(lcpars):
    return k_lnprob(lcpars, keblat, polyorder=2)

def ilnprob2(isopars, lc_constraints=None):
    return ilnprob(isopars, keblat, lc_constraints=lc_constraints)

def ilnlike(isopars0):
    msum, mrat, z0, age, dist, ebv = isopars0
    isopars = np.array([msum, mrat, z0, age, dist, ebv, 119., 0.03])
    magsmod = keblat.isofit(isopars)
 #/ np.sqrt(self.emagsobs**2 + isoerr**2)
    lc_inputs = np.array([(keblat.r1+keblat.r2)/(isopars[0])**(1./3.), keblat.r2/keblat.r1, keblat.frat])

    lc_priors = (lc_inputs-lc_constraints)/(0.008*lc_constraints)

    if np.any(np.isinf(lc_inputs)):
        return 1e25*np.ones(len(keblat.magsobs)+len(lc_constraints))
        
    if np.any(np.isinf(magsmod)):
        return 1e25*np.ones(len(keblat.magsobs)+len(lc_constraints))

    isores = np.concatenate(((magsmod - keblat.magsobs) / np.sqrt(keblat.emagsobs**2 + isopars[-1]**2),
                             lc_priors)) #/ np.sqrt(self.emagsobs**2 + isoerr**2)

    return isores

ndim = len(opt_lcpars)
nwalkers = 128
niter = 10000
header = prefix+'lc_'#postml_'
footer = str(nwalkers)+'x'+str(niter/1000)+'k'
mcfile = header+footer+'.mcmc'
isonames = ['msum', 'rsum', 'rrat', 'period', 'tpe', 'esinw', 'ecosw', 'b', 'frat', 'q1', 'q2', 'q3', 'q4', 'lcerr']

#opt_lcpars[-1] = np.log(np.median(abs(np.diff(keblat.flux))))
p0_scale = np.ones(ndim)*1e-4
p0_scale[3] = 1e-7
p0_scale[4] = 1e-6
p0_scale[[2, 8]] = 1e-5
p0 = [opt_lcpars + p0_scale * opt_lcpars * np.random.randn(ndim) for ii in range(nwalkers)]
p0 = np.array(p0)
success = False
if os.path.isfile(mcfile):
    params, r1, temp1, logg1, mlpars, success = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter,
                                                        burnin=niter*3/4, plot=True, posteriors=True,
                                        huber_truths=[None]*(len(isonames)+3), isonames=isonames, iso_extras=False)

if not success or not os.path.isfile(mcfile) or clobber_lc:
    if not os.path.isfile(mcfile) or clobber_lc:
        print "MCMC file does not exist, creating..."
        outf = open(mcfile, "w")
        outf.close()
    else:
        if not success:
            print "MCMC file not complete, appending..."
            #niter = niter-mlpars
            outf = open(mcfile, "a")
            outf.close()
    start_time = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, k_lnprob2, threads=4)

    print("Running "+str(niter/1000)+"k MCMC chain")

    for res in sampler.sample(p0, iterations=niter, storechain=False):
        if sampler.iterations % 10 == 0:
            position = res[0]
            outf = open(mcfile, "a")
            for k in range(position.shape[0]):
                blobs = np.array(res[3][k][1:-1].split(","), dtype=float)
                outf.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(sampler.iterations,
                           k, sampler.acceptance_fraction[k], str(res[1][k]),
                            " ".join([str(ii) for ii in position[k]]),
                            str(blobs[0]), str(blobs[1]), str(blobs[2])))
            outf.close()
        if sampler.iterations % 10000 == 0:
            print "Time Elapsed since niter = ", sampler.iterations, time.time()-start_time

    print "Total Time Elapsed for MCMC Run = ", time.time()-start_time

    print "Tot. Acceptance Fraction = ", np.mean(sampler.acceptance_fraction)
    try:
        print "Tot autocorr time = ", np.mean(sampler.acor)
    except:
        print "Could not compute autocorr time..."

    params, r1, temp1, logg1, mlpars, success = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter,
                                                        burnin=niter*3/4, plot=True, posteriors=True,
                                        huber_truths=[None]*(len(isonames)+3), isonames=isonames, iso_extras=False)

###################################################################################
############################# SED ONLY OPTIMIZATION ###############################
###################################################################################

opt_lcpars = mlpars
upper_b = 1.4
if opt_lcpars[-6]*2. > upper_b:
    upper_b = opt_lcpars[-6]*2.

if keblat.magsobs[-1] > 12.:
    dist = 1250.
if keblat.magsobs[-1] > 14.:
    dist = 1500.

lc_constraints = np.array([opt_lcpars[1]/opt_lcpars[0]**(1./3.), opt_lcpars[2], opt_lcpars[8]])
print "LC Constraints (rsum/msum^(1/3), rrat, frat) are: ", lc_constraints

if os.path.isfile(prefix+'isopars.lmfit2') and not clobber_sed:
    print "isopars lmfit file already exists, loading..."
    opt_isopars = np.loadtxt(prefix+'isopars.lmfit2')
    fit_isopars = opt_sed(keblat, msum=opt_isopars[0], mrat=opt_isopars[1],
                          z0=opt_isopars[2], age=opt_isopars[3], dist=opt_isopars[4],
                          ebv=opt_isopars[5], h0=opt_isopars[6], isoerr=opt_isopars[7],
                          ret_lcpars=False)
#    fit_isopars = opt_sed(opt_isopars, keblat, lc_constraints, ebv_dist, ebv_arr, fit_ebv=True, ret_lcpars=False)
else:
    #ebv_trials = [ebv] + list(np.linspace(0.01, 0.1, 4))
#    keblat.parbounds['isoerr'] = [1e-6, 0.1]
    msum_trials = [opt_lcpars[0]] + [1.0, 1.5, 2.0, 2.5, 4.0]
    mrat_trials = [np.clip(opt_lcpars[2]**(1./0.8), 0.1, 0.9)] + [0.3, 0.5, 0.9]
    iso_bestchi2 = 1e25
    iso_counter=0
    for i_msum, i_mrat, i_vary_z0, i_lc in list(itertools.product(msum_trials, 
                                                                  mrat_trials, [False, True], 
                                                                  [lc_constraints])):
        m1 = i_msum/(1.+i_mrat)
#        m2 = i_msum/(1.+1./i_mrat)
        ### HERE ###
        for i_age in get_age_trials(m1):
#            isopars0 = [m1, m2, keblat.z0, i_age, dist, ebv, h0, np.log(0.05)]
#            fit_isopars0 = opt_sed(isopars0, keblat, lc_constraints, ebv_dist, ebv_arr, 
#                                   fit_ebv=True, ret_lcpars=False, vary_z0=i_vary_z0)
            fit_isopars0 = opt_sed(keblat, msum=i_msum, mrat=i_mrat, z0=keblat.z0,
                                   age=i_age, dist=dist, ebv=ebv, h0=h0, 
                                   isoerr=np.log(0.005), ret_lcpars=False, 
                                   vary_z0=i_vary_z0, lc_constraints=i_lc)
            isores = keblat.ilnlike(fit_isopars0, lc_constraints=lc_constraints, residual=True)
            iso_redchi2 = np.sum(isores**2) / len(isores)
            if (iso_counter<1) or (iso_redchi2 < iso_bestchi2):
                fit_isopars = fit_isopars0.copy()
                iso_bestchi2 = iso_redchi2*1.0
            iso_counter+=1
#        if iso_bestchi2 <= 1.:
#            break
    opt_isopars = keblat.ilnlike(fit_isopars, retpars=True)
#    keblat.parbounds['isoerr'] = [-25, -1]  
    
###### using least_squares from scipy to minimize residuals... wonky #######
#iso_bestchi2 = 1e25
#iso_counter=0
#for i_msum, i_mrat in list(itertools.product([opt_lcpars[0], 1.0, 1.5, 2.0], 
#                                             [np.clip(opt_lcpars[2]**(1./0.8), 0.1, 0.9), 0.3, 0.5, 0.9])):
#    m1 = i_msum/(1.+i_mrat)
#    for i_age in get_age_trials(m1):
#        fit_isopars0 = least_squares(ilnlike, np.array([i_msum, i_mrat, keblat.z0, i_age, dist, ebv]))
#        isores = keblat.ilnlike(np.append(fit_isopars0.x, [119., 0.03]), lc_constraints=lc_constraints, residual=True)
#        iso_redchi2 = np.sum(isores**2) / len(isores)
#        if (iso_counter<1) or (iso_redchi2 < iso_bestchi2):
#            fit_isopars = fit_isopars0.copy()
#            iso_bestchi2 = iso_redchi2*1.0
#        iso_counter+=1

    #
    # msum_trials = [opt_lcpars[0]] + [1.0, 1.5, 2.0, 2.5]
    # mrat_trials = opt_lcpars[2] + [0.3, 0.5, 0.9]
    # iso_bestchi2 = 1e25
    # iso_counter=0
    # for i_msum, i_mrat in list(itertools.product(msum_trials, mrat_trials)):
    #     m1 = i_msum/(1.+i_mrat)
    #     m2 = i_msum/(1.+1./i_mrat)
    #     ### HERE ###
    #     for i_age in get_age_trials(m1):
    #         isopars0 = [m1, m2, keblat.z0, i_age, dist, ebv, h0, np.log(0.05)]
    #         fit_isopars0 = minimize(sed_chisq, isopars0, method='nelder-mead', args=(lc_constraints,), options={'xtol':1e-7, 'disp':True})
    #         isores = keblat.ilnlike(fit_isopars0.x, lc_constraints=lc_constraints, residual=True)
    #         iso_redchi2 = np.sum(isores**2) / len(isores)
    #         if (iso_counter<1) or (iso_redchi2 < iso_bestchi2):
    #             fit_isopars = fit_isopars0.copy()
    #             iso_bestchi2 = iso_redchi2*1.0
    #         iso_counter+=1
    #     if iso_bestchi2 <= 1.:
    #         break

    np.savetxt(prefix+'isopars.lmfit2', opt_isopars)

if np.isinf(ilnprob(opt_isopars, keblat, lc_constraints = lc_constraints)[0]):
    print "Somehow isopars are out of bounds? You should double check this, DIana; using initial iso parameters instead"
    opt_isopars = isopars0
###################################################################################
################################## SED ONLY MCMC ##################################
###################################################################################
ndim = len(opt_isopars)
nwalkers = 32
niter = 20000
p0_scale = np.ones(ndim)*1e-4
if (opt_isopars[3] > 10.) or (opt_isopars[0] > 1.):
    p0_scale[3] = 1e-5
p0 = [opt_isopars + p0_scale * opt_isopars * np.random.randn(ndim) for ii in range(nwalkers)]
p0 = np.array(p0)
p0[:,6] = 119.

header = prefix+'sed_'#postml_'
footer = str(nwalkers)+'x'+str(niter/1000)+'k'
mcfile = header+footer+'.mcmc'
isonames = ['m1', 'm2', 'z0', 'age', 'dist', 'ebv', 'h0', 'isoerr']

success = False
if os.path.isfile(mcfile):
    params, r1, temp1, logg1, mlpars, success = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter,
                                                        burnin=niter*3/4, plot=True, posteriors=True,
                                        huber_truths=[None]*(len(isonames)) + list(lc_constraints), isonames=isonames, iso_extras=True)

if not success or not os.path.isfile(mcfile) or clobber_sed:
    if not os.path.isfile(mcfile) or clobber_sed:
        print "MCMC file does not exist, creating..."
        outf = open(mcfile, "w")
        outf.close()
    else:
        if not success:
            print "MCMC file not complete, appending..."
            #niter = niter-mlpars
            outf = open(mcfile, "a")
            outf.close()
    start_time = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ilnprob2, args=(lc_constraints,), threads=4)

    print("Running "+str(niter/1000)+"k MCMC chain")

    for res in sampler.sample(p0, iterations=niter, storechain=False):
        if sampler.iterations % 10 == 0:
            position = res[0]
            outf = open(mcfile, "a")
            for k in range(position.shape[0]):
                blobs = np.array(res[3][k][1:-1].split(","), dtype=float)
                outf.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(sampler.iterations,
                           k, sampler.acceptance_fraction[k], str(res[1][k]),
                            " ".join([str(ii) for ii in position[k]]),
                            str(blobs[0]), str(blobs[1]), str(blobs[2])))
            outf.close()
        if sampler.iterations % 10000 == 0:
            print "Time Elapsed since niter = ", sampler.iterations, time.time()-start_time

    print "Total Time Elapsed for MCMC Run = ", time.time()-start_time

    print "Tot. Acceptance Fraction = ", np.mean(sampler.acceptance_fraction)
    try:
        print "Tot autocorr time = ", np.mean(sampler.acor)
    except:
        print "Could not compute autocorr time..."

    params, r1, temp1, logg1, mlpars, success = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter,
                                                        burnin=niter*3/4, plot=True, posteriors=True,
                                        huber_truths=[None]*(len(isonames)) + list(lc_constraints), isonames=isonames, iso_extras=True)

###################################################################################
#################################### SEDLC OPT ####################################
###################################################################################
##from scipy.optimize import least_squares
##bounds = ((0.1, 0.1, 0.001, 6.0, 10., 0., 118., keblat.period-0.1, keblat.tpe-5., opt_lcpars[5]-0.05, opt_lcpars[6]-0.05,  opt_lcpars[7]-0.1, 0., 0., 0., 0., 0., 0.), (12., 12., 0.06, 10.1, 15000., 1.0, 120., keblat.period+0.1, keblat.tpe+5., opt_lcpars[5]+0.05, opt_lcpars[6]+0.05, opt_lcpars[7]+0.1, 1., 1., 1., 1., 0.0001, 0.02))
#
## bounds = ((0.1, 0.1, 0.001, 6.0, 10., 0., 118., opt_lcpars[3]*0.999, opt_lcpars[4]*0.99, opt_lcpars[5]-0.1*abs(opt_lcpars[5]), opt_lcpars[6]-0.05*abs(opt_lcpars[6]), opt_lcpars[7]*0.99, 0., 0., 0., 0., 0., 0.), (12., 12., 0.06, 10.1, 15000., 2.0, 120., opt_lcpars[3]*1.001, opt_lcpars[4]*1.01, opt_lcpars[5]+0.1*abs(opt_lcpars[5]), opt_lcpars[6]+0.05*abs(opt_lcpars[6]), opt_lcpars[7]*1.01, 1., 1., 1., 1., 0.0002, 0.02))
#
##msum_trials = [opt_lcpars[0], mlpars[0]+mlpars[1]] + [1.0, 1.5, 2.0]
##mrat_trials = [opt_lcpars[2]**(1/0.8), mlpars[1]/mlpars[0]] + [0.3, 0.5, 0.9]
##mass_trials = [[opt_lcpars[0],opt_lcpars[2]**(1/0.8)], [mlpars[0]+mlpars[1], mlpars[1]/mlpars[0]]]
#mass_trials = [[mlpars[0]+mlpars[1], mlpars[1]/mlpars[0]], 
#               [opt_isopars[0]+opt_isopars[1], opt_isopars[1]/opt_isopars[0]]]
mass_trials = [[opt_isopars[0], opt_isopars[1]], [opt_lcpars[0], np.clip(opt_lcpars[2]**(1./0.8), 0.1, 0.95)]]
#mlpars = opt_isopars.copy()
allpars_bestchi2 = 1e25
opt_all_counter=0
if (keblat.z0 <= 0.06) and (keblat.z0 >= 0.001):
    z0 = keblat.z0 * 1.0
else:
    z0 = keblat.zsun


#keblat.updatebounds('period', 'tpe', 'ecosw')

if abs(opt_lcpars[6]) < 0.01:
    keblat.parbounds['ecosw'] = [-0.01, 0.01]

for i_mass, i_age, i_porder, i_b, i_ew, i_za, i_lc in list(itertools.product(mass_trials, [mlpars[3], 9.1], 
                                                                             [2],
                                                                              [True, False], [False, True],
                                                                              [False, True],
                                                                              [lc_constraints])):
    # opt_allpars0 = opt_sedlc(fit_isopars, opt_lcpars, ebv_dist, ebv_arr, keblat.jd, keblat.phase, keblat.flux,
    #                     keblat.fluxerr, keblat.crowd, keblat.clip, mciso=mlpars, fit_ebv=True, set_upperb=upper_b,
    #                         init_porder=i_porder, init_varyb=i_b, init_varyew=i_ew, init_varyza=i_za, lc_constraints=i_lc)
    opt_allpars0 = np.concatenate((np.array(mlpars)[:-1], opt_lcpars[3:8], opt_lcpars[9:13], np.array([1e-5, 0.005])))
    # opt_allpars0 = opt_allpars.copy()
    i_msum, i_mrat = i_mass
#    _m1 = i_msum/(1.+i_mrat)
#    _m2 = i_msum/(1.+1./i_mrat)
    #_age = min(mlpars[3], get_age_trials(_m1)[-1])
#    _age = mlpars[3]
    opt_allpars0[0:2] = [i_msum, i_mrat]
    opt_allpars0[3] = i_age

    _kwargs = dict(zip(parnames_dict['lcsed'], opt_allpars0))
    _kwargs.update({'init_porder': i_porder, 'init_varyb': i_b, 'init_varyew': i_ew, 'init_varyza': i_za, 'lc_constraints': i_lc})

    opt_allpars0 = opt_sedlc(keblat, **_kwargs)
    opt_allpars0 = np.asarray(opt_allpars0)
    allres = lnlike_lmfit(opt_allpars0, keblat, qua=np.unique(keblat.quarter), polyorder=2, residual=True)

    allpars_chi2 = np.sum(allres**2) / len(allres)

    if (opt_all_counter < 1) or (allpars_chi2 < allpars_bestchi2):
        opt_allpars = opt_allpars0*1.
        allpars_bestchi2 = allpars_chi2
    if allpars_bestchi2 < 2:
        break

opt_allpars0 = np.concatenate((np.array(mlpars)[:-1], opt_lcpars[3:8], opt_lcpars[9:13], np.array([1e-5, 0.005])))
_kwargs = dict(zip(parnames_dict['lcsed'], opt_allpars0))
_kwargs.update({'init_varyb': True, 'init_varyew': False, 'freeze_iso': False,
                'lc_constraints':None})
opt_allpars0 = opt_sedlc(keblat, **_kwargs)
opt_allpars0 = np.asarray(opt_allpars0)
allres = lnlike_lmfit(opt_allpars0, keblat, lc_constraints=i_lc, qua=np.unique(keblat.quarter), polyorder=2, residual=True)

allpars_chi2 = np.sum(allres**2) / len(allres)
if (allpars_chi2 < allpars_bestchi2):
    opt_allpars = opt_allpars0*1.
    allpars_bestchi2 = allpars_chi2

print "Writing and making sedlc optimized plots..."
np.savetxt(prefix+'allpars.lmfit2', opt_allpars)
plt.close('all')
if allpars_chi2>=1e10:
    print "The sedlc yielded no good fits. No sedlc_opt plots will be made. "
else:
    keblat.plot_sedlc(opt_allpars, prefix, suffix='sedlc_opt', savefig=True, polyorder=2, ooe=False)



#############
kepQs = np.arange(18)
crowd_vals = np.zeros(len(kepQs))
crowd_vals[np.in1d(kepQs, np.unique(keblat.quarter))] = crowd
print("""{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n""".format(kic, 
                                                   " ".join(str(jj) for jj in opt_allpars),
                                                    keblat.r1, keblat.r2, keblat.frat, 
                                                    keblat.temp1, keblat.temp2, 
                                                    keblat.logg1, keblat.logg2,
                                                    keblat.pars['inc'], 
                                                    " ".join(str(zz) for zz in crowd_vals)))
#########
#finalkics = np.loadtxt('data/finalkics.pars')
finalkics = np.loadtxt('goodEBs_nov2016.list')
final_ind = finalkics[:,0] == kic
opt_allpars = finalkics[final_ind,1:].ravel()[:18]
_opt_allpars = opt_allpars.copy()
opt_allpars[0] = _opt_allpars[1]+_opt_allpars[0]
opt_allpars[1] = _opt_allpars[1]/_opt_allpars[0]
#if opt_allpars[1]>1.:
    #opt_allpars[1] = 0.95

keblat.plot_sedlc(opt_allpars, prefix, savefig=False)
if keblat.temp1<keblat.temp2:
    keblat.parbounds['mrat'] = [0.0085, 1.0]
    if opt_allpars[1]>1.:
        opt_allpars[1] = 0.95
else:
    keblat.parbounds['mrat'] = [0.0085, 1.5]
if abs(keblat.pars['ecosw'])<0.01:
    keblat.parbounds['ecosw'] = [-0.01, 0.01]
elif abs(keblat.pars['ecosw'])<0.03:
    keblat.parbounds['ecosw'] = [-0.1, 0.1]
else:
    keblat.updatebounds('ecosw', partol=0.01)
if abs(keblat.pars['esinw'])<0.01:
    keblat.parbounds['esinw'] = [-0.1, 0.1]
else:
    keblat.updatebounds('esinw', partol=0.01)
keblat.updatebounds('tpe', 'period', partol=0.0001)
opt_allpars0 = opt_allpars.copy()
_kwargs = dict(zip(parnames_dict['lcsed'], opt_allpars0))
_kwargs.update({'init_porder': 2, 'init_varyb': True, 
                'init_varyew': True, 'init_varyza': True, 'lc_constraints': None})
opt_allpars0 = opt_sedlc(keblat, **_kwargs)
opt_allpars0 = np.asarray(opt_allpars0)
allres = lnlike_lmfit(opt_allpars, keblat, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
allres0 = lnlike_lmfit(opt_allpars0, keblat, qua=np.unique(keblat.quarter), polyorder=2, residual=True)

if np.sum(allres0**2) < np.sum(allres**2):
    print("Improvement")
    opt_allpars = opt_allpars0.copy()
    keblat.plot_sedlc(opt_allpars, prefix, savefig=False)

crowd = keblat._crowdsap.ravel()


kepQs = np.arange(18)
crowd_vals = np.zeros(len(kepQs))
crowd_vals[np.in1d(kepQs, np.unique(keblat.quarter))] = crowd
print("""{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n""".format(kic, 
                                                   " ".join(str(jj) for jj in opt_allpars),
                                                    keblat.r1, keblat.r2, keblat.frat, 
                                                    keblat.temp1, keblat.temp2, 
                                                    keblat.logg1, keblat.logg2,
                                                    keblat.pars['inc'], 
                                                    " ".join(str(zz) for zz in crowd_vals)))
#########

lcpars = keblat.getpars('lc')
msum, rsum, rrat, period, tpe, esinw, ecosw, b, frat, q1, q2, q3, q4 = lcpars[:13]
opt_lcpars0 = opt_lc_crowd(keblat, msum=msum, rsum=rsum, rrat=rrat, period=period, 
                           tpe=tpe, esinw=esinw, ecosw=ecosw, b=b, frat=frat, 
                           q1=q1, q2=q2, q3=q3, q4=q4, vary_msum=False, 
                           vary_ew=True, vary_frat=True, vary_b=True, 
                           vary_rrat=True, vary_rsum=True)

crowd_fits = keblat.broadcast_crowd(keblat.quarter, opt_lcpars0[13:])
keblat.crowd = crowd_fits
lcchi2 = np.sum(rez(opt_lcpars0[:13], keblat, polyorder=2)**2)/(np.sum(keblat.clip) - len(opt_lcpars0) - 1)
