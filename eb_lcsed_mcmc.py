import matplotlib
#print("Here")
#matplotlib.use('TkAgg')
import sys, itertools, time, os
print(os.uname())
from keblat import *
import emcee
from emcee.utils import MPIPool
from emcee import autocorr as autoc
#from schwimmbad import MPIPool
from helper_funcs import *
from eb_fitting import *

kic = int(sys.argv[1])
nwalkers=int(sys.argv[2])
niter=int(sys.argv[3])
nsets = int(sys.argv[4])
prefix = sys.argv[5]
#threads = int(sys.argv[6])
threads = 4
try:
    lcfile = sys.argv[6]
except:
    lcfile = None

#prefix = os.path.join(prefix, '{}/'.format(kic))

keblat = Keblat(preload=False)
keblat.loadiso2(isoname='isodata_jun4.dat')
keblat.loadsed(sedfile='data/kepsedall_0216.dat')
keblat.loadvkeb(filename='data/kebproperties_0216.dat')

goodv = keblat.kiclookup(kic, target=keblat.vkeb[:, 0])
keblat.morph = keblat.vkeb[goodv, 8]
keblat.loadlc(kic, keblat.vkeb[goodv, [1, 2, 5, 6, 7]], user_specified=lcfile, clip_tol=2.0)
magsobs, emagsobs, extinction, glat, z0 = keblat.getmags(kic)
ebv = extinction[0]
excludelist = get_excludelist(fname='data/sed_flag_file_0328')
if np.isnan(z0):
    z0 = keblat.zsun
try:
    exclusive = excludelist[kic]
except:
    exclusive = []
keblat.isoprep(magsobs, emagsobs, extinction, glat, z0, exclude=exclusive)#exclude=[]) #excludelist[goodlist_ind])#'gmag','rmag','imag','zmag'])


# load lmfit parameters
#goodEBs = np.loadtxt('data/finalkics2_new.pars')
#mcmcEBs = np.loadtxt('data/mcmc_results_all_may30.mlpars')
#goodEBs = np.loadtxt('data/all_fullsuite.pars')
goodEBs = np.loadtxt('data/fullsuite_feb2019_final.pars')

_ind = (goodEBs[:,0].astype(int) == kic)
#crowd = goodEBs[_ind, 27:].ravel()
crowd = goodEBs[_ind, 30:].ravel()

crowd = crowd[crowd>0]
keblat.crowd = keblat.broadcast_crowd(keblat.quarter, crowd)
#_ind2 = (mcmcEBs[:,0].astype(int) == kic)
#try:
#    allpars = mcmcEBs[_ind2, 1:19].ravel()
#except:
#    allpars = goodEBs[_ind,1:19].ravel()
#allpars = goodEBs[_ind,1:19].ravel()
allpars = goodEBs[_ind,2:20].ravel()

if allpars[-1] == 0:
    allpars[-1] = 1e-6
#residuals = keblat.lnlike(allpars, qua=np.unique(keblat.quarter), polyorder=2, residual=True)
try:
    keblat.plot_sedlc(allpars, '', suffix='', savefig=False)
except:
    keblat.loadiso2()
    keblat.isoprep(magsobs, emagsobs, extinction, glat, z0, exclude=exclusive)#exclude=[]) #excludelist[goodlist_ind])#'gmag','rmag','imag','zmag'])
    keblat.plot_sedlc(allpars, '', suffix='', savefig=False)

if (max(keblat.pwidth, keblat.swidth) > 0.04) and (keblat.pwidth+keblat.swidth > 0.091):
    clip_tol = 1.5
#elif ((keblat.pw3idth > 0.01) and (keblat.swidth > 0.01)) or (keblat.pwidth+keblat.swidth>:
#    clip_tol = 1.5
else:
    clip_tol = 1.7
print("Clip tolerance = {0}".format(clip_tol))

keblat.updatephase(keblat.pars['tpe'], keblat.pars['period'], clip_tol=clip_tol)

check_dir_exists(prefix)

keblat.start_errf(prefix+'lcfit.err')

def lnprior_mc(allpars, crowd_fit=False):       
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

def lnlike_mc(allpars, polyorder=2, crowd_fit=None):
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
    

def lnprob_mc(allpars, lc_constraints=None, crowd_fit=False):
    lp = lnprior_mc(allpars, crowd_fit=crowd_fit)
    if np.isinf(lp):
        return -np.inf, str([-np.inf]*(3+len(np.unique(keblat.quarter[keblat.clip]))))
#    allpars[-2:] = np.exp(allpars[-2:])
#    ll = keblat.lnlike(allpars[:18], lc_constraints=lc_constraints, 
#                       qua=np.unique(keblat.quarter),
#                       crowd_fit=allpars[18:] if crowd_fit else None)
    ll, crowd = lnlike_mc(allpars[:18])
    if (np.isnan(ll) or np.isinf(ll)):
        return -np.inf, str([-np.inf]*(3+len(crowd)))
    return lp + ll, str([keblat.r1, keblat.r2, keblat.frat]+crowd)

def lnprob(allpars0, gaussprior=False, ecosw_prior=None, gd_prior=False, z_prior=False):
    allpars = allpars0[:18].copy()
    sigma_ebv = allpars0[18]
    #print allpars[:18]
    lp = keblat.lnprior(allpars, gaussprior=gaussprior)

    if np.isinf(lp):
        return -np.inf, str((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf))
    if z_prior:
        lp = lp - 0.5*(((allpars[2] - keblat.z0)/(0.2 * np.log(10) * keblat.z0))**2)

    #prior check on sigma_ebv, in ln space
    if sigma_ebv>=keblat.parbounds['sigma_ebv'][0] and sigma_ebv<=keblat.parbounds['sigma_ebv'][1]:
        sigma_ebv = np.exp(sigma_ebv) #sampling in ln space, so have to exponentiate
        lp = lp - 0.5*(((allpars[5]-keblat.ebv)/sigma_ebv)**2 + 2.*np.log(sigma_ebv))
    else:
        return -np.inf, str((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf))

    if ecosw_prior is not None:
        lp = lp - 0.5*(((allpars[9]-ecosw_prior)/(0.2*ecosw_prior))**2 + 2.*np.log(abs(0.2*ecosw_prior)))
    if gd_prior:
#        sigma_d = allpars0[19]
        sigma_d = allpars0[19]
        if sigma_d>=keblat.parbounds['sigma_d'][0] and sigma_d<=keblat.parbounds['sigma_d'][1]:
            sigma_d = np.exp(sigma_d)
            lp = lp - 0.5*(((allpars[4]-keblat.dist)/sigma_d)**2 + 2.*np.log(sigma_d))
        else:
            return -np.inf, str((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf))
    allpars[-2:] = np.exp(allpars[-2:])
    ll = keblat.lnlike(allpars, qua=np.unique(keblat.quarter))
    if (np.isnan(ll) or np.isinf(ll)):
        return -np.inf, str((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf))
    return lp + ll, str((keblat.r1, keblat.r2, keblat.frat, keblat.mact1, keblat.mact2))
#    return keblat.lnprob(allpars, qua=np.unique(keblat.quarter), gaussprior=gaussprior)
    
#def get_ac_time(chains, ndim, c=1):
#    """chains: (niter, nwalkers, ndim) ndarray"""
#    tau = np.zeros(ndim)
#    print(chains.shape)
#    for ii in range(ndim):
#        try:
#            tau[ii] = autoc.integrated_time(np.nanmean(chains[:,:,ii].T, axis=0), axis=0, low=1, high=None,
#                              step=1, c=c, fast=False)
#        except:
#            print("Autocorrelation time couldn't be computed for {}".format(parnames_dict['lcsed'][ii]))
#    return tau

######## load in gaia prior distance ###########
gaia_distance = np.loadtxt('data/ebs_get_gaia_distance.list')
gaia_prior = gaia_distance[np.where(gaia_distance[:,0] == kic)[0], 1:].ravel()
if np.isnan(gaia_prior[0]):
    gd_prior=False
    keblat.debv=0.01
    keblat.dist_err=0.01
else:
    keblat.gaia_prior(gaia_prior[0], gaia_prior[1], gaia_prior[2])
    gd_prior=True
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)


###################################################################################
################################# LC + SED MCMC ###################################
###################################################################################
#_, _uqrt = np.unique(keblat.quarter, return_index=True)
#crowd_pars = keblat.crowd[_uqrt]
#allpars = np.append(allpars, crowd_pars)

allpars = np.append(allpars, max(np.log(keblat.debv), np.log(abs(keblat.ebv-allpars[5]))))
allpars = np.append(allpars, np.log(keblat.dist_err))
    
ndim = len(allpars)

#nwalkers = 128
#nsets=1
#niter = 20000
header = prefix+'lcsed_mini_flatz_fitsig_gaia{}_'.format(int(gd_prior))#postml_'
#footer = str(nwalkers)+'x'+str(niter*nsets/1000)+'k'
footer = str(nwalkers)+'x'+str(niter/1000)+'k'
mcfile = header+footer+'.mcmc2' ## 3/29/18 change back to .mcmc when done diagnosing old chains


###### 3/29/18 edits here ##########
#if os.path.isfile(mcfile):
#    mcresults = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter, 
#                    burnin=int(3*niter/4.), plot=False, posteriors=False,
#                    isonames=parnames_dict['lcsed'], blob_names = ['r1', 'r2', 'frat'])
#
#    print("Median tau * 50 < Niter/10: {}".format(np.percentile(mcresults[-3], 86)*50<niter/10))
#    if np.percentile(mcresults[-3], 86)*50<niter/10 and mcresults[-2] >= 0.01:
#        outf = open('/astro/store/gradscratch/tmp/windemut/take2/converged_mcmc.list', "a")
#        outf.write("{} {} {} {}\n".format(kic, mcresults[-2], np.percentile(mcresults[-3], 86),
#                   " ".join([str(zz) for zz in mcresults[-3]])))
#        outf.close()
#    else:
#        outf = open('/astro/store/gradscratch/tmp/windemut/take2/notconverged_mcmc.list', "a")
#        outf.write("{} {} {} {}\n".format(kic, mcresults[-2], np.percentile(mcresults[-3], 86),
#                   " ".join([str(zz) for zz in mcresults[-3]])))
#        outf.close()
#else:
#    print("{} does not exist".format(mcfile))

#print(blah)

#isonames = parnames_dict['lcsed']+['cr'+str(crq) for crq in np.unique(keblat.quarter)]
#mcresults = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter, burnin=5000, isonames=isonames)
#print blah
if allpars[16] > 0:
    allpars[16:18] = np.log(allpars[16:18])
if allpars[16]<-20:
    allpars[16] = np.log(np.nanmedian(abs(np.diff(keblat.flux))))
    allpars[17] = np.log(np.nanmedian(keblat.emagsobs))
#if abs(keblat.ebv-allpars[5])>keblat.debv:
#    keblat.debv = abs(keblat.ebv-allpars[5])*1.4
if kic == 9837578:
    z_prior=False#True
    allpars = np.loadtxt('/astro/store/gradscratch/tmp/windemut/eb_fitting_mini_hyak/eb_fitting_mini/gscratch/stf/windemut/eb/eb/9837578/sedlc_bf_highz.lmfit')
    allpars[16:18] = [-6.9, -2.0]
    allpars = np.append(allpars, max(np.log(keblat.debv), np.log(abs(keblat.ebv-allpars[5]))))
    allpars = np.append(allpars, max(np.log(keblat.dist_err), np.log(abs(keblat.dist-allpars[4]))))
        
else:
    z_prior=False
p0 = make_p0_ball(allpars, ndim, nwalkers)
p0[:,6] = 119.
if not gd_prior:
    p0[:,19] = np.log(keblat.dist_err)
    ndim-=1 #this should be commented out after making plots of last condor batch
#p0[:,18] = np.log(keblat.debv)
#p0[:,-2] = allpars[-2] + 1e-7*allpars[-2]*np.random.randn(nwalkers)
#p0[:,-1] = allpars[-1] + 1e-6*allpars[-1]*np.random.randn(nwalkers)

parname_key = 'lcsed_sigmas' #'lcsed'
for ii in range(ndim):
    p0[:,ii] = np.clip(p0[:,ii], keblat.parbounds[parnames_dict[parname_key][ii]][0], 
      keblat.parbounds[parnames_dict[parname_key][ii]][1])

#mcresults = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter*nsets, 
#            burnin=None, plot=True, posteriors=True, acf=True, 
#            isonames=parnames_dict['lcsed_sigmas'][:ndim], 
#            blob_names = ['r1', 'r2', 'frat', 'mact1', 'mact2'],
#            write_mc=True)
#
print(blah)
clobber=False
if not os.path.isfile(mcfile) or clobber:
    print("MCMC file {} does not exist, creating...".format(mcfile))

    outf = open(mcfile, "w")
    outf.close()
    
    outf2 = open(mcfile+'.acor', "w")
    outf2.close()
    nsets0 = 0

else:
    total_progress = wc(mcfile)
    total_progress = total_progress/nwalkers * 10
    if total_progress*0.9 >= niter*nsets:
        print("MCMC file {} already exists with {} iter, making plots for KIC {}".format(mcfile, total_progress, kic))
        mcresults = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter*nsets, 
                    burnin=None, plot=True, posteriors=True,
                    isonames=parnames_dict['lcsed_sigmas'][:ndim], blob_names = ['r1', 'r2', 'frat', 'mact1', 'mact2'],
                    write_mc=True)
        sys.exit()
    else:
        p0 = tail(mcfile, nwalkers)
        nsets0 = int(np.round(p0[0,0]/niter)) - 1
        print("MCMC file {} already exists with {} iter, appending for KIC {}".format(mcfile, total_progress, kic))
        p0 = p0[:, 4:4+ndim]
#        p0[:, 18] = np.log(keblat.debv)

if (keblat.vkeb[goodv,3]<0.19):
#    keblat.updatebounds('tpe', 'period', partol=0.001)
    if np.sqrt(np.sum(allpars[9:11]**2))<0.3:
        keblat.parbounds['e'] = [0., 0.9]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads, 
                                kwargs={'gaussprior': (keblat.vkeb[goodv,3]<0.19),
                                        'gd_prior':gd_prior, 'z_prior':z_prior})
print("Running "+str(niter*nsets/1000)+"k MCMC chain")
#for zz in range(0, nsets):
#    start_time = time.time()
#
#    print("Appending to MCMC file... Set {} w/niter={}".format(zz, niter))
#    sampler.run_mcmc(p0, niter)
#    print(time.time()-start_time)
#
#    blobs = np.asarray(sampler.blobs)
#    outf = open(mcfile, "a")
#    for ii,jj in itertools.product(range(0, niter, 10), range(nwalkers)):
#        outf.write("{0} {1} {2} {3} {4} {5}\n".format(ii, jj, sampler.acceptance_fraction[jj],
#                   sampler.lnprobability[jj, ii], 
#                   " ".join([str(kk) for kk in sampler.chain[jj, ii, :]]),
#                   " ".join([str(blb) for blb in blobs[ii, jj].strip("(").strip(")").replace(" ", "").split(",")]), ))
#    outf.close()
#    outf2 = open(mcfile+'.acor', "a")
#    try:
#        outf2.write("{}\n".format(" ".join([str(sac) for sac in sampler.acor])))
#        print("Mean autocorr={}".format(np.mean(sampler.acor)))
#    except:
#        print("Could not compute autocorr")
#    outf2.close()
#    print("Gelman-Rubin Statistic = {}".format(" ".join([str(gelman_rubin(sampler.chain[:,:,jj])) for jj in range(ndim)])))
#    p0 = sampler.chain[:,-1,:].copy()
#    sampler.reset()
#



#print(blah)
for zz in range(nsets0, nsets):
    start_time = time.time()

    print("Running "+str(niter/1000)+"k MCMC chain")
    for res in sampler.sample(p0, iterations=niter, storechain=False):
        if sampler.iterations % 10 == 0:
            position = res[0]
            outf = open(mcfile, "a")
            for k in range(position.shape[0]):
                blobs = np.array(res[3][k][1:-1].split(","), dtype=float)
                outf.write("{0} {1} {2} {3} {4} {5}\n".format(sampler.iterations,
                           k, sampler.acceptance_fraction[k], res[1][k],
                            " ".join([str(ii) for ii in position[k]]),
                            " ".join([str(kk) for kk in blobs])))

#        outf.write("{0} {1} {2} {3} {4} {5}\n".format(ii, jj, sampler.acceptance_fraction[jj],
#                   sampler.lnprobability[jj, ii], 
#                   " ".join([str(kk) for kk in sampler.chain[jj, ii, :]]),
#                   " ".join([str(blb) for blb in blobs[ii, jj].strip("(").strip(")").replace(" ", "").split(",")]), ))

            outf.close()
        if sampler.iterations % 10000 == 0:
            print "Time Elapsed since niter = ", sampler.iterations, time.time()-start_time
    print "Total Time Elapsed for MCMC Run = ", time.time()-start_time

    print "Tot. Acceptance Fraction = ", np.mean(sampler.acceptance_fraction)
#    try:
#        print "Tot autocorr time = ", np.mean(sampler.acor)
#    except:
#        print "Could not compute autocorr time..."
    p0 = position.copy()
    np.savez(header+footer+'.'+str(zz)+'.mcstate', res=res)
#    bad_walkers,params,blobs,mlpars, tau, afrac, logli = test_convergence(mcfile, keblat, header, footer, nwalkers, ndim, niter, 
#                                 burnin=None, isonames=parnames_dict['lcsed_sigmas'][:ndim],
#                                 blob_names=['r1', 'r2', 'frat', 'mact1', 'mact2'])
#
#    if (zz+1)*niter / 10. / np.nanmax(tau) >= 50:
#        print("Effective chain size > 50, stopping MCMC!")        
#        break
    if np.nanmedian(sampler.acceptance_fraction)<0.1 and sampler.a>1.:
        sampler.a*=0.8
        print("Reducing scale parameter to {:.2f} for better acceptance fraction".format(sampler.a))


#print "Total Time Elapsed for MCMC Run = ", time.time()-start_time
#isonames = parnames_dict['lcsed']+['cr'+str(crq) for crq in np.unique(keblat.quarter)]
mcresults = plot_mc(mcfile, keblat, header, footer, nwalkers, ndim, niter, 
                    burnin=None, plot=True, posteriors=True,
                    isonames=parnames_dict['lcsed_sigmas'][:ndim], blob_names = ['r1', 'r2', 'frat', 'mact1', 'mact2'],
                    write_mc=True)

#print("Final effective chain size = {:.1f}".format((zz+1)*niter / 10. / np.nanmax(tau)))
#print("Median tau * 50 < Niter/10: {}".format(np.percentile(mcresults[-3], 86)*50<nsets*niter/10))

################################### append to mcmc_converged.mlpars ######################
#kiclist, perlist, pdeplist, sdeplist, morphlist = np.loadtxt('data/kebproperties_0216.dat', 
#                                                             usecols=(0,1,3,4,8), delimiter=';', unpack=True)
#_, _unique, _cnts = np.unique(kiclist.astype(int), return_index=True, return_counts=True)
#kiclist = kiclist[_unique]
#perlist = perlist[_unique]
#morphlist = morphlist[_unique]
#pdeplist = pdeplist[_unique]
#sdeplist = sdeplist[_unique]
#ebs = np.loadtxt('data/finalkics2_new.pars')
#masterfilename = 'data/converged_mcmc_all_may30.mlpars'
#mcmcebs = np.loadtxt('data/converged_mcmc_all_may30.mlpars')
#
#for kic in np.unique(ebs[:,0][~np.in1d(ebs[:,0], mcmcebs[:,0])].astype(int)): 
#    _ind = np.where(kiclist == kic)[0]
#    morph_par = morphlist[_ind][0]
#    ii = np.where(ebs[:,0] == kic)[0]
#    try:
#        mlpars = np.loadtxt('/astro/store/gradscratch/tmp/windemut/take2/'+str(kic)+'/lcsed_128x110k.mcpars')
#        goodv = keblat.kiclookup(kic, target=keblat.vkeb[:, 0])
#        keblat.loadlc(kic, keblat.vkeb[goodv, [1, 2, 5, 6, 7]])#, clip_tol=2.0)
#        magsobs, emagsobs, extinction, glat, z0 = keblat.getmags(kic)
#        ebv = extinction[0]
#        if np.isnan(z0):
#            z0 = keblat.zsun
#        try:
#            exclusive = excludelist[kic]
#        except:
#            exclusive = []
#        keblat.isoprep(magsobs, emagsobs, extinction, glat, z0, exclude=exclusive)#exclude=[]) #excludelist[goodlist_ind])#'gmag','rmag','imag','zmag'])
#        crowd = ebs[ii, 27:].ravel()
#        keblat.crowd = keblat.broadcast_crowd(keblat.quarter, crowd[crowd>0])
#        _ = keblat.lnlike(mlpars, lc_constraints=None, qua=np.unique(keblat.quarter), residual=False)
#        kepQs = np.arange(18)
#        crowd_vals = np.zeros(len(kepQs))
#        crowd_vals[np.in1d(kepQs, np.unique(keblat.quarter))] = crowd
#        master_file = open(masterfilename, "a")
#        master_file.write("""{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n""".format(kic, morph_par,
#                                                       " ".join([str(val) for val in mlpars]),
#                                                        keblat.r1, keblat.r2, keblat.frat, 
#                                                        keblat.temp1, keblat.temp2, 
#                                                        keblat.logg1, keblat.logg2,
#                                                        keblat.pars['inc'], 
#                                                        " ".join(str(zz) for zz in crowd_vals)))
#        master_file.close()
#    except:
#        print(kic)

###########################################################################
######### oct 22 2018 testing fix one parameter +/- 3sigma ################
######### and optimizing all others to map out chi2 spare  ################
###########################################################################
#pars50p = np.array([1.59367845009,0.245859087338,0.00408129544015,9.42673406813])
#pars1sigp = np.array([+0.00672140091,+0.000952001969,+0.00028107953356,+0.011056955385])
#pars1sigm = np.array([-0.00800414437,-0.00073904138556,-0.000625346318389,-0.0063727641638])
##parsbf = np.array([1.60202813424,0.24822009215,0.00290081473676,9.42061948436])
#pars1sig = np.nanmax(abs(np.vstack((pars1sigp, pars1sigm))), axis=0)
#
#lc_constraints=None
#qua=np.unique(keblat.quarter)
#polyorder=2
#residual=True
#
#mlpars_orig = mlpars.copy()
#_kwargs = dict(zip(parnames_dict['lcsed'], mlpars))
#_kwargs.update({'init_porder':2, 'niter_max':3})
#mlpars = opt_sedlc_fix1par(keblat, **_kwargs)
#
#
#plt.figure(figsize=(8,8))
#pars2check = ['msum', 'mrat', 'z0', 'age']
#
#res = keblat.lnlike(mlpars, lc_constraints=lc_constraints, qua=qua, polyorder=polyorder, residual=residual)
#chi2_orig = np.sum(res**2) 
##par2fix='age'
#Ntrials = 10
#test_pars_all = np.zeros((4, Ntrials+1, len(mlpars)))
#chi2_vals = np.zeros((4, Ntrials+1))
#test_pars_blobs = np.zeros((4, Ntrials+1, 5))
#for jj in range(len(pars2check)):
##    plt.subplot(2,2,jj+1)
##    trial_vals = np.sort(np.append(parsbf[jj], pars50p[jj]+np.linspace(pars1sigm[jj]*3, pars1sigp[jj]*3, 30)))
#    trial_vals = np.sort(mlpars[jj] + np.append(0,np.linspace(pars1sig[jj]*-1, pars1sig[jj]*1, Ntrials)))
#
#    par2fix=pars2check[jj]
#    test_pars = mlpars.copy()
#    start_i = np.where(trial_vals == mlpars[jj])[0][0]
#    
#    for ii in range(start_i+1):
#        test_pars[dict(zip(parnames_dict['lcsed'], np.arange(len(parnames_dict['lcsed']))))[par2fix]] = trial_vals[start_i-ii]
#        _kwargs = dict(zip(parnames_dict['lcsed'], test_pars))
#        _kwargs.update({'init_porder':2, 'niter_max':5, 'par2fix':par2fix})
#        test_pars = opt_sedlc_fix1par(keblat, **_kwargs)
#        res = keblat.lnlike(test_pars, lc_constraints=lc_constraints, qua=qua, polyorder=polyorder, residual=residual)
#        chi2_vals[jj, start_i-ii] = np.sum(res**2) 
#        test_pars_blobs[jj,start_i-ii,:] = keblat.mact1, keblat.mact2, keblat.r1, keblat.r2, keblat.frat
#        test_pars_all[jj,start_i-ii,:] = test_pars.copy()
#
#    test_pars = mlpars.copy()
#
#    for ii in range(len(trial_vals)-start_i):
#        test_pars[dict(zip(parnames_dict['lcsed'], np.arange(len(parnames_dict['lcsed']))))[par2fix]] = trial_vals[start_i+ii]
#        _kwargs = dict(zip(parnames_dict['lcsed'], test_pars))
#        _kwargs.update({'init_porder':2, 'niter_max':3, 'par2fix':par2fix})
#        test_pars = opt_sedlc_fix1par(keblat, **_kwargs)
#        res = keblat.lnlike(test_pars, lc_constraints=lc_constraints, qua=qua, polyorder=polyorder, residual=residual)
#        chi2_vals[jj, start_i+ii] = np.sum(res**2) 
#        
#        test_pars_blobs[jj,start_i+ii,:] = keblat.mact1, keblat.mact2, keblat.r1, keblat.r2, keblat.frat
#        test_pars_all[jj,start_i+ii,:] = test_pars.copy()
#
##    plt.plot(trial_vals, chi2_vals[jj,:]-chi2_orig, 'o')
###    plt.plot(trial_vals[-1], chi2_vals[-1]-chi2_orig, 'r*')
##    plt.xlabel(par2fix)
###    plt.ylim((0, 15))
##    plt.ylabel('$\Delta\chi^2$')
#
#min_ind = np.where(chi2_vals == np.nanmin(chi2_vals))
#
#plt.figure()
#for jj in range(len(pars2check)):
#    plt.subplot(2,2,jj+1)
#    for kk in range(len(pars2check)):
#        plt.plot(test_pars_all[kk,:,jj], chi2_vals[kk,:] - chi2_orig, 'o', label='fixed {}'.format(pars2check[kk]))
#    plt.plot(test_pars_all[:,:,jj][min_ind], chi2_vals[min_ind]-chi2_orig, 'k*')
#    plt.axvline(mlpars[jj], color='k', linestyle='--', lw=2, alpha=0.5)
#    plt.xlabel('{}'.format(pars2check[jj]))
#    plt.ylabel('$\Delta \chi^2$')
#plt.legend()
#plt.show()
#
#mlpars = test_pars_all[min_ind[0], min_ind[1], :].ravel().copy()
