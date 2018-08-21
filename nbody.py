import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ctypes
import os
from numpy.ctypeslib import ndpointer
import platform
import time
import rebound

if platform.system() == "Darwin":
    try:
        lib = ctypes.CDLL('./nbody_mac.so')
    except:
        raise Exception("Can't find .so file; please type ``make`` to compile the code.")
elif platform.system() == "Linux":
    try:
        lib = ctypes.CDLL('./nbody_linux.so')
    except:
        raise Exception("Can't find .so file; please type ``make`` to compile the code.")
else:
    raise Exception("Unknown platform.")


nbody_rk = lib.nbody_rk
nbody_rk.restype = ctypes.c_int
nbody_rk.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                     ndpointer(dtype=ctypes.c_double), ctypes.c_double, ctypes.c_int, 
                    ctypes.c_int, ctypes.c_int, ctypes.c_int]

rsky = lib.rsky
rsky.restype = ctypes.c_int
rsky.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double), ctypes.c_double,
                 ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]

occultquad = lib.occultquad
occultquad.restype = ctypes.c_int
occultquad.argtypes = [ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_double),
                       ndpointer(dtype=ctypes.c_double), ctypes.c_double, ctypes.c_double,
                       ctypes.c_double, ctypes.c_int]

r2au = 0.0046491 #solar radii to AU
d2y = 365.242 #days in a year
TWOPI = 2.*np.pi
re2au = 4.263e-5 #earth radii to AU
hr2d = 0.04167 #1 hr to day
me2ms = 3.003467e-6 # earth mass to solar mass
re2rs = 0.00915 #earth radius to solar radius


def nbody_c(r_i, v_i, mass, dt, Nsteps, Nbody=3, Ndim=3, downsample=1, method='rk'):
    # r_i should be in units of earth radii;
    # v_i in units of earth radii / hr
    r = np.zeros((Nbody, Ndim, Nsteps/downsample))
    v = np.zeros((Nbody, Ndim, Nsteps/downsample))
    r[:,:,0] = r_i
    v[:,:,0] = v_i
    r = r.flatten()
    v = v.flatten()
    if method == 'rk':
        which=0
    elif method == 'vv':
        which=1
    else:
        print("Method must be 'rk' or 'vv'.")
        return
    nbody_rk(r, v, mass, dt, Nbody, Nsteps, downsample, which)
    return r, v


def kep_geta(P, msum):
    return ((P/d2y)**2 * (msum))**(1./3.)


def kep_orb(t, e, t0, P, m1, m2):#, omega, inc):
    maf = np.zeros(len(t))
    rsky(t, maf, e, P, t0, 1e-8, len(t))
    a = kep_geta(P, m1+m2)
    r = a * (1-e**2) / (1. + e*np.cos(maf))
    #v = 2*np.pi/P * a / np.sqrt(1.-e**2) * (1. + 2.*e*np.cos(maf) + e**2)
        #z = r*np.sqrt(1.-np.sin(omega+maf)**2*np.sin(inc)**2)
    return maf, r#, v


def kep2sky(r, maf):
    # obsolete. don't use this.. use kep2cart instead
    x = r*np.cos(maf)
    y = r*np.sin(maf)
    z = x*0.
    return x, y, z


def kep2cart(t, P, m1, m2, t0, e, Omega, inc, omega, nsteps=1, ndim=3):
    t = np.atleast_1d(t)
    assert len(t) == nsteps, "Nsteps == len(t)"
    a = kep_geta(P, m1+m2)
    maf = np.zeros(nsteps)
    rsky(t, maf, e, P, t0, 1e-8, nsteps)
    r = a * (1-e**2) / (1. + e*np.cos(maf))
    R = np.zeros((ndim, nsteps))
    R[0, :] = r * (np.cos(Omega) * np.cos(omega+maf) - np.sin(Omega) * np.sin(omega+maf) * np.cos(inc))
    R[1, :] = r * (np.sin(Omega) * np.cos(omega+maf) + np.cos(Omega) * np.sin(omega+maf) * np.cos(inc))
    R[2, :] = r * np.sin(omega+maf) * np.sin(inc)

    nes = TWOPI/P / np.sqrt(1-e**2) * a
    V = R * 0.
    V[0, :] = -nes * (np.sin(Omega)*np.cos(inc)*(e*np.cos(omega)+np.cos(omega+maf)) +
                      np.cos(Omega)*(e*np.sin(omega)+np.sin(omega+maf)))

    V[1,:] = nes * (np.cos(Omega)*np.cos(inc) * (e*np.cos(omega)+np.cos(omega+maf)) -
                    np.sin(Omega)*(e*np.sin(omega)+np.sin(omega+maf)))
    V[2,:] = nes * np.sin(inc) * (e*np.cos(omega) + np.cos(omega+maf))
    return R, V


def COM(r, v, mass):
    # r = (nbody, ndim, nsteps)
    # mass = (nbody,)
    mass = mass[:, np.newaxis, np.newaxis]
    Rcm = np.sum(mass * r, axis=0) / np.sum(mass)
    Vcm = np.sum(mass * v, axis=0) / np.sum(mass)
    return Rcm[np.newaxis, :, :], Vcm[np.newaxis, :, :]


def helio2bary(r, v, mass):
    Rcm, Vcm = COM(r, v, mass)
    return r-Rcm, v-Vcm


def bary2helio(r, v, mass):
    Rcm, Vcm = COM(r, v, mass)
    return r+Rcm, v+Vcm


def cartb_to_sky_obsolete(r, asc_node, inc, arg_peri, Nbody):
    R = r*0.
    #trans_matrix = np.multiply(np.multiply(Pz(asc_node), Px(inc)), Pz(arg_peri))
    if Nbody == 1:
        trans_matrix = np.multiply(np.multiply(Pz(asc_node), Px(inc)), Pz(arg_peri))
        R = trans_matrix.dot(r)
    else:
        for j in range(Nbody):
            trans_matrix = np.multiply(np.multiply(Pz(asc_node[j]), Px(inc[j])), Pz(arg_peri[j]))
            R[j, :, :] = trans_matrix.dot(r[j, :, :])
    return R


def Px(phi):
    trans_matrix = np.zeros((3, 3))
    trans_matrix[0,0] = 1.0
    trans_matrix[1,1] = np.cos(phi)
    trans_matrix[1,2] = -np.sin(phi)
    trans_matrix[2,1] = np.sin(phi)
    trans_matrix[2,2] = np.cos(phi)
    return trans_matrix


def Pz(phi):
    trans_matrix = np.zeros((3, 3))
    trans_matrix[2,2] = 1.0
    trans_matrix[0,0] = np.cos(phi)
    trans_matrix[0,1] = -np.sin(phi)
    trans_matrix[1,0] = np.sin(phi)
    trans_matrix[1,1] = np.cos(phi)
    return trans_matrix


def kep2cart_delete(pos, asc_node, inc, omega):
    #assert pos.ndim == 3, "pos vector should be of shape (Nbody, Ndim, Nsteps)"
    r, maf = pos
    X = r * (np.cos(asc_node) * np.cos(omega+maf) - np.sin(asc_node) * np.sin(omega+maf) * np.cos(inc))
    Y = r * (np.sin(asc_node) * np.cos(omega+maf) + np.cos(asc_node) * np.sin(omega+maf) * np.cos(inc))
    Z = r * np.sin(omega+maf) * np.sin(inc)
    #sky_pos = cartb_to_sky(pos, asc_node, inc, omega, 1)
    return X, Y, Z


def kep2cart_bary_obsolete(pos, asc_node, inc, omega, m1, m2):
    # units of AU
    #assert pos.ndim == 3, "pos vector should be of shape (Nbody, Ndim, Nsteps)"
    r, maf = pos
    X = np.zeros((2, len(r)))
    Y = X * 0.
    Z = X * 0.

    r2 = r * m1/(m1+m2)
    r1 = r * m2/(m1+m2)
    # 'secondary' component
    X[1,:] = r2 * (np.cos(asc_node) * np.cos(omega+maf) - np.sin(asc_node) * np.sin(omega+maf) * np.cos(inc))
    Y[1,:] = r2 *(np.sin(asc_node) * np.cos(omega+maf) + np.cos(asc_node) * np.sin(omega+maf) * np.cos(inc))
    Z[1,:] = r2 * np.sin(omega+maf) * np.sin(inc)

    #'primary' component
    omega += np.pi
    X[0,:] = r1 * (np.cos(asc_node) * np.cos(omega+maf) - np.sin(asc_node) * np.sin(omega+maf) * np.cos(inc))
    Y[0,:] = r1 * (np.sin(asc_node) * np.cos(omega+maf) + np.cos(asc_node) * np.sin(omega+maf) * np.cos(inc))
    Z[0,:] = r1 * np.sin(omega+maf) * np.sin(inc)

    #sky_pos = cartb_to_sky(pos, asc_node, inc, omega, 1)
    return X, Y, Z


def init_kep_obsolete(t, P, m1, m2, t0, e):
    t = np.atleast_1d(t)
    a = kep_geta(P, m1+m2)
    maf = np.zeros(len(t))
    rsky(t, maf, e, P, t0, 1e-8, len(t))
    r = a * (1-e**2) / (1. + e*np.cos(maf))
    return maf, r


def kep2vel_obsolete(maf, P, e, m1, m2, asc_node, omega, inc):
    # in units of AU / day
    a = kep_geta(P, m1+m2)
    nes = TWOPI/P / np.sqrt(1-e**2) * a
    vx = np.zeros((2, len(maf)))
    vy = vx*0.
    vz = vx*0.

    # 'secondary' component
    vx[1,:] = -m1 / (m1+m2) * nes * (np.sin(asc_node)*np.cos(inc)*(e*np.cos(omega)+np.cos(omega+maf)) + \
        np.cos(asc_node)*(e*np.sin(omega)+np.sin(omega+maf)))
    vy[1,:] = m1 / (m1+m2) * nes * (np.cos(asc_node)*np.cos(inc) * (e*np.cos(omega)+np.cos(omega+maf)) - \
        np.sin(asc_node)*(e*np.sin(omega)+np.sin(omega+maf)))
    vz[1,:] = m1 / (m1+m2) * nes * np.sin(inc) * (e*np.cos(omega) + np.cos(omega+maf))

    # 'primary' component
    omega += np.pi
    vx[0,:] = -m2 / (m1+m2) * nes * (np.sin(asc_node)*np.cos(inc)*(e*np.cos(omega)+np.cos(omega+maf)) + \
        np.cos(asc_node)*(e*np.sin(omega)+np.sin(omega+maf)))
    vy[0,:] = m2 / (m1+m2) * nes * (np.cos(asc_node)*np.cos(inc) * (e*np.cos(omega)+np.cos(omega+maf)) - \
        np.sin(asc_node)*(e*np.sin(omega)+np.sin(omega+maf)))
    vz[0,:] = m2 / (m1+m2) * nes * np.sin(inc) * (e*np.cos(omega) + np.cos(omega+maf))

    return vx, vy, vz

    
def sim_lc_test(orbel, mass, Nbody, Ndim):
    t = np.arange(0, 365*4., 0.00204340278)
    sky_r = np.zeros((Nbody, Ndim, len(t)))
    sky_v = np.zeros((Nbody, Ndim, len(t)))
    orb_elements = np.zeros((Nbody-1, 5, len(t)))
    sim = rebound.Simulation()
    sim.G = 6.674e-11
    sim.units = ('day', 'AU', 'Msun')
    sim.add(m=mass[0])
    for j in range(Ndim-1):
        sim.add(m=mass[j+1], P=orbel[j][0], e=orbel[j][1], inc=orbel[j][2], 
                T=orbel[j][3], Omega=orbel[j][4])
    parti = sim.particles
    sim.move_to_com()
    print("Moved to COM")
    #sim.dt = 1e-6
    sim.integrator = 'ias15'
    for i, time in enumerate(t):
        sim.integrate(time)
        oel = sim.calculate_orbits(heliocentric=True)
        for j in range(Ndim):
            sky_r[j,:,i] = parti[j].x, parti[j].y, parti[j].z
            sky_v[j,:,i] = parti[j].vx, parti[j].vy, parti[j].vz
        for j in range(Ndim-1):
            orb_elements[j,:,i] = oel[j].a, oel[j].e, oel[j].inc, oel[j].omega, oel[j].Omega
    print("dt={0}".format(sim.dt))    
    print("Done integrating for {0} time steps".format(len(t)))    
    z_bin = np.sqrt((sky_r[0,0,:]-sky_r[1,0,:])**2 + (sky_r[0,1,:]-sky_r[1,1,:])**2)
    return sky_r, sky_v, sim, z_bin
    
def sim_lc_rebound(orbel, mass, r1, r2, rp, Nbody, Ndim, frat, u1, u2, u3, u4):
    assert len(mass) == Nbody, "len(mass) must = Nbody"
    t = np.arange(0, 365*4., 0.00204340278)
    sky_r = np.zeros((Nbody, Ndim, len(t)))
    sky_v = np.zeros((Nbody, Ndim, len(t)))
    orb_elements = np.zeros((Nbody-1, 5, len(t)))
    sim = rebound.Simulation()
    sim.G = 6.674e-11
    sim.units = ('day', 'AU', 'Msun')
    sim.add(m=mass[0])
    for j in range(Nbody-1):
        sim.add(m=mass[j+1], P=orbel[j][0], e=orbel[j][1], inc=orbel[j][2], 
                omega=orbel[j][3], T=orbel[j][4], Omega=orbel[j][5])
    parti = sim.particles
    sim.move_to_com()
    print("Moved to COM")
    for i, time in enumerate(t):
        sim.integrate(time)
        oel = sim.calculate_orbits()
        for j in range(Nbody):
            sky_r[j,:,i] = parti[j].x, parti[j].y, parti[j].z
            sky_v[j,:,i] = parti[j].vx, parti[j].vy, parti[j].vz
        for j in range(Nbody-1):
            orb_elements[j,:,i] = oel[j].a, oel[j].e, oel[j].inc, oel[j].omega, oel[j].Omega
    print("Done integrating for {0} time steps".format(len(t)))    
    z_bin = np.sqrt((sky_r[0,0,:]-sky_r[1,0,:])**2 + (sky_r[0,1,:]-sky_r[1,1,:])**2)

    # obj2 transiting obj1; obj1 transiting obj2
    eclipse21 = (sky_r[0,2,:] < sky_r[1,2,:]) & (z_bin < 1.05*(r1+r2)*r2au)
    eclipse12 = (sky_r[0,2,:] > sky_r[1,2,:]) & (z_bin < 1.05*(r1+r2)*r2au)

    if Nbody > 2:
        z_p1 = np.sqrt((sky_r[0,0,:]-sky_r[2,0,:])**2 + (sky_r[0,1,:]-sky_r[2,1,:])**2)
        z_p2 = np.sqrt((sky_r[1,0,:]-sky_r[2,0,:])**2 + (sky_r[1,1,:]-sky_r[2,1,:])**2)
        # obj3 transiting obj1; obj3 transiting obj2
        eclipse31 = (sky_r[0,2,:] < sky_r[2,2,:]) & (z_p1 < 1.05*(r1+rp)*r2au)
        eclipse32 = (sky_r[1,2,:] < sky_r[2,2,:]) & (z_p2 < 1.05*(r2+rp)*r2au)
    else:
        eclipse31 = np.array([False, False])
        eclipse32 = np.array([False, False])
        z_p1, z_p2 = None, None
    dep21,dep12,dep32,dep31=0,0,0,0
    f = np.ones(len(t))
    f2 = np.ones(len(t))
    f3 = np.ones(len(t))
    #plt.figure()
    if eclipse21.sum() > 0:
        #print "eclipse 21", eclipse21.sum()
        _f = f[eclipse21]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_bin[eclipse21]/(r1*r2au), u1, u2, r2/r1, len(_f))
        f[eclipse21] *= (_f + frat) / (1. + frat)
        dep21 = 1-np.nanmin((_f + frat) / (1. + frat))

    if eclipse12.sum() > 0:
        #print "eclipse 12", eclipse12.sum()
        _f = f[eclipse12]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_bin[eclipse12]/(r2*r2au), u3, u4, r1/r2, len(_f))
        f[eclipse12] *= (_f + 1./frat) / (1. + 1./frat)
        dep12 = 1-np.nanmin(_f + 1./frat) / (1. + 1./frat)

    if eclipse31.sum() > 0:
        #print "eclipse 31", eclipse31.sum()
        _f = f2[eclipse31]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_p1[eclipse31]/(r1*r2au), u1, u2, rp/r1, len(_f))
        f2[eclipse31] *= (_f + frat) / (1. + frat)
        dep31 = 1-np.nanmin(_f + frat) / (1. + frat)

        #plt.plot(t[eclipse31], _f, 'r-')
        #plt.plot(t[eclipse31], f2[eclipse31], 'g-')
    if eclipse32.sum() > 0:
        #print "eclipse 32", eclipse32.sum()
        _f = f3[eclipse32]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_p2[eclipse32]/(r2*r2au), u3, u4, rp/r2, len(_f))
        f3[eclipse32] *= (_f + 1./frat) / (1. + 1./frat)
        #plt.plot(t[eclipse32], _f, 'b-')
        #plt.plot(t[eclipse32], f2[eclipse32], 'c-')
        #plt.show()
        dep32 = 1-np.nanmin(_f + 1./frat) / (1. + 1./frat)
    return t, f, f2, f3, sky_r, sky_v, orb_elements, z_bin, z_p1, z_p2, dep21,dep12,dep31,dep32

    
def sim_lc(r_i, v_i, mass, r1, r2, rp, dt, Nbody, Ndim, Nsteps, downsample, frat, u1, u2, u3, u4, method='rk'):
    #r, v in sky/obs plane
    assert len(mass) == Nbody, "len(mass) must = Nbody"
    t = np.arange(Nsteps/downsample)*dt*downsample

    sky_r, sky_v = nbody_c(r_i, v_i, mass, dt, Nsteps, Nbody=Nbody, downsample=downsample, method=method)
    sky_r = sky_r.reshape((Nbody, Ndim, Nsteps/downsample)) * re2au #convert to AU
    sky_v = sky_v.reshape((Nbody, Ndim, Nsteps/downsample)) * re2au / hr2d #convert to AU/d #*1771.7047 # convert to m/s
    
    z_bin = np.sqrt((sky_r[0,0,:]-sky_r[1,0,:])**2 + (sky_r[0,1,:]-sky_r[1,1,:])**2)

    # obj2 transiting obj1; obj1 transiting obj2
    eclipse21 = (sky_r[0,2,:] < sky_r[1,2,:]) & (z_bin < 1.05*(r1+r2)*r2au)
    eclipse12 = (sky_r[0,2,:] > sky_r[1,2,:]) & (z_bin < 1.05*(r1+r2)*r2au)

    if Nbody > 2:
        z_p1 = np.sqrt((sky_r[0,0,:]-sky_r[2,0,:])**2 + (sky_r[0,1,:]-sky_r[2,1,:])**2)
        z_p2 = np.sqrt((sky_r[1,0,:]-sky_r[2,0,:])**2 + (sky_r[1,1,:]-sky_r[2,1,:])**2)
        # obj3 transiting obj1; obj3 transiting obj2
        eclipse31 = (sky_r[0,2,:] < sky_r[2,2,:]) & (z_p1 < 1.05*(r1+rp)*r2au)
        eclipse32 = (sky_r[1,2,:] < sky_r[2,2,:]) & (z_p2 < 1.05*(r2+rp)*r2au)
    else:
        eclipse31 = np.array([False, False])
        eclipse32 = np.array([False, False])
        z_p1, z_p2 = None, None
    dep21,dep12,dep32,dep31=0,0,0,0
    f = np.ones(len(t))
    f2 = np.ones(len(t))
    f3 = np.ones(len(t))
    #plt.figure()
    if eclipse21.sum() > 0:
        #print "eclipse 21", eclipse21.sum()
        _f = f[eclipse21]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_bin[eclipse21]/(r1*r2au), u1, u2, r2/r1, len(_f))
        f[eclipse21] *= (_f + frat) / (1. + frat)
        dep21 = 1-np.nanmin((_f + frat) / (1. + frat))

    if eclipse12.sum() > 0:
        #print "eclipse 12", eclipse12.sum()
        _f = f[eclipse12]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_bin[eclipse12]/(r2*r2au), u3, u4, r1/r2, len(_f))
        f[eclipse12] *= (_f + 1./frat) / (1. + 1./frat)
        dep12 = 1-np.nanmin(_f + 1./frat) / (1. + 1./frat)

    if eclipse31.sum() > 0:
        #print "eclipse 31", eclipse31.sum()
        _f = f2[eclipse31]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_p1[eclipse31]/(r1*r2au), u1, u2, rp/r1, len(_f))
        f2[eclipse31] *= (_f + frat) / (1. + frat)
        dep31 = 1-np.nanmin(_f + frat) / (1. + frat)

        #plt.plot(t[eclipse31], _f, 'r-')
        #plt.plot(t[eclipse31], f2[eclipse31], 'g-')
    if eclipse32.sum() > 0:
        #print "eclipse 32", eclipse32.sum()
        _f = f3[eclipse32]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_p2[eclipse32]/(r2*r2au), u3, u4, rp/r2, len(_f))
        f3[eclipse32] *= (_f + 1./frat) / (1. + 1./frat)
        #plt.plot(t[eclipse32], _f, 'b-')
        #plt.plot(t[eclipse32], f2[eclipse32], 'c-')
        #plt.show()
        dep32 = 1-np.nanmin(_f + 1./frat) / (1. + 1./frat)

    return t, f, f2, f3, sky_r, sky_v, z_bin, z_p1, z_p2, dep21,dep12,dep31,dep32


def cart2kep(pos, vel, m1, m2):
    # pos in AU, vel in AU / d
    X, Y, Z = pos
    vx, vy, vz = vel
    #	X, Y, Z, vx, vy, vz = np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z), np.atleast_1d(vx), \
    #				np.atleast_1d(vy), np.atleast_1d(vz)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    V2 = vx**2 + vy**2 + vz**2
    Rdot = (X*vx + Y*vy + Z*vz) / R

    hhat = np.vstack((Y*vz - Z*vy, Z*vx - X*vz, X*vy - Y*vx))
    h2 = np.sum(hhat**2, axis=0)
    G = 2.959e-4 #AU^3 / d^2 / Msun
    mu = G*(m1+m2)
    a = 1./(2./R - V2/mu)
    e = np.sqrt(1. - h2/(mu*a))
    if np.any(np.isnan(e)):
        print "nan e", e, h2, mu, a
    inc = np.arccos(hhat[-1,:]/np.sqrt(h2))
    asc_node = np.arctan2(hhat[0,:], -hhat[1,:])
    sin_mafo = Z / (R * np.sin(inc))
    cos_mafo = 1./np.cos(asc_node) * (X/R + np.sin(asc_node) * sin_mafo * np.cos(inc))
    sinf = a * (1.-e**2) * Rdot / (np.sqrt(h2) * e)
    cosf = (a * (1-e**2) / R - 1) / e
    omega = np.arctan2(sin_mafo * cosf - cos_mafo * sinf, sin_mafo * sinf + cos_mafo * cosf)
    return a, e, inc, asc_node % TWOPI, omega % TWOPI


def sim_lc_backup(r_i, v_i, mass, dt, Nbody, Ndim, Nsteps, downsample, asc_node, inc, omega):
    t = np.arange(Nsteps/downsample)*dt
    r, v = nbody_c(r_i, v_i, mass, dt, Nsteps, Nbody=Nbody, downsample=downsample)
    f = np.ones(len(t))
    f2 = f*1.0
    r = r.reshape((Nbody, Ndim, Nsteps/downsample)) * 4.2563739e-5 #convert to AU
    v = v.reshape((Nbody, Ndim, Nsteps/downsample)) *1771.7047 # convert to m/s
    sky_r = cartb_to_sky(r, asc_node, inc, omega, Nbody)

    z_bin = np.sqrt((sky_r[0,0,:]-sky_r[1,0,:])**2 + (sky_r[0,1,:]-sky_r[1,1,:])**2) / (1.0 * r2au)
    z_p = np.sqrt((sky_r[0,0,:]-sky_r[2,0,:])**2 + (sky_r[0,1,:]-sky_r[2,1,:])**2) / (1.0 * r2au)

    # obj2 transiting obj1; obj1 transiting obj2
    eclipse21 = (sky_r[0,2,:] < sky_r[1,2,:])
    eclipse12 = (sky_r[0,2,:] > sky_r[1,2,:])

    # obj3 transiting obj1; obj3 transiting obj2
    eclipse31 = (sky_r[0,2,:] < sky_r[2,2,:])
    eclipse32 = (sky_r[1,2,:] < sky_r[2,2,:])

    if eclipse21.sum() > 0:
        print "eclipse 21", eclipse21.sum()
        _f = f[eclipse21]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_bin[eclipse21], 0.0, 1.0, 0.1, len(_f))
        f[eclipse21] = _f
    """
    if eclipse12.sum() > 0:
        print "eclipse 12", eclipse12.sum()
        _f = f[eclipse12]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_bin[eclipse12], 0.0, 1.0, 10., len(_f))
        f[eclipse12] = _f
    """
    if eclipse31.sum() > 0:
        print "eclipse 31", eclipse31.sum()
        _f = f2[eclipse31]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_p[eclipse31], 0.0, 1.0, 0.01, len(_f))
        f2[eclipse31] = _f
    """
    if eclipse32.sum() > 0:
        print "eclipse 32", eclipse32.sum()
        _f = f2[eclipse32]*0.
        _f0 = _f*0.
        occultquad(_f0, _f, z_p[eclipse32], 0.0, 1.0, 0.1, len(_f))
        f2[eclipse32] = _f
    """
    return t, f, f2, r, sky_r


ti = 0.
downsample = 100 #1
dt = 0.0204340278/30. * 24. / downsample #units of hr #kepler short cadence ~ 0.002 d
Nbody = 3 #5
Ndim = 3
Nsteps = int(np.round(4*365.*24/dt, -2))#800000 

"""
e = 0.3
m1 = 1.0
m2 = 0.5
r1, r2, rp = 1.0, 0.5, 0.0995
P = 7.5

t0 = 1.5
Pp = 47.2
mp = 0.0009
asc_node = np.pi
asc_nodep = np.pi
omega = 260 * np.pi/180.
omegap = 42. * np.pi/180.
inc = 89.5 * np.pi/180.
incp = 89.7 * np.pi/180.
t0p = 4.7
ep = 0.01
frat = 0.1

u1, u2 = 0.45, 0.2
u3, u4 = 0.64, 0.099

Ri = np.zeros((Nbody, Ndim, 1))    
Vi = Ri * 0.
#helio centric coordinates, must transform s.t. origin is binary com.
Ri[1,:,:], Vi[1,:,:] = kep2cart(ti, P, m1, m2, t0, e, asc_node, inc, omega)
Ri[2,:,:], Vi[2,:,:] = kep2cart(ti, Pp, m1+m2, mp, t0p, ep, asc_nodep, incp, omegap)

Ri[:2,:,:], Vi[:2,:,:] = helio2bary(Ri[:2,:,:], Vi[:2,:,:], np.array([m1, m2]))


#convert r from au to rE, v from au/d to rE/hr, mass from msun to mE
Ri /= re2au
Vi = Vi / re2au * hr2d
mass = np.array([m1, m2, mp]) / me2ms

#do nbody integration and mandel & agol transit model
t, f, f2, sky_r, sky_v, z_bin, z_p1, z_p2 = sim_lc(Ri.reshape((Nbody,Ndim)), Vi.reshape((Nbody,Ndim)), mass, r1, r2, rp, dt, Nbody, Ndim, Nsteps, downsample, frat, u1, u2, u3, u4)

#convert back to 'helio' cart coords to extract keplerian elements
#sky_rhelio, sky_vhelio = sky_r.copy(), sky_v.copy()
#sky_rhelio[:2,:,:], sky_vhelio[:2,:,:] = bary2helio(sky_r[:2,:,:], sky_v[:2,:,:], np.array([m1, m2]))

#a_t, e_t, inc_t, Omega_t, omega_t = cart2kep(sky_rhelio[1,:,:], sky_vhelio[1,:,:], m1, m2)
#ap_t, ep_t, incp_t, Omegap_t, omegap_t = cart2kep(sky_rhelio[2,:,:], sky_vhelio[2,:,:], m1+m2, mp)
a_t, e_t, inc_t, Omega_t, omega_t = cart2kep(sky_r[1,:,:]-sky_r[0,:,:], sky_v[1,:,:]-sky_v[0,:,:], m1, m2)
ap_t, ep_t, incp_t, Omegap_t, omegap_t = cart2kep(sky_r[2,:,:], sky_v[2,:,:], m1+m2, mp)

fig = plt.figure()
#plt.plot(t/24./365., f, 'b-', alpha=0.4)
#plt.plot(t/24./365., f2, 'r-', alpha=0.4)
as_strided = np.lib.stride_tricks.as_strided
_f = f*f2
tt = as_strided(t, (len(t)+1-30, 30), (t.strides * 2))
ff = as_strided(_f, (len(_f)+1-30, 30), (_f.strides * 2))
tt, ff = np.mean(tt, axis=1), np.mean(ff, axis=1)

plt.plot(tt[::10]/24./365., ff[::10], 'k-x')
plt.xlabel('t (yr)')
plt.ylabel('Norm Flux')

fig, ax = plt.subplots(6, 1, sharex=True)
ax[0].plot(t/24./365., a_t, 'b-')
ax[0].plot(t/24./365., ap_t, 'r-')
ax[0].plot(ti, kep_geta(P, m1+m2), 'b*')
ax[0].plot(ti, kep_geta(Pp, m1+m2+mp), 'r*')
ax[0].set_ylabel('a (AU)')

ax[1].plot(t/24./365., e_t, 'b-')
ax[1].plot(t/24./365., ep_t, 'r-')
ax[1].plot(ti, e, 'b*')
ax[1].plot(ti, ep, 'r*')
ax[1].set_ylabel('e')

ax[2].plot(t/24./365., inc_t, 'b-')
ax[2].plot(t/24./365., incp_t, 'r-')
ax[2].plot(ti, inc, 'b*')
ax[2].plot(ti, incp, 'r*')
ax[2].set_ylabel('inc (rad)')

ax[3].plot(t/24./365., Omega_t, 'b-')
ax[3].plot(t/24./365., Omegap_t, 'r-')
ax[3].plot(ti, asc_node, 'b*')
ax[3].plot(ti, asc_nodep, 'r*')
ax[3].set_ylabel('Omega (rad)')

ax[4].plot(t/24./365., omega_t, 'b-')
ax[4].plot(t/24./365., omegap_t, 'r-')
ax[4].plot(ti, omega, 'b*')
ax[4].plot(ti, omegap, 'r*')
ax[4].set_ylabel('omega (rad)')

ax[5].plot(t/24/365., z_bin, 'b-', label='bin')
ax[5].plot(t/24/365., z_p1, 'r-', label='p-1')
ax[5].plot(t/24/365., z_p2, 'g-', label='p-2')
ax[5].set_ylabel('sep (AU)')
ax[5].set_xlabel('t (yr)')

fig.subplots_adjust(wspace=0, hspace=0)
#plt.legend()
plt.show()

"""
