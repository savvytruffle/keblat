# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:26:56 2014

@author: eakruse
"""


def dbconnect(host='tddb.astro.washington.edu', user='tddb', password='tddb', db='Kepler', **kwargs):
    """
    Log into a database using MySQLdb.

    Parameters
    ----------
    host : string, optional
        Default tddb.astro.washington.edu
    user : string, optional
        Default eakruse
    password : string, optional
        Default tddb
    db : string, optional
        Default Kepler

    Returns
    -------
    dbconnect : Connect
        MySQLdb connector.
    """
    import MySQLdb
    return MySQLdb.connect(host=host, user=user, passwd=password, db=db)

def loadlc_static(KIC, **kwargs):
    """
    Load Kepler static data from the local tddb database.

    Hacking Ethan's code.

    Parameters
    ----------
    KIC : int
        Kepler Input Catalog number for the target.

    Returns
    -------
    staticinfo: list of floats
        Values of glon, glat, gmag, rmag, imag, zmag, jmag, hmag, kmag, kepmag, Teff, logg, FeH from KIC
    """    
    db = dbconnect(**kwargs)
    cursor = db.cursor()

    exstr = "SELECT glon, glat, gmag, rmag, imag, zmag, jmag, hmag, kmag, kepmag, teff, logg, feh FROM object_static WHERE keplerid = %s"
    cursor.execute(exstr, (int(KIC),))
    staticinfo = cursor.fetchall()
    return staticinfo[0]

def loadlc_pos(KIC, **kwargs):
    db = dbconnect(**kwargs)
    cursor = db.cursor()

    exstr = "SELECT ra_obj, dec_obj FROM object_static WHERE keplerid = %s"
    cursor.execute(exstr, (int(KIC),))
    staticinfo = cursor.fetchall()
    return staticinfo[0]
    
def loadlc_db(KIC, usepdc=True, lc=True, raw=False, **kwargs):
    """
    Load Kepler data from the local tddb database.

    Can pass optional MySQLdb keyword arguments for logging into the database
    (host, user, passwd, db). Those default values should work though.

    Parameters
    ----------
    KIC : int
        Kepler Input Catalog number for the target.
    usepdc : bool, optional
        Default True. If True, use the PDCSAP data instead of the raw SAP.
    lc : bool, optional
        Whether to select long or short cadence. Defaults to True, or LC data.

    Returns
    -------
    time : ndarray
        Kepler times of center of exposure
    flux : ndarray
        Kepler fluxes normalized for each quarter
    fluxerr : ndarray
        Kepler flux errors for each exposure
    cadence : ndarray
        Cadence number
    quarter : ndarray
        Kepler quarter
    quality : ndarray
        Kepler data quality flag
    """
    import numpy as np

    if lc:
        lcflag = "LCFLAG > 0"
    else:
        lcflag = "LCFLAG = 0"

    if usepdc:
        fluxstr = "pdcsap_flux, pdcsap_flux_err "
    else:
        fluxstr = "sap_flux, sap_flux_err "

    db = dbconnect(**kwargs)
    cursor = db.cursor()

    toex = "SELECT cadenceno, quarter, sap_quality, time, {0} FROM source WHERE keplerid = %s AND {1};".format(fluxstr, lcflag)

    cursor.execute(toex, (int(KIC),))
    results = cursor.fetchall()
    
    # ADD IN CROWDSAP STUFF #
    toex2 = "SELECT crowdsap FROM object_quarterly WHERE keplerid = %s AND {0};".format(lcflag)
    cursor.execute(toex2, (int(KIC),))
    crowdsap = cursor.fetchall()
    crowdsap = np.array(crowdsap)
    
    cadence = np.array([x[0] for x in results], dtype=np.int32)
    quarter = np.array([x[1] for x in results], dtype=np.int32)
    quality = np.array([x[2] for x in results], dtype=np.int32)
    time = np.array([x[3] for x in results], dtype=np.float64)
    flux = np.array([x[4] for x in results], dtype=np.float32)
    fluxerr = np.array([x[5] for x in results], dtype=np.float32)

    # guarantee the light curve is in sequential order
    # %timeit says that doing the ordering in python is faster than including
    # an 'ORDER BY time' flag in the mysql search. I have no idea why, but
    # I'll keep doing the ordering here.
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    fluxerr = fluxerr[order]
    quality = quality[order]
    cadence = cadence[order]
    quarter = quarter[order]

    if raw:
        db.close()
        return time, flux, fluxerr, cadence, quarter, quality, crowdsap

    # go from raw CCD counts to normalized fluxes per quarter
    uquarts = np.unique(quarter)
    for ii in uquarts:
        val = np.where(quarter == ii)[0]
        fluxerr[val] /= np.median(flux[val])
        flux[val] /= np.median(flux[val])

    db.close()
    return time, flux, fluxerr, cadence, quarter, quality, crowdsap
