"""
Plotting utilities for single channel currents.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import math
import numpy as np
from pylab import figure, semilogx, savefig

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs import scburst
from scalcs import popen
from scalcs import pdfs
from scalcs import cjumps

def Popen(mec, tres):
    """
    Calculate Popen curve parameters and data for Popen curve plot.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM.
    pe : ndarray of floats, shape (num of points,)
        Open probability corrected for missed events.
    pi : ndarray of floats, shape (num of points,)
        Ideal open probability.
    """

    iEC50 = popen.EC50(mec, 0)
    eEC50 = popen.EC50(mec, tres)
    pmax, cx = popen.maxPopen(mec, 0)
    nH = popen.nH(mec, 0)

    # Plot ideal and corrected Popen curves.
    cmin = iEC50 / 20
    cmax = iEC50 * 500
    log_start = int(np.log10(cmin)) - 1
    log_end = int(np.log10(cmax)) - 1
    points = 512

    c = np.logspace(log_start, log_end, points)
    pe = np.zeros(points)
    pi = np.zeros(points)
    H = np.zeros(points)
    for i in range(points):
        pe[i] = popen.Popen(mec, tres, c[i])
        pi[i] = popen.Popen(mec, 0, c[i])
        H[i] = pmax / (math.pow((iEC50 / c[i]), nH) + 1) # Hill equation

    c = c * 1000000 # x axis in microM

    return c, pe, pi#,  H

def burst_length_pdf(mec, multicomp=False, conditional=False,
    tmin=0.00001, tmax=1000, points=512):
    """
    Calculate the mean burst length and data for burst length distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conditional : bool
        True if conditional distribution is plotted.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points)
        Burst length pdf.
    cfbrst : ndarray of floats, shape (num of open states, num of points)
        Conditional burst length pdf.
    """

    eigs, w = scburst.length_pdf_components(mec)
    tmax = 20 / min(eigs)
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)
    fbst = t * pdfs.expPDF(t, 1 / eigs, w / eigs)

    if multicomp:
        mfbst = np.zeros((mec.kE, points))
        for i in range(mec.kE):
             mfbst[i] = t * pdfs.expPDF(t, 1 / eigs[i], w[i] / eigs[i])
        return t * 1000, fbst, mfbst

    if conditional:
        cfbst = np.zeros((points, mec.kA))
        for i in range(points):
            cfbst[i] = t[i] * scburst.length_cond_pdf(mec, t[i])
        cfbrst = cfbst.transpose()
        return t * 1000, fbst, cfbrst

    t = t * 1000 # x axis in millisec

    return t, fbst

def burst_openings_pdf(mec, n, conditional=False):
    """
    Calculate the mean number of openings per burst and data for the
    distribution of openings per burst.

    Parameters
    ----------
    mec : instance of type Mechanism
    n  : int
        Number of openings.
    conditional : bool
        True if conditional distribution is plotted.

    Returns
    -------
    r : ndarray of ints, shape (num of points,)
        Number of openings per burst.
    Pr : ndarray of floats, shape (num of points,)
        Fraction of bursts.
    cPr : ndarray of floats, shape (num of open states, num of points)
        Fraction of bursts for conditional distribution.
    """

    r = np.arange(1, n+1)
    Pr = np.zeros(n)
    for i in range(n):
        Pr[i] = scburst.openings_distr(mec, r[i])

    if conditional:
        cPr = np.zeros((n, mec.kA))
        for i in range(n):
            cPr[i] = scburst.openings_cond_distr_depend_on_start_state(mec, r[i])
        cPr = cPr.transpose()

        return r, Pr, cPr

    return r, Pr

def corr_open_shut(mec, lag):
    """
    Calculate data for the plot of open, shut and open-shut time correlations.
    
    Parameters
    ----------
    mec : instance of type Mechanism
    lag : int
        Number of lags.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    br : ndarray of floats, shape (num of points,)
        Mean burst length in millisec.
    brblk : ndarray of floats, shape (num of points,)
        Mean burst length in millisec corrected for fast block.
    """
    
    kA, kF = mec.kA, mec.kI
    GAF, GFA = qml.iGs(mec.Q, kA, kF)
    XAA, XFF = np.dot(GAF, GFA), np.dot(GFA, GAF)
    phiA, phiF = qml.phiA(mec).reshape((1,kA)), qml.phiF(mec).reshape((1,kF))
    varA = scl.corr_variance_A(phiA, mec.QAA, kA)
    varF = scl.corr_variance_A(phiF, mec.QII, kF)
    
    r = np.arange(1, lag + 1)
    roA, roF, roAF = np.zeros(lag), np.zeros(lag), np.zeros(lag)
    for i in range(lag):
        covA = scl.corr_covariance_A(i+1, phiA, mec.QAA, XAA, kA)
        roA[i] = scl.correlation_coefficient(covA, varA, varA)
        covF = scl.corr_covariance_A(i+1, phiF, mec.QII, XFF, kF)
        roF[i] = scl.correlation_coefficient(covF, varF, varF)
        covAF = scl.corr_covariance_AF(i+1, phiA, mec.QAA, mec.QII,
            XAA, GAF, kA, kF)
        roAF[i] = scl.correlation_coefficient(covAF, varA, varF)
            
    return r, roA, roF, roAF

def mean_open_next_shut(mec, tres, points=512):
    """
    Calculate plot of mean open time preceding/next-to shut time.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    sht : ndarray of floats, shape (num of points,)
        Shut times.
    mp : ndarray of floats, shape (num of points,)
        Mean open time preceding shut time.
    mn : ndarray of floats, shape (num of points,)
        Mean open time next to shut time.
    """
    
    Froots = scl.asymptotic_roots(tres,
        mec.QII, mec.QAA, mec.QIA, mec.QAI, mec.kI, mec.kA)
    tmax = (-1 / Froots.max()) * 5
    sht = np.logspace(math.log10(tres), math.log10(tmax), points)
    mp, mn = scl.HJC_adjacent_mean_open_to_shut_time_pdf(sht, tres, mec.Q, 
        mec.QAA, mec.QAI, mec.QII, mec.QIA)
        
    # return in ms
    return sht * 1000, mp * 1000, mn * 1000

def dependency_plot(mec, tres, points=512):
    """
    Calculate 3D dependency plot.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    top : ndarray of floats, shape (num of points,)
        Open times.
    tsh : ndarray of floats, shape (num of points,)
        Shut times.
    dependency : ndarray 
        Mean open time next to shut time.
    """
    
    Froots = scl.asymptotic_roots(tres,
        mec.QII, mec.QAA, mec.QIA, mec.QAI, mec.kI, mec.kA)
    tsmax = (-1 / Froots.max()) * 20
    tsh = np.logspace(math.log10(tres), math.log10(tsmax), points)
    
    Aroots = scl.asymptotic_roots(tres,
        mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kA, mec.kI)
    tomax = (-1 / Aroots.max()) * 20
    top = np.logspace(math.log10(tres), math.log10(tomax), points)
    
    dependency = scl.HJC_dependency(top, tsh, tres, mec.Q, 
        mec.QAA, mec.QAI, mec.QII, mec.QIA)
    
    return np.log10(top*1000), np.log10(tsh*1000), dependency

def burst_length_versus_conc_plot(mec, cmin, cmax):
    """
    Calculate data for the plot of burst length versus concentration.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin, cmax : float
        Range of concentrations in M.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    br : ndarray of floats, shape (num of points,)
        Mean burst length in millisec.
    brblk : ndarray of floats, shape (num of points,)
        Mean burst length in millisec corrected for fast block.
    """

    points = 100
    c = np.linspace(cmin, cmax, points)
    br = np.zeros(points)
    brblk = np.zeros(points)

    for i in range(points):
        mec.set_eff('c', c[i])
        br[i] = scburst.length_mean(mec)
        if mec.fastblock:
            brblk[i] = br[i] * (1 + c[i] / mec.KBlk)
        else:
            brblk[i] = br[i]
    c = c * 1000000 # x axis scale in mikroMoles
    br = br * 1000
    brblk= brblk * 1000

    return c, br, brblk

def conc_jump_on_off_taus_versus_conc_plot(mec, cmin, cmax, width):
    """
    Calculate data for the plot of square concentration pulse evoked current 
    (occupancy) weighted on and off time constants versus concentration.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin, cmax : float
        Range of concentrations in M.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    ton, toff : floats
        On and off weighted time constants.
    """

    points = 100
    c = np.logspace(int(np.log10(cmin)), int(np.log10(cmax)), points)
    
    wton = np.zeros(points)
    wtoff = np.zeros(points)
    ton = np.zeros((points, mec.k-1))
    toff = np.zeros((points, mec.k-1))
    for i in range(points):
        mec.set_eff('c', c[i])
        wton[i], ton[i], wtoff[i], toff[i] = cjumps.weighted_taus(mec, c[i], width)

    ton = ton.transpose()
    toff = toff.transpose()

    return c * 1000, wton * 1000, ton * 1000, wtoff * 1000, toff * 1000

def open_time_pdf(mec, tres, tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal asymptotic and exact open time distributions.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
        Ideal, exact and asymptotic open time distributions.
    """

    open = True

    # Asymptotic pdf
    roots = scl.asymptotic_roots(tres,
        mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kA, mec.kI)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    areas = scl.asymptotic_areas(tres, roots,
        mec.QAA, mec.QII, mec.QAI, mec.QIA,
        mec.kA, mec.kI, GAF, GFA)
    apdf = scl.asymptotic_pdf(t, tres, -1 / roots, areas)

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = scl.exact_GAMAxx(mec,
        tres, open)
    epdf = np.zeros(points)
    for i in range(points):
        epdf[i] = (t[i] * scl.exact_pdf(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))
            
    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, epdf, apdf

def adjacent_open_time_pdf(mec, tres, u1, u2, 
    tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate pdf's of ideal all open time and open time adjacent to specified shut
    time range.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, ajpdf : ndarrays of floats, shape (num of points)
        Ideal all and adjacent open time distributions.
    """

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    tmax = (1 / eigs.max()) * 100
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)
    
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    # Ajacent open time pdf
    eigs, w = scl.adjacent_open_to_shut_range_pdf_components(u1, u2, 
        mec.QAA, mec.QAI, mec.QII, mec.QIA, qml.phiA(mec).reshape((1,mec.kA)))
#    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ajpdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac
           
    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, ajpdf

def scaled_pdf(t, pdf, dt, n):
    """
    Scale pdf to the data histogram.

    Parameters
    ----------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    pdf : ndarray of floats, shape (num of points)
        pdf to scale.
    dt : float
        Histogram bin width in log10 units.
    n : int
        Total number of events.

    Returns
    -------
    spdf : ndarray of floats, shape (num of points)
        Scaled pdf.
    """

    spdf = n * dt * 2.30259 * pdf
    #spdf = n * dt * pdf
    return spdf

def shut_time_pdf(mec, tres, tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal asymptotic and exact shut time distributions.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
        Ideal, exact and asymptotic shut time distributions.
    """

    open = False

    # Asymptotic pdf
    roots = scl.asymptotic_roots(tres, mec.QII, mec.QAA, mec.QIA, mec.QAI,
        mec.kI, mec.kA)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    areas = scl.asymptotic_areas(tres, roots,
        mec.QII, mec.QAA, mec.QIA, mec.QAI,
        mec.kI, mec.kA, GFA, GAF)
    apdf = scl.asymptotic_pdf(t, tres, -1 / roots, areas)

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = scl.exact_GAMAxx(mec, tres, open)
    epdf = np.zeros(points)
    for i in range(points):
        epdf[i] = (t[i] * scl.exact_pdf(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, epdf, apdf

def subset_time_pdf(mec, tres, state1, state2,
    tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal pdf of any subset dwell times.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    state1, state2 : ints
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    spdf : ndarray of floats, shape (num of points)
        Subset dwell time pdf.
    """

    open = False
    if open:
        eigs, w = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    else:
        eigs, w = scl.ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))

    tmax = tau.max() * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    spdf = np.zeros(points)
    for i in range(points):
        spdf[i] = t[i] * scl.ideal_subset_time_pdf(mec.Q,
            state1, state2, t[i]) * fac

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, spdf

def png_save_pdf_fig(outfile, ints, mec, conc, tres, type):
    x, y, dx = prepare_hist(ints, tres)
    mec.set_eff('c', conc)
    if type == 'open':
        t, ipdf, epdf, apdf = open_time_pdf(mec, tres)
    elif type == 'shut':
        t, ipdf, epdf, apdf = shut_time_pdf(mec, tres)
    else:
        print ('Wrong type.')

    sipdf = scaled_pdf(t, ipdf, math.log10(dx), len(ints))
    sepdf = scaled_pdf(t, epdf, math.log10(dx), len(ints))
    figure(figsize=(6, 4))
    semilogx(x*1000, y, 'k-', t, sipdf, 'r--', t, sepdf, 'b-')
    savefig(outfile, bbox_inches=0)