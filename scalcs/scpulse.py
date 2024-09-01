import numpy as np
from scalcs import scalcslib as scl
from scalcs import qmatlib as qm
from scalcs import pdfs

def first_latency_asymptotic_roots_areas(tres, mec0, mec1):
    """
    Find the areas of the asymptotic pdf.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    mec0 : dcpyps.Mechanism
        The initial mechanism.
    mec1 : dcpyps.Mechanism
        The final mechanism.

    Returns
    -------
    roots : ndarray
        Roots of the asymptotic pdf.
    areas : ndarray
        Areas of the asymptotic pdf.
    """
    phiF0 = qm.pinf(mec0.Q)[mec0.kA:]
    roots = scl.asymptotic_roots(tres, mec1.QFF, mec1.QAA, mec1.QFA, mec1.QAF, mec1.kF, mec1.kA)
    R = qm.AR(roots, tres, mec1.QFF, mec1.QAA, mec1.QFA, mec1.QAF, mec1.kF, mec1.kA)
    uA = np.ones((mec1.kA, 1))
    #areas = -1 / roots * np.einsum('ijk,jk->i', R, np.dot(mec1.QFA, qm.expQ(mec1.QAA, tres)).dot(uA))
    # Initialize areas array
    areas = np.zeros(mec1.kF)

    # Calculate areas for each root
    for i in range(mec1.kF):
        areas[i] = (-1 / roots[i]) * np.dot(phiF0, np.dot(np.dot(R[i], np.dot(mec1.QFA, qm.expQ(mec1.QAA, tres))), uA)).item()

    return roots, areas

def first_latency_exact_GAMAxx(tres, mec, phi):
    """
    Calculate gamma coefficients for the exact pdf.

    Parameters
    ----------
    tres : float
        Time resolution.
    phi : ndarray
        Probability distribution of initial states.
    mec : dcpyps.Mechanism
        The mechanism to be analyzed.

    Returns
    -------
    eigen : ndarray
        Eigenvalues of -Q matrix.
    gamma00, gamma10, gamma11 : ndarrays
        Constants for the exact open/shut time pdf.
    """

    expQAA = qm.expQ(mec.QAA, tres)
    eigen, A = qm.eigenvalues_and_spectral_matrices(-mec.Q)
    Z00, Z10, Z11 = qm.Zxx(mec.Q, eigen, A, mec.kA, mec.QAA, mec.QFA, mec.QAF, expQAA, False)

    u = np.ones((mec.kA, 1))
    gamma00 = np.dot(np.dot(phi, Z00), u).ravel()
    gamma10 = np.dot(np.dot(phi, Z10), u).ravel()
    gamma11 = np.dot(np.dot(phi, Z11), u).ravel()

    return eigen, gamma00, gamma10, gamma11

def calculate_first_latency_pdf_exact(tres, tseq, mec0, mec1):
    """
    Calculate the exact first latency probability density function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    tseq : ndarray
        Time sequence for PDF calculation.
    mec0 : dcpyps.Mechanism
        The initial mechanism.
    mec1 : dcpyps.Mechanism
        The final mechanism.

    Returns
    -------
    epdf : ndarray
        Exact first latency probability density function.
    """
    phiF0 = qm.pinf(mec0.Q)[mec0.kA:]
    roots, areas = first_latency_asymptotic_roots_areas(tres, mec0, mec1)
    eigen, gamma00, gamma10, gamma11 = first_latency_exact_GAMAxx(tres, mec1, phiF0)
    #epdf = np.vectorize(lambda ti: scl.exact_pdf(ti, tres, roots, areas, eigen, gamma00, gamma10, gamma11))(tseq)
    epdf = np.zeros(len(tseq))
    for i in range(len(tseq)):
        epdf[i] = (scl.exact_pdf(tseq[i], tres,
            roots, areas, eigen, gamma00, gamma10, gamma11))
    return epdf

def calculate_first_latency_pdf_asymptotic(tres, tseq, mec0, mec1):
    """
    Calculate the asymptotic first latency probability density function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    tseq : ndarray
        Time sequence for PDF calculation.
    mec0 : dcpyps.Mechanism
        The initial mechanism.
    mec1 : dcpyps.Mechanism
        The final mechanism.

    Returns
    -------
    apdf : ndarray
        Asymptotic first latency probability density function.
    """
    roots, areas = first_latency_asymptotic_roots_areas(tres, mec0, mec1)
    t1, t2 = tseq[tseq < tres], tseq[tseq >= tres]
    apdf2 = pdfs.expPDF(t2 - tres, -1 / roots, areas)
    apdf = np.concatenate((t1 * 0.0, apdf2))
    return apdf

def calculate_first_latency_pdf_ideal(tseq, mec0, mec1):
    """
    Calculate the ideal first latency probability density function.

    Parameters
    ----------
    tseq : ndarray
        Time sequence for PDF calculation.
    mec0 : dcpyps.Mechanism
        The initial mechanism.
    mec1 : dcpyps.Mechanism
        The final mechanism.

    Returns
    -------
    ipdf : ndarray
        Ideal first latency probability density function.
    """
    phiF0 = qm.pinf(mec0.Q)[mec0.kA:]
    eigs, w = scl.ideal_dwell_time_pdf_components(mec1.QFF, phiF0)
    ipdf = pdfs.expPDF(tseq, 1 / eigs, w / eigs)
    return ipdf
