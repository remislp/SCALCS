#!/usr/bin/env python
"""
Calculate the probability density function (pdf) of the time to first opening
"""

import math
import matplotlib.pyplot as plt
import numpy as np

from samples import samples
from scalcs import scpulse as scp


def plot_first_latency_pdf(tseq, ipdf, apdf, epdf, x_lims=None):
    
    plt.plot(tseq, ipdf, 'r--', label="ideal pdf")
    plt.plot(tseq, apdf, 'b--', label="asymptotic pdf")
    plt.plot(tseq, epdf, 'b-', label="exact + asymptotic pdf")
    plt.legend(loc="upper right")
    if x_lims:
        plt.xlim(x_lims)
    plt.show()

def load_example(c0, c1, tstart, tend, points, eff_func):
    mec0 = eff_func()
    mec0.set_eff('c', c0)
    
    mec1 = eff_func()
    mec1.set_eff('c', c1)

    tseq = np.logspace(math.log10(tstart), math.log10(tend), points)
    return tseq, mec0, mec1

def run_simulation(tres, tseq, mec0, mec1):
    
    # Calculate first latency ideal pdf
    ipdf = scp.calculate_first_latency_pdf_ideal(tseq, mec0, mec1)

    # Calculate first latency asymptotic pdf
    apdf = scp.calculate_first_latency_pdf_asymptotic(tres, tseq, mec0, mec1)

    # Calculate first latency exact pdf
    epdf = scp.calculate_first_latency_pdf_exact(tres, tseq, mec0, mec1)
        
    plot_first_latency_pdf(tseq, ipdf, apdf, epdf)
    return ipdf, apdf, epdf


if __name__ == "__main__":

    # Load Colquhoun & Hawkes 1982 numerical example
    c0_1, c1_1 = 0.0, 0.0001
    tstart_1, tend_1, points_1 = 1.0e-6, 0.5e-3, 512
    tseq1, mec0_1, mec1_1 = load_example(c0_1, c1_1, tstart_1, tend_1, points_1, samples.CH82)
    tres_1 = 0.0001
    ipdf1, apdf1, epdf1 = run_simulation(tres_1, tseq1, mec0_1, mec1_1)

    # Load Colquhoun, Hawkes, Merlushkin & Edmonds 1997 numerical example
    c0_2, c1_2 = 0.0, 0.001
    tstart_2, tend_2, points_2 = 1.0e-6, 0.1, 512
    tseq2, mec0_2, mec1_2 = load_example(c0_2, c1_2, tstart_2, tend_2, points_2, samples.CHME97)
    tres_2 = 0.0007
    ipdf2, apdf2, epdf2 = run_simulation(tres_2, tseq2, mec0_2, mec1_2)
    plot_first_latency_pdf(tseq2, ipdf2, apdf2, epdf2, x_lims=[0.0, 0.001])

    print('Finished!')