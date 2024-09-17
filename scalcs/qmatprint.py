import sys
from math import sqrt
import numpy as np
from tabulate import tabulate
from numpy import linalg as nplin

from scalcs import qmatlib as qml
from scalcs.qmatlib import QMatrix
from scalcs.scburst import SCBurst
from scalcs.scalcslib import SCDwells, SCCorrelations

class QMatrixPrints(QMatrix):
    '''
    Print Q-Matrix stuff.
    '''

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        # Initialize the QMatrix instance.
        QMatrix.__init__(self, Q, kA=kA, kB=kB, kC=kC, kD=kD)

    @property
    def print_Q(self):
        q_mat_str = '\nQ (units [1/s]) = \n'
        q_mat_str += tabulate((['{:.3f}\t'.format(item) for item in row]
                              for row in self.Q), 
                              headers=[i+1 for i in range(self.k)], 
                              showindex=[i+1 for i in range(self.k)],
                              tablefmt='orgtbl')       
        return q_mat_str

    @property
    def print_pinf(self):
        pinf_str = '\nEquilibrium state occupancies:'
        pinf_str += tabulate([self.pinf], headers=[i+1 for i in range(self.k)], tablefmt='orgtbl' )
        return pinf_str

    @property
    def print_Popen(self):
        return '\nEquilibrium open probability Popen = {0:.6f}'.format(self.Popen())

    @property
    def print_state_lifetimes(self):
        lifetimes_str = '\nMean lifetimes of each individual state (ms):'
        lifetimes_str += tabulate([1000 / self.state_lifetimes()], headers=[i+1 for i in range(self.k)], tablefmt='orgtbl' )
        return lifetimes_str

    @property
    def print_transition_matrices(self):
        """
        Print the transition probabilities and frequencies.
        """
        pi = self.transition_probability()
        fi = self.transition_frequency()

        info_str = "\n\nProbability of transitions regardless of time:\n"
        info_str += tabulate((['{:.4f}\t'.format(item) for item in row] for row in pi), 
                              headers=[i+1 for i in range(self.k)], 
                              showindex=[i+1 for i in range(self.k)],
                              tablefmt='orgtbl')       

        info_str += "\n\nFrequency of transitions (per second):\n"
        info_str += tabulate((['{:.4f}\t'.format(item) for item in row] for row in fi), 
                              headers=[i+1 for i in range(self.k)], 
                              showindex=[i+1 for i in range(self.k)],
                              tablefmt='orgtbl')
        return info_str

    @property
    def print_subset_probabilities(self):

        info_str = '\nInitial probabilities of finding channel in classes of states:'
        info_str += '\nA states: P1(A)= {:.4g}'.format(self.P('A'))
        info_str += '\nB states: P1(B)= {:.4g}'.format(self.P('B'))
        info_str += '\nC states: P1(C)= {:.4g}'.format(self.P('C'))
        info_str += '\nF states: P1(F)= {:.4g}'.format(self.P('F'))

        info_str += '\n\nConditional probabilities:'
        info_str += '\nP1(B|F)= {:.4g}'.format(self.P('B|F')) + '\t\tOf channels that a shut (in F) at t=0, probability being in B'
        info_str += '\nP1(C|F)= {:.4g}'.format(self.P('C|F')) + '\t\tOf channels that a shut (in F) at t=0, probability being in C'

        return info_str

    @property
    def print_initial_vectors(self):
        info_str = '\nInitial vectors (conditional probability distribution over a subset of states):'
        info_str += '\nphi(A)= {}'.format(self.phi('A'))
        info_str += '\nphi(B)= {}'.format(self.phi('B'))
        info_str += '\nphi(F)= {}'.format(self.phi('F'))
        return info_str

    @property
    def print_DC_table(self):
        """
        """
        
        mean_latencies = []
        for i in range(1, self.k+1):
            mean_latencies.append(self.mean_latency_given_start_state(i))

        dc_str = '\n'
        header_open = ['Open \nstates', 'Equilibrium\n occupancy', 'Mean lifetime\n (ms)',
                       'Mean latency (ms)\nto next shutting\ngiven start \nin this state']
        DCtable_open = []
        for i in range(self.kA):
            if i == 0:
                mean_life_A = self.subset_mean_lifetime(i+1, self.kA)
                DCtable_open.append(['Subset A ', sum(self.pinf[:self.kA]), mean_life_A * 1000, ' '])
            DCtable_open.append([i+1, self.pinf[i], 1000*self.state_lifetimes()[i], 1000*mean_latencies[i]])

        dc_str += (tabulate(DCtable_open, headers=header_open, tablefmt='orgtbl', floatfmt=".6f") + '\n\n')

        header_shut = ['Shut\nstates', 'Equilibrium\n occupancy', 'Mean lifetime\n (ms)',
                       'Mean latency (ms)\nto next opening\ngiven start \nin this state']
        DCtable_shut = []
        for i in range(self.kA, self.k):
            if i == self.kA:
                mean_life_B = self.subset_mean_lifetime(self.kA+1, self.kE)
                DCtable_shut.append(['Subset B ', sum(self.pinf[self.kA: self.kE]), mean_life_B * 1000, '-'])
            if i == self.kE:
                mean_life_C = self.subset_mean_lifetime(self.kE+1, self.kG)
                DCtable_shut.append(['Subset C ', sum(self.pinf[self.kE: self.kG]), mean_life_C * 1000, '-'])
            if i == self.kG:
                mean_life_D = self.subset_mean_lifetime(self.kG+1, self.k)
                DCtable_shut.append(['Subset D ', sum(self.pinf[self.kG: self.k]), mean_life_D * 1000, '-'])
            DCtable_shut.append([i+1, self.pinf[i], 1000*self.state_lifetimes()[i], 1000*mean_latencies[i]])

        dc_str += tabulate(DCtable_shut, headers=header_shut, tablefmt='orgtbl', floatfmt=(None, '.6f', '.6g', '.6g',))
        return dc_str

class SCBurstPrints(SCBurst):
    '''
    Print Q-Matrix stuff.
    '''

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        SCBurst.__init__(self, Q, kA=kA, kB=kB, kC=kC, kD=kD)

    @property
    def print_all(self):
        """
        Output burst calculations.
        """

        all_str = ('\n*******************************************\n' +
        'CALCULATED SINGLE CHANNEL BURST MEANS, PDFS, ETC....\n')

        all_str += self.print_start_end_vectors
        all_str += self.print_popen
        all_str += self.print_length_pdf
        all_str += self.print_openings_pdf
        all_str += self.print_shuttings_pdf
        return all_str

    @property
    def print_start_end_vectors(self):

        info_str = '\nBurst start vector = '
        for i in range(self.kA):
            info_str += '\t{0:.5g}\t'.format(self.start_burst[i])
        info_str += '\nBurst end vector ='
        for i in range(self.kA):
            info_str += '\t{0:.5g}\t'.format(self.end_burst[i, 0])
        return info_str

    @property
    def print_means(self):
        
        info_str = '\n\nMean number of opening per burst = {0:.6g}'.format(self.mean_number_of_openings)
#        print('\nNo of gaps within burst per unit open time = {0:.6g} \n'.
#              format((self.mean_number_of_openings() - 1) / self.mean_open_time()))
        info_str += '\nMean burst length (ms)= {0:.6g}'.format(1000 * self.mean_length)
        info_str += '\nMean open time per burst (ms)= {0:.6g}'.format(1000 * self.mean_open_time)
        info_str += '\nMean shut time per burst (ms; all bursts)= {0:.6g}'.format(1000 * self.mean_shut_time)
        info_str += ('\nMean shut time per burst (ms; excluding single opening bursts)= {0:.6g}'.
            format(1000 * self.mean_shut_time / self.probability_more_than_one_opening))
        info_str += '\nMean shut time between bursts (ms)= {0:.6g}'.format(1000 * self.mean_shut_times_between_bursts)
        return info_str
 
    @property
    def print_popen(self):
        
        info_str = '\n\nPopen WITHIN BURST = (open time/burst)/(bst length) = {0:.5g}'.format(self.burst_popen)
        info_str += ('\nTotal Popen = (open time/bst)/(bst_length + mean gap between burst) = {0:.5g}'.format(self.total_popen))
        return info_str

    @property
    def print_length_pdf(self):
        e1, w1 = self.length_pdf_components()
        info_str = (expPDF_printout(e1, w1, '\nPDF of total burst length, unconditional'))
        e2, w2 = self.length_pdf_no_single_openings_components()
        info_str += expPDF_printout(e2, w2, '\nPDF of burst length for bursts with 2 or more openings')
        return info_str
 
    @property
    def print_openings_pdf(self):
        e1, w1 = self.total_open_time_pdf_components()
        info_str = expPDF_printout(e1, w1, '\nPDF of total open time per burst')
        e2, w2 = self.first_opening_length_pdf_components()
        info_str += (expPDF_printout(e2, w2, '\nPDF of first opening in a burst with 2 or more openings'))
        rho, w = self.openings_distr_components()
        info_str += (geometricPDF_printout(rho, w, '\nGeometric PDF of of number (r) of openings / burst (unconditional)'))
        return info_str

    @property
    def print_shuttings_pdf(self):
        e1, w1 = self.shut_times_inside_burst_pdf_components()
        info_str = (expPDF_printout(e1, w1, '\nPDF of gaps inside burst'))
        e2, w2 = self.shut_times_between_burst_pdf_components()
        info_str += (expPDF_printout(e2, w2, '\nPDF of gaps between burst'))
        e3, w3 = self.shut_time_total_pdf_components_2more_openings()
        info_str += (expPDF_printout(e3, w3, '\nPDF of total shut time per bursts for bursts with at least 2 openings'))
        return info_str

def expPDF_printout(eigs, amps, title):
    """
    """

    info_str = '\n'+title+ '\n'
    table = []
    for i in range(len(eigs)):
        table.append([i+1, amps[i], eigs[i], 1000 / eigs[i], 100 * amps[i] / eigs[i]])

    info_str += tabulate(table, 
                              headers=['Term', 'Amplitude', 'Rate (1/sec)', 'tau (ms)', 'Area (%)'], 
                              tablefmt='orgtbl')       

    mean = np.sum((amps/eigs) * (1/eigs))
    var = np.sum((amps/eigs) * (1/eigs) * (1/eigs))
    sd = sqrt(2 * var - mean * mean)
    info_str += ('\nMean (ms) = {0:.5g}'.format(mean * 1000) +
        '\t\tSD = {0:.5g}'.format(sd * 1000) +
        '\t\tSD/mean = {0:.5g}\n'.format(sd / mean))
        
    return info_str

def expPDF_asymptotic_printout(eigs, areas, tres, title):
    """
    """

    info_str = '\n'+title+ '\n'
    areast0 = areas * np.exp(tres * eigs)
    areast0 /= np.sum(areast0)
    table = []
    for i in range(len(eigs)):
        table.append([i+1, eigs[i], 1000 / eigs[i], 100 * areas[i], 100 * areast0[i]])
    info_str += tabulate(table, 
                              headers=['Term', 'Rate (1/sec)', 'tau (ms)', 'Area (%)', 'Area renormalised for t=0 to inf'], 
                              tablefmt='orgtbl')       
    return info_str

def expPDF_exact_printout(eigs, gamma00, gamma10, gamma11, title):
    """
    """

    info_str = '\n'+title+ '\n'
    table = []
    for i in range(len(eigs)):
        table.append([eigs[i], gamma00[i], gamma10[i], gamma11[i]])
    info_str += tabulate(table, 
                              headers=['Eigenvalue', 'g00(m)', 'g10(m)', 'g11(m)'], 
                              tablefmt='orgtbl')       
    return info_str


def geometricPDF_printout(rho, w, title):
    """
    """

    info_str = '\n'+title+ '\n'
    table = []
    norm = 1 / (np.ones((rho.shape[0])) - rho)
    for i in range(len(rho)):
        table.append([i+1, w[i], rho[i], 100 * w[i] * norm[i], norm[i]])

    info_str += tabulate(table, 
                              headers=['Term', 'w', 'rho', 'area(%)', 'Norm mean'], 
                              tablefmt='orgtbl')
            
    k = rho.shape[0]
    mean = np.sum(w / np.power(np.ones((k)) - rho, 2))
    var = np.sum(w * (np.ones((k)) + rho) / np.power(np.ones((k)) - rho, 3))
    sd = sqrt(var - mean * mean)

    info_str += ('\nMean number of openings per burst =\t {0:.5g}'.format(mean) +
        '\n\tSD =\t {0:.5g}'.format(sd) +
        '\tSD/mean =\t {0:.5g}\n'.format(sd / mean))
    return info_str


class SCDwellsPrints(SCDwells):
    '''
    Print Q-Matrix stuff.
    '''

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        SCDwells.__init__(self, Q, kA=kA, kB=kB, kC=kC, kD=kD)

    @property
    def print_all(self):
        """
        Output burst calculations.
        """

        all_str = ('\n*******************************************\n' +
        'CALCULATED SINGLE CHANNEL IDEAL DWELL MEANS, PDFS, ETC....\n')

        return all_str

    @property
    def print_initial_vectors(self):
        initial_str = ('\nInitial vector for ideal openings =\n')
        #phiAi = qml.phiA(mec)
        for i in range(self.kA):
            initial_str += ('\t{0:.5g}'.format(self.phiA[i]))
        initial_str += ('\nInitial vector for ideal shuttings =\n')
        #phiFi = qml.phiF(mec)
        for i in range(self.kF):
            initial_str += ('\t{0:.5g}'.format(self.phiF[i]))
        initial_str += '\n'
        return initial_str

    @property
    def print_initial_HJC_vectors(self):
        initial_str = ('\nInitial vector for HJC openings (tres={0:.0f} us) =\n'.format(1e6 * self.tres))
        for i in range(self.kA):
            initial_str += ('\t{0:.5g}'.format(self.HJCphiA[i]))
        initial_str += ('\nInitial vector for HJC shuttings =\n')
        for i in range(self.kF):
            initial_str += ('\t{0:.5g}'.format(self.HJCphiF[i]))
        initial_str += '\n'
        return initial_str

    @property
    def print_open_time_pdf(self):
        e, w = self.ideal_open_time_pdf_components()
        info_str = (expPDF_printout(e, w, '\nIdeal open time, unconditional'))
        return info_str

    @property
    def print_open_time_pdf(self):
        e, w = self.ideal_open_time_pdf_components()
        info_str = (expPDF_printout(e, w, '\nIdeal open time, unconditional'))
        return info_str
    
    @property
    def print_HJC_asymptotic_open_time_pdf(self):

        e, a = self.HJC_asymptotic_open_time_pdf_components()
        info_str = expPDF_asymptotic_printout(e, a, self.tres, '\nASYMPTOTIC OPEN TIME DISTRIBUTION')
        info_str += ('\nApparent mean open time (ms) = {0:.5g}\n'.format(self.apparent_mean_open_time * 1000))
        return info_str

    @property
    def print_HJC_exact_open_time_pdf(self):

        eigvals, gamma00, gamma10, gamma11 = self.exact_GAMAxx(open=True)
        info_str = expPDF_exact_printout(eigvals, gamma00, gamma10, gamma11, '\nEXACT OPEN TIME DISTRIBUTION')
        return info_str
    
    @property
    def print_HJC_exact_shut_time_pdf(self):

        eigvals, gamma00, gamma10, gamma11 = self.exact_GAMAxx(open=False)
        info_str = expPDF_exact_printout(eigvals, gamma00, gamma10, gamma11, '\nEXACT SHUT TIME DISTRIBUTION')
        return info_str

    @property
    def print_shut_time_pdf(self):
        e, w = self.ideal_shut_time_pdf_components()
        info_str = (expPDF_printout(e, w, '\nIdeal shut time, unconditional'))
        return info_str

    @property
    def print_HJC_asymptotic_shut_time_pdf(self):

        e, a = self.HJC_asymptotic_shut_time_pdf_components()
        info_str = expPDF_asymptotic_printout(e, a, self.tres, '\nASYMPTOTIC SHUT TIME DISTRIBUTION')
        info_str += ('\nApparent mean shut time (ms) = {0:.5g}\n'.format(self.apparent_mean_shut_time * 1000))
        return info_str


class CorrelationPrints(SCCorrelations):
    """
    Prints various correlation and Q-matrix calculations.
    """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD)
        self.varA, self.varF = self._variance(open=True), self._variance(open=False)

    @property
    def print_all(self):
        return (f"\n***** CORRELATIONS *****\n" +
                self.print_ranks +
                self.print_open_correlations +
                self.print_shut_correlations +
                self.print_open_shut_correlations)

    @property
    def print_ranks(self):
        """
        Print ranks and eigenvalues of the matrices.
        """
        return (f"\n Ranks of GAF, GFA = {self.rank_GAF}, {self.rank_GFA}"
                f"\n Rank of GFA * GAF = {self.rank_XFF}"
                f"\n Rank of GAF * GFA = {self.rank_XAA}")

    def _format_correlation_info(self, var, var_n, n, correlation_limit, correlation_type):
        """
        Helper method to format correlation information for open and shut times.
        """
        percentage_diff = 100 * (sqrt(var_n / (n * n)) - sqrt(var / n)) / sqrt(var / n)
        limiting_percentage = 100 * (sqrt(1 + 2 * correlation_limit / var) - 1)
        
        return (f"\nVariance of {correlation_type} time = {var:.5g}\n"
                f"SD of all {correlation_type} times = {sqrt(var) :.5g}\n"
                f"SD of means of {n} {correlation_type} times if uncorrelated = {sqrt(var / n) :.5g}\n"
                f"Actual SD of mean = {sqrt(var_n / (n * n)) :.5g}\n"
                f"Percentage difference as result of correlation = {percentage_diff:.5g}\n"
                f"Limiting value of percent difference for large n = {limiting_percentage:.5g}")

    def _format_correlation_coefficients(self, var, n, open=True):
        """
        Helper method to format correlation coefficients for open or shut times.
        """
        correlation_str = '\nCorrelation coefficients, r(k), for up to lag k = 5:'
        for i in range(n):
            cov = self._covariance(i + 1, open=open)
            corr_coeff = self._coefficient(cov, var, var)
            correlation_str += f"\nr({i+1}) = {corr_coeff:.5g}"
        return correlation_str

    @property
    def print_open_correlations(self):
        """
        Print open-open time correlations.
        """

        varA_n = self.variance_n(50, open=True)
        correlation_limit_A = self.correlation_limit(open=True)

        open_str = '\n\n OPEN-OPEN TIME CORRELATIONS'
        open_str += self._format_correlation_info(self.varA, varA_n, 50, correlation_limit_A, 'open')
        open_str += self._format_correlation_coefficients(self.varA, 5, open=True)
        
        return open_str

    @property
    def print_shut_correlations(self):
        """
        Print shut-shut time correlations.
        """

        varF_n = self.variance_n(50, open=False)
        correlation_limit_F = self.correlation_limit(open=False)

        shut_str = '\n\n SHUT-SHUT TIME CORRELATIONS'
        shut_str += self._format_correlation_info(self.varF, varF_n, 50, correlation_limit_F, 'shut')
        shut_str += self._format_correlation_coefficients(self.varF, 5, open=False)

        return shut_str

    @property
    def print_open_shut_correlations(self):
        """
        Print open-shut time correlations.
        """
        open_shut_str = '\n\n OPEN - SHUT TIME CORRELATIONS'
        open_shut_str += '\nCorrelation coefficients, r(k), for up to lag k = 5:'
        
        for i in range(5):
            covAF = self.covariance_AF(i + 1)
            open_shut_str += f"\nr({i+1}) = {self._coefficient(covAF, self.varA, self.varF):.5g}"
        
        return open_shut_str
