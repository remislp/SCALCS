from math import sqrt
import numpy as np
from tabulate import tabulate

from scalcs.qmatlib import QMatrix
from scalcs.scburst import SCBurst

class QMatrixPrints(QMatrix):
    '''
    Print Q-Matrix stuff.
    '''

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        # Initialize the QMatrix instance.
        QMatrix.__init__(self, Q, kA=kA, kB=kB, kC=kC, kD=kD)

    def print_Q(self):
        q_mat_str = '\nQ (units [1/s]) = \n'
        q_mat_str += tabulate((['{:.3f}\t'.format(item) for item in row]
                              for row in self.Q), 
                              headers=[i+1 for i in range(self.k)], 
                              showindex=[i+1 for i in range(self.k)],
                              tablefmt='orgtbl')       
        print(q_mat_str)

    def print_pinf(self):
        print('\nEquilibrium state occupancies:')
        print(tabulate([self.pinf], headers=[i+1 for i in range(self.k)], tablefmt='orgtbl' ))

    def print_Popen(self):
        print('\nEquilibrium open probability Popen = {0:.6f}'.format(self.Popen()))

    def print_state_lifetimes(self):
        print('\nMean lifetimes of each individual state (ms):')
        print(tabulate([1000 / self.state_lifetimes()], headers=[i+1 for i in range(self.k)], tablefmt='orgtbl' ))

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
        print(info_str)

    def print_subset_probabilities(self):

        info_str = '\nInitial probabilities of finding channel in classes of states:'
        info_str += '\nA states: P1(A)= {:.4g}'.format(self.P('A'))
        info_str += '\nB states: P1(B)= {:.4g}'.format(self.P('B'))
        info_str += '\nC states: P1(C)= {:.4g}'.format(self.P('C'))
        info_str += '\nF states: P1(F)= {:.4g}'.format(self.P('F'))

        info_str += '\n\nConditional probabilities:'
        info_str += '\nP1(B|F)= {:.4g}'.format(self.P('B|F')) + '\t\tOf channels that a shut (in F) at t=0, probability being in B'
        info_str += '\nP1(C|F)= {:.4g}'.format(self.P('C|F')) + '\t\tOf channels that a shut (in F) at t=0, probability being in C'

        print(info_str)

    def print_initial_vectors(self):
        info_str = '\nInitial vectors (conditional probability distribution over a subset of states):'
        info_str += '\nphi(A)= {}'.format(self.phi('A'))
        info_str += '\nphi(B)= {}'.format(self.phi('B'))
        info_str += '\nphi(F)= {}'.format(self.phi('F'))
        print(info_str)

    def print_DC_table(self):
        """
        """
        
        mean_latencies = []
        for i in range(1, self.k+1):
            mean_latencies.append(self.mean_latency_given_start_state(i))

        print('\n') 
        header_open = ['Open \nstates', 'Equilibrium\n occupancy', 'Mean lifetime\n (ms)',
                       'Mean latency (ms)\nto next shutting\ngiven start \nin this state']
        DCtable_open = []
        for i in range(self.kA):
            if i == 0:
                mean_life_A = self.subset_mean_lifetime(i+1, self.kA)
                DCtable_open.append(['Subset A ', sum(self.pinf[:self.kA]), mean_life_A * 1000, ' '])
            DCtable_open.append([i+1, self.pinf[i], 1000*self.state_lifetimes()[i], 1000*mean_latencies[i]])

        print(tabulate(DCtable_open, headers=header_open, tablefmt='orgtbl', floatfmt=".6f"), '\n\n')

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

        print(tabulate(DCtable_shut, headers=header_shut, tablefmt='orgtbl', floatfmt=(None, '.6f', '.6g', '.6g',)))
    

class SCBurstPrints(SCBurst):
    '''
    Print Q-Matrix stuff.
    '''

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        SCBurst.__init__(self, Q, kA=kA, kB=kB, kC=kC, kD=kD)

    def print_start_end_vectors(self):

        print('\nBurst start vector = ')
        print(self.start_burst())
        print('\nBurst end vector =')
        print(self.end_burst())

    def print_means(self):
        
        print('\nMean number of opening per burst = {0:.6g}'.format(self.mean_number_of_openings()))
#        print('\nNo of gaps within burst per unit open time = {0:.6g} \n'.
#              format((self.mean_number_of_openings() - 1) / self.mean_open_time()))
        print('Mean burst length (ms)= {0:.6g}'.format(1000 * self.mean_length()))
        print('Mean open time per burst (ms)= {0:.6g}'.format(1000 * self.mean_open_time()))
        print('Mean shut time per burst (ms; all bursts)= {0:.6g}'.format(1000 * self.mean_shut_time()))
        print('Mean shut time per burst (ms; excluding single opening bursts)= {0:.6g}'.
            format(1000 * self.mean_shut_time() / self.probability_more_than_one_opening()))
        print('Mean shut time between bursts (ms)= {0:.6g}'.format(1000 * self.mean_shut_times_between_bursts()))

        print('\nPopen WITHIN BURST = (open time/burst)/(bst length) = {0:.5g} \n'.format(self.mean_open_time() / self.mean_length()))
        print('Total Popen = (open time/bst)/(bst_length + mean gap between burst) = {0:.5g} \n'.
              format(self.mean_open_time() / (self.mean_length() + self.mean_shut_times_between_bursts())))

    def print_length_pdf(self):
        e1, w1 = self.length_pdf_components()
        print(expPDF_printout(e1, w1, 'PDF of total burst length, unconditional'))
        
        e2, w2 = self.length_pdf_no_single_openings_components()
        print(expPDF_printout(e2, w2, 'PDF of burst length for bursts with 2 or more openings'))

    def print_openings_pdf(self):
        e1, w1 = self.total_open_time_pdf_components()
        print(expPDF_printout(e1, w1, 'PDF of total open time per burst'))

        e2, w2 = self.first_opening_length_pdf_components()
        print(expPDF_printout(e2, w2, 'PDF of first opening in a burst with 2 or more openings'))

        rho, w = self.openings_distr_components()
        print(geometricPDF_printout(rho, w, 'Geometric PDF of of number (r) of openings / burst (unconditional)'))

    def print_shuttings_pdf(self):

        e1, w1 = self.shut_times_inside_burst_pdf_components()
        print(expPDF_printout(e1, w1, 'PDF of gaps inside burst'))

        e2, w2 = self.shut_times_between_burst_pdf_components()
        print(expPDF_printout(e2, w2, 'PDF of gaps between burst'))

        e3, w3 = self.shut_time_total_pdf_components_2more_openings()
        print(expPDF_printout(e3, w3, 'PDF of total shut time per bursts for bursts with at least 2 openings'))

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
