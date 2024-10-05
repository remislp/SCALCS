import math
import sys
from deprecated import deprecated
import scipy.optimize as so
import numpy as np
from tabulate import tabulate


class TCrits:
    """
    Calculates critical times between exponential components.
    
    Parameters:
    ----------
    taus : np.array
        Array of time constants (taus) for different components.
    areas : np.array
        Array of areas corresponding to the exponential components.
    """

    def __init__(self, taus, areas):
        # Sort taus and areas based on the real part of taus
        sorted_indices = taus.real.argsort()
        self.taus = taus[sorted_indices]
        self.areas = areas[sorted_indices]
        self.num_components = self.taus.shape[0] - 1
        
        # Initialize a (3 x num_components) array to store tcrits for each method
        self.tcrits = np.empty((3, self.num_components))

        # Initialize storage for misclassified numbers and fractions
        # Shape: (3 criteria, num_components, 4 values [enf, ens, pf, ps])
        self.misclassified_results = np.empty((3, self.num_components, 4))
        
        # Calculate critical times for all components using the different methods
        self.calculate_tcrits()

    def calculate_tcrits(self):
        """
        Calculate the critical times for all methods (DC, CN, Jackson).
        """
        for i in range(self.num_components):
            self.tcrits[0, i] = self.find_tcrit(self.tcrit_DC, i, 0)
            self.tcrits[1, i] = self.find_tcrit(self.tcrit_CN, i, 1)
            self.tcrits[2, i] = self.find_tcrit(self.tcrit_Jackson, i, 2)

    def find_tcrit(self, criterion_function, comp_index, crit_index):
        """
        Find the critical time `tcrit` using a given criterion function, and store misclassification results.
        
        Parameters:
        ----------
        criterion_function : callable
            The function to calculate the critical time based on the desired criterion.
        comp_index : int
            The current component index for which the tcrit is calculated.
        crit_index : int
            Index indicating which criterion (DC, CN, Jackson) is being used.
        
        Returns:
        -------
        tcrit : float or None
            The critical time if found, otherwise None.
        """
        try:
            tcrit = so.bisect(
                criterion_function,
                self.taus[comp_index],
                self.taus[comp_index + 1],
                args=(self.taus, self.areas, comp_index + 1)
            )
            # Store misclassified numbers and fractions at this critical time
            enf, ens, pf, ps = self.misclassified(tcrit, self.taus, self.areas, comp_index + 1)
            self.misclassified_results[crit_index, comp_index, :] = [enf, ens, pf, ps]

        except (ValueError, RuntimeError):
            tcrit = None
            # Store NaNs for misclassified values in case of failure
            self.misclassified_results[crit_index, comp_index, :] = [np.nan, np.nan, np.nan, np.nan]
        
        return tcrit

    def misclassified(self, tcrit, taus, areas, comp):
        """
        Calculate the number and fraction of misclassified events after division by tcrit.
        
        Parameters:
        ----------
        tcrit : float
            The critical time dividing fast and slow components.
        taus : np.array
            Array of time constants (taus).
        areas : np.array
            Array of areas corresponding to the exponential components.
        comp : int
            Index indicating the separation between fast and slow components.
        
        Returns:
        -------
        enf, ens, pf, ps : tuple
            Number and fraction of misclassified events for fast and slow components.
        """
        t_fast, t_slow = taus[:comp], taus[comp:]
        a_fast, a_slow = areas[:comp], areas[comp:]
        
        # Number of misclassified events
        enf = np.sum(a_fast * np.exp(-tcrit / t_fast))
        ens = np.sum(a_slow * (1 - np.exp(-tcrit / t_slow)))
        
        # Fraction misclassified
        pf = enf / np.sum(a_fast)
        ps = ens / np.sum(a_slow)
        
        return enf, ens, pf, ps

    def tcrit_DC(self, tcrit, taus, areas, comp):
        """
        DC criterion: equal percentage misclassified for fast and slow components.
        
        Parameters:
        ----------
        tcrit : float
            The critical time being tested.
        taus : np.array
            Array of time constants (taus).
        areas : np.array
            Array of areas corresponding to the exponential components.
        comp : int
            Index indicating the separation between fast and slow components.
        
        Returns:
        -------
        float
            The difference between the fraction misclassified for fast and slow components.
        """
        _, _, pf, ps = self.misclassified(tcrit, taus, areas, comp)
        return ps - pf

    def tcrit_CN(self, tcrit, taus, areas, comp):
        """
        Clapham & Neher criterion: equal number of misclassified events.
        
        Parameters:
        ----------
        tcrit : float
            The critical time being tested.
        taus : np.array
            Array of time constants (taus).
        areas : np.array
            Array of areas corresponding to the exponential components.
        comp : int
            Index indicating the separation between fast and slow components.
        
        Returns:
        -------
        float
            The difference between the number of misclassified events for fast and slow components.
        """
        enf, ens, _, _ = self.misclassified(tcrit, taus, areas, comp)
        return ens - enf

    def tcrit_Jackson(self, tcrit, taus, areas, comp):
        """
        Jackson et al criterion: minimize the total number of misclassified events.
        
        Parameters:
        ----------
        tcrit : float
            The critical time being tested.
        taus : np.array
            Array of time constants (taus).
        areas : np.array
            Array of areas corresponding to the exponential components.
        comp : int
            Index indicating the separation between fast and slow components.
        
        Returns:
        -------
        float
            The total number of misclassified events for fast and slow components.
        """
        t_fast, t_slow = taus[:comp], taus[comp:]
        a_fast, a_slow = areas[:comp], areas[comp:]
        
        # Number of misclassified events per component, weighted by time constants
        enf = np.sum((a_fast / t_fast) * np.exp(-tcrit / t_fast))
        ens = np.sum((a_slow / t_slow) * np.exp(-tcrit / t_slow))
        
        return enf - ens

    def print_critical_times_summary(self):
        """
        Print the critical times and misclassified data for all components.
        """
        num_components = self.num_components
        criteria_names = ['Equal % misclassified (DC criterion)', 
                        'Equal # misclassified (Clapham & Neher criterion)', 
                        'Minimum total # misclassified (Jackson et al criterion)']

        # Header for the summary
        summary_header = "\nSUMMARY of tcrit values:\nComponents  DC      C&N     Jackson\n"
        summary_rows = []
        summary_str = ''

        for i in range(num_components):
            summary_str += f"\nCritical time between components {i+1} and {i+2}\n"
            
            for crit_idx, criterion_name in enumerate(criteria_names):
                tcrit = self.tcrits[crit_idx, i]
                enf, ens, pf, ps = self.misclassified_results[crit_idx, i, :]
                
                # Printing the detailed results for each criterion
                summary_str += f"\n{criterion_name}"
                if tcrit is not None:
                    summary_str += f"\ntcrit = {tcrit:.5f} ms"
                    summary_str += f"\n% misclassified: short = {pf*100:.5f}; long = {ps*100:.5f}"
                    summary_str += f"\n# misclassified (out of 100): short = {enf:.5f}; long = {ens:.5f}"
                    summary_str += f"\nTotal # misclassified (out of 100) = {pf*100 + ps*100:.5f}\n"
                else:
                    summary_str += "\ntcrit could not be determined\n"
            
            # Store the tcrit values for the summary table
            summary_rows.append(f"\n{i+1} to {i+2}  {self.tcrits[0, i]:.5f}  {self.tcrits[1, i]:.5f}  {self.tcrits[2, i]:.5f}")

        # Print the summary table
        summary_str += summary_header
        summary_str += "\n".join(summary_rows)

        return summary_str


    def misclassified_printout(self, tcrit, enf, ens, pf, ps):
        """
        """

        return ('tcrit = {0:.5g} ms\n'.format(tcrit * 1000) +
            '% misclassified: short = {0:.5g};'.format(pf * 100) +
            ' long = {0:.5g}\n'.format(ps * 100) +
            '# misclassified (out of 100): short = {0:.5g};'.format(enf * 100) +
            ' long = {0:.5g}\n'.format(ens * 100) +
            'Total # misclassified (out of 100) = {0:.5g}\n\n'
            .format((enf + ens) * 100))


    def printout_tcrit(self):
        """
        Output calculations based on division into bursts by critical time (tcrit).

        Parameters
        ----------
        mec : dcpyps.Mechanism
            The mechanism to be analysed.
        output : output device
            Default device: sys.stdout
        """
          
        tcrit_str = ('\nSUMMARY of tcrit values:\n' +
            'Components  DC\tC&N\tJackson\n')
        for i in range(self.num_components):
            tcrit_str += ('{0:d} to {1:d} '.format(i+1, i+2) +
                '\t{0:.5g}'.format(self.tcrits[0, i] * 1000) +
                '\t{0:.5g}'.format(self.tcrits[1, i] * 1000) +
                '\t{0:.5g}\n'.format(self.tcrits[2, i] * 1000))
                
        return tcrit_str


class ExpPDF:
    def __init__(self, taus, areas):
        self.taus = taus
        self.areas = areas
        self.num_components = self.taus.shape[0]
        self.mean = np.sum(self.areas * self.taus)
        self.variance = np.sum(self.areas * self.taus * self.taus)
        self.SD = np.sqrt(2 * self.variance - self.mean * self.mean)

    def expPDF(self, t): #, tau, area):
        """ Calculate exponential probability density function. """
        if isinstance(self.taus, np.ndarray):
            f = np.zeros(t.shape)
            for i in range(self.taus.shape[0]):
                f += (self.areas[i] / self.taus[i]) * np.exp(-t / self.taus[i])
        else:
            f = (self.areas / self.taus) * np.exp(-t / self.taus)
        return f

    def printout(self, title, print_mean=True): #eigs, amps, title):
        """ Print exponential PDF data.  """
        header = ['Term', 'Amplitude', 'Rate (1/sec)', 'tau (ms)', 'Area (%)']
        table = [ [i+1, self.areas[i] / self.taus[i], 1 / self.taus[i], 1000 * self.taus[i], 100 * self.areas[i]]
            for i in range(self.num_components) ]
        table_str = f'\n{title}\n' + tabulate( table, headers=header, tablefmt='orgtbl' )
        if not print_mean: return table_str
        mean_str = f'\nMean (ms) = {self.mean * 1000:.5g}\t\tSD = {self.SD * 1000:.5g}\t\tSD/mean = {self.SD / self.mean:.5g}\n'
        return table_str + mean_str

    def printout_asymptotic(self, tres, title):
        """
        """

        info_str = '\n'+title+ '\n'
        areast0 = self.areas * np.exp(tres / self.taus)
        areast0 /= np.sum(areast0)
        table = []
        for i in range(self.num_components):
            table.append([i+1, 1 / self.taus[i], 1000 * self.taus[i], 100 * self.areas[i], 100 * areast0[i]])
        info_str += tabulate(table, 
                                headers=['Term', 'Rate (1/sec)', 'tau (ms)', 'Area (%)', 'Area renormalised for t=0 to inf'], 
                                tablefmt='orgtbl')       
        return info_str



class GeometricPDF:
    def __init__(self, rho, w):
        self.rho = rho
        self.w = w
        self.k = self.rho.shape[0]
        self.ONE = np.ones(self.k)
        self.norm = 1 / (self.ONE - self.rho)
        self.mean = np.sum(self.w / np.power(self.ONE - self.rho, 2))
        self.variance = np.sum(self.w * (self.ONE + self.rho) / np.power(self.ONE - self.rho, 3))
        self.SD = np.sqrt(self.variance - self.mean * self.mean)

    def printout(self, title, print_mean=True): #rho, w, title):
        """ Print geometric PDF data. """
        
        header = ['Term', 'w', 'rho', 'area(%)', 'Norm mean']
        table = [
            [i+1, self.w[i], self.rho[i], 100 * self.w[i] * self.norm[i], self.norm[i]]
            for i in range(self.k) ]
        table_str = f'\n{title}\n' + tabulate(table, headers=header, tablefmt='orgtbl')
        if not print_mean: return table_str
        mean_str = f'\nMean number of openings per burst = {self.mean:.5g}\n\tSD = {self.SD:.5g}\tSD/mean = {self.SD / self.mean:.5g}\n'
        return table_str + mean_str


###############   DEPRECATED FUNCTIONS   ###########################

def geometricPDF_mean_sd(rho, w):
    """
    Calculate mean and standard deviation for geometric PDF.

    Parameters
    ----------
    rho : ndarray, shape(k, 1)
        Probabilities.
    w : ndarray, shape(k, 1)
        Component amplitudes.

    Returns
    -------
    m : float
        Mean.
    sd : float
        Standard deviation.
    """

    k = rho.shape[0]
    m = np.sum(w / np.power(np.ones((k)) - rho, 2))
    var = np.sum(w * (np.ones((k)) + rho) / np.power(np.ones((k)) - rho, 3))
    sd = math.sqrt(var - m * m)

    return m, sd

def geometricPDF_printout(rho, w):
    """
    """

    norm = 1 / (np.ones((rho.shape[0])) - rho)
    str = ('term\tw\trho\tarea(%)\tNorm mean')
    for i in range(rho.shape[0]):
        str += ('{0:d}'.format(i+1) +
            '\t{0:.5g}'.format(w[i]) +
            '\t{0:.5g}'.format(rho[i]) +
            '\t{0:.5g}'.format(w[i] * norm[i] * 100) +
            '\t{0:.5g}\n'.format(norm[i]))

    mean, sd = geometricPDF_mean_sd(rho, w)
    str += ('Mean number of openings per burst =\t {0:.5g}'.format(mean) +
        '\n\tSD =\t {0:.5g}'.format(sd) +
        '\tSD/mean =\t {0:.5g}\n'.format(sd / mean))
    return str

@deprecated("Use '...'")
def expPDF(t, tau, area):
    """
    Calculate exponential probabolity density function.

    Parameters
    ----------
    t : float or ndarray.
        Time.
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    f : float or ndarray.
    """

    #if type(tau) == type(np.array(())):
    if isinstance(tau, np.ndarray):
        f = np.zeros(t.shape)
        for i in range(tau.shape[0]):
            f += (area[i] / tau[i]) * np.exp(-t / tau[i])
        #f = np.sum((area / tau) * np.exp(-t[:, np.newaxis] / tau), axis=0)
    else:
        f = (area / tau) * np.exp(-t / tau)
    return f

@deprecated("Use '...'")
def expPDF_mean_sd(tau, area):
    """
    Calculate mean and standard deviation for exponential PDF.

    Parameters
    ----------
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    m : float
        Mean.
    sd : float
        Standard deviation.
    """

    m = np.sum(area * tau)
    var = np.sum(area * tau * tau)
    sd = math.sqrt(2 * var - m * m)

    return m, sd

@deprecated("Use '...'")
def expPDF_printout(eigs, ampl):
    """
    """

    str = ('term\tw\trate (1/sec)\ttau (ms)\tarea (%)\n')
    for i in range(eigs.shape[0]):
        str += ('{0:d}'.format(i+1) +
            '\t{0:.5g}'.format(ampl[i]) +
            '\t{0:.5g}'.format(eigs[i]) +
            '\t{0:.5g}'.format(1000 / eigs[i]) +
            '\t{0:.5g}\n'.format(100 * ampl[i] / eigs[i]))

    mean, sd = expPDF_mean_sd(1 / eigs, ampl / eigs)
    str += ('Mean (ms) =\t {0:.5g}'.format(mean * 1000) +
        '\tSD =\t {0:.5g}'.format(sd * 1000) +
        '\tSD/mean =\t {0:.5g}\n'.format(sd / mean))
        
    return str

@deprecated("Use '...'")
def expPDF_misclassified(tcrit, tau, area, comp):
    """
    Calculate number and fraction of misclassified events after division into
    bursts by critical time, tcrit.
    """

    tfast = tau[:comp]
    tslow = tau[comp:]
    afast = area[:comp]
    aslow = area[comp:]

    # Number of misclassified.
    enf = np.sum(afast * np.exp(-tcrit / tfast))
    ens = np.sum(aslow * (1 - np.exp(-tcrit / tslow)))

    # Fraction misclassified.
    pf = enf / np.sum(afast)
    ps = ens / np.sum(aslow)

    return enf, ens, pf, ps

@deprecated("Use '...'")
def expPDF_misclassified_printout(tcrit, enf, ens, pf, ps):
    """
    """

    return ('tcrit = {0:.5g} ms\n'.format(tcrit * 1000) +
        '% misclassified: short = {0:.5g};'.format(pf * 100) +
        ' long = {0:.5g}\n'.format(ps * 100) +
        '# misclassified (out of 100): short = {0:.5g};'.format(enf * 100) +
        ' long = {0:.5g}\n'.format(ens * 100) +
        'Total # misclassified (out of 100) = {0:.5g}\n\n'
        .format((enf + ens) * 100))

@deprecated("Use '...'")
def expPDF_tcrit_DC(tcrit, tau, area, comp):
    """
    """

    enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, comp)
    return ps - pf

@deprecated("Use '...'")
def expPDF_tcrit_CN(tcrit, tau, area, comp):
    """
    """

    enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, comp)
    return ens - enf

@deprecated("Use '...'")
def expPDF_tcrit_Jackson(tcrit, tau, area, comp):
    """
    """

    tfast = tau[:comp]
    tslow = tau[comp:]
    afast = area[:comp]
    aslow = area[comp:]

    # Number of misclassified.
    enf = np.sum((afast / tfast) * np.exp(-tcrit / tfast))
    ens = np.sum((aslow / tslow) * np.exp(-tcrit / tslow))

    return enf - ens

