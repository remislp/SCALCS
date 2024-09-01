from samples import samples
from scalcs.qmatprint import QMatrixPrints, SCBurstPrints
from scalcs.scburst import SCBurst


print('\n\tNUMERICAL EXAMPLE CH82')
# Load Colquhoun & Hawkes 1982 numerical example
c = 0.0000001 # 0.1 uM
mec = samples.CH82()
mec.set_eff('c', c)
print(mec)

# Create an instances of the QMatrix, QOccupancies and QTransitions classes
q_matrix = QMatrixPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
q_matrix.print_Q()
q_matrix.print_pinf() # print equilibrium state occupancies
q_matrix.print_Popen()
q_matrix.print_state_lifetimes()
q_matrix.print_transition_matrices()
q_matrix.print_subset_probabilities()
q_matrix.print_initial_vectors()
q_matrix.print_DC_table()

# Calculating burst parameters
q_burst = SCBurstPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
q_burst.print_means()
q_burst.print_length_pdf()
q_burst.print_openings_pdf()
q_burst.print_shuttings_pdf()
