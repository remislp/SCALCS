"""This module contains functions for reading MEC, SCN, SSD, ABF files;
choosing and reading a kinetic mechanism from a mec file generated by DCPROGS.
"""

import os
import struct
import time
from array import array
import numpy as np
import pandas as pd

from scalcs import mechanism

def mec_get_list(mecfile):
    """
    Read list of mechanisms saved in mec file.

    Parameters
    ----------
    mecfile : filename

    Returns
    -------
    version : int
        Version of mec file.
    meclist : list; each element is another list containing
        jstart : int; start byte for mechanism in mefile.
        mecnum : int; mechanism sequence number in mecfile.
        mectitle : string
        ratetitle : string
    max_mecnum : int
        Number of different mechanisms in mec file.
    """

    f = open(mecfile, 'rb')
    ints = array('i')

    # Read version of mec file. Latest version is 102.
    ints.fromfile(f,1)
    version = ints.pop()

    # Read number of rate sets (records) stored in the file
    ints.fromfile(f,1)
    nrecs = ints.pop()

    # Read byte value for next record
    ints.fromfile(f,1)
    nextrec = ints.pop()

    # Read byte value where last record starts
    ints.fromfile(f,1)
    ireclast = ints.pop()

    # Read start byte value for storage of the ith record
    jstart = np.zeros(nrecs, 'int32')    # jstart()- start byte # for storage of the ith record (2000 bytes)
    for i in range(nrecs):
        ints.fromfile(f, 1)
        jstart[i] = ints.pop()

    meclist = []
    max_mecnum = 0
    for i in range(nrecs):
        f.seek(jstart[i] - 1 + 4)
        ints.fromfile(f,1)
        mecnum = ints.pop()
        if mecnum > max_mecnum:
            max_mecnum = mecnum
        mectitle = f.read(74).decode("utf-8")
        ints.fromfile(f,5)
        ratetitle = f.read(74).decode("utf-8")
        
        set = []
        set.append(jstart[i])
        set.append(mecnum)
        set.append(mectitle)
        set.append(ratetitle)
        meclist.append(set)

    f.close()
    return version, meclist, max_mecnum

def mec_choose_from_list(meclist, max_mecnum):
    """
    Choose mechanism from a list of mechanisms in file.

    Parameters
    ----------
    meclist : list; each element is another list containing:
        jstart : int; start byte for mechanism in mefile.
        mecnum : int; mechanism sequence number in mecfile.
        mectitle : string
        ratetitle : string
    max_mecnum : int
        Number of different mechanisms in mec file.

    Returns
    -------
    mecnum : int
        Sequence number of a mechanism to read.
    ratenum : int
        Sequence number of rate set to read.
    """

    # List all mechs and choose one.
    print (' Model #              title')
    ndisp = 0
    for i in range(1, (max_mecnum + 1)):
            present = False
            id = 0
            for j in range(len(meclist)):
                if i == meclist[j][1]:
                    present = True
                    id = j
            if present:
                print (i, meclist[id][2])
                ndisp += 1
                if ndisp % 20 == 0:
                    input('\n   Hit ENTER for more... \n')
    try:
        mecnum = int(input(
            "\nWhich mechanism would you like to read (1 to %d)? ... "
            %max_mecnum))
    except:
        print ("\nError: model number not entered!")
        mecnum = max_mecnum

    # List and choose rate constants.
    print (
        "\nFor model %d the following rate constants have been stored:"
        %mecnum)

    ndisp = 0
    for i in range(len(meclist)):
       if meclist[i][1] == mecnum:
           print ((i+1), meclist[i][3])
           ndisp += 1
           if ndisp % 20 == 0:
               input("\n   Hit ENTER for more... \n")
    try:
        ratenum = (int(input(
            "\nWhich rate set would you like to read?... ")) - 1)
    except:
        print ("Error: rate set number not entered!")

    if (ratenum < 0) or (ratenum > len(meclist)):
        print ("Error: not valid rate set number!")

    return mecnum, ratenum

def mec_load(mecfile, start):
    """
    Load chosen mec.

    Parameters
    ----------
    mecfile : filename
    start : int
        Start byte in mecfile for mechanism to read.

    Returns
    -------
    mec.Mechanism(RateList, StateList, ncyc) : instance of Mechanism class.
    """

    # Make dummy arrays to read floats, integers and short integers.
    doubles = array('d')
    floats = array ('f')
    ints = array('i')

    f=open(mecfile, 'rb')	# open the .mec file as read only
    f.seek(start - 1);
    ints.fromfile(f, 1)
    version1 = ints.pop()
    ints.fromfile(f, 1)
    mecnum = ints.pop()
    mectitle = f.read(74);

    # Read number of states.
    ints.fromfile(f,1)
    k = ints.pop()
    ints.fromfile(f,1)
    kA = ints.pop()
    ints.fromfile(f,1)
    kB = ints.pop()
    ints.fromfile(f,1)
    kC = ints.pop()
    ints.fromfile(f,1)
    kD = ints.pop()

    # In mec files version=103 all shut states are of type 'C'.
    # Check and leave just one in state 'C', others go as 'B'.
    if kB == 0:
        kB = kC - 1
        kC = 1

    ratetitle = f.read(74)

    # Read size of chess board to draw mechanism.
    ints.fromfile(f,1)
    ilast = ints.pop()
    ints.fromfile(f,1)
    jlast= ints.pop()

    # nrateq- number of non-zero rates in Q; = 2*ncon (always)
    ints.fromfile(f,1)
    nrateq = ints.pop()

    # Number of connections.
    ints.fromfile(f,1)
    ncon = ints.pop()

    # Number of concentration dependent rates
    ints.fromfile(f,1)
    ncdep = ints.pop()

    # Number of ligands
    ints.fromfile(f,1)
    nlig = ints.pop()

    # ? if char mechanism is presnt
    ints.fromfile(f,1)
    chardef = ints.pop()

    # ???
    ints.fromfile(f,1)
    boundef = ints.pop()

    # Number of cycles.
    ints.fromfile(f,1)
    ncyc = ints.pop()

    # Voltage.
    floats.fromfile(f,1)
    vref = floats.pop()
#    print 'vref=', vref

    # Number of voltage dependent rates.
    ints.fromfile(f,1)
    nvdep = ints.pop()

    # ???
    ints.fromfile(f,1)
    kmfast = ints.pop()

    # Independent subunit model.
    # False for all old models (npar=nrateq=2*ncon)
    # True when npar < nrateq=2*ncon. In this case must have nsetq>0
    ints.fromfile(f,1)
    indmod = ints.pop()

    # Number of basic rates constants.
    # Normally npar=nrateq and nsetq=0, but when indmod=T then npar<nrateq.
    ints.fromfile(f,1)
    npar = ints.pop()

    # ???
    ints.fromfile(f,1)
    nsetq = ints.pop()

    # ???
    ints.fromfile(f,1)
    kstat = ints.pop()

    # Output of mechanism in characters
    # TODO clean characters
    Jch = []
    for j in range(0, jlast):
        Ich = []
        for i in range(0, ilast):  # 500 is max
             charmod = f.read(2)#.decode("utf-8")
             Ich.append(charmod)
        Jch.append(Ich)
    for i in range(0,ilast):
        IIch = []
        for j in range(0, jlast):
            IIch.append(Jch[j][i])
        #print (''.join(IIch))

    # Read rate constants.
    irate = []
    for i in range(nrateq):
        ints.fromfile(f,1)
        irate.append(ints.pop())
    jrate = []
    for i in range(nrateq):
        ints.fromfile(f,1)
        jrate.append(ints.pop())
    QT = np.zeros((k, k), 'float64')
    for i in range(nrateq):
        doubles.fromfile(f, 1)
        QT[irate[i]-1, jrate[i]-1] = doubles.pop()
    ratename = []
    for i in range(npar):
        ratename.append(f.read(10).decode("utf-8"))
        #print ratename[i], "QT[",irate[i],",",jrate[i],"]=", QT[irate[i]-1,jrate[i]-1]

    # Read ligand name and ligand molecules bound in each state.
    for j in range(0, nlig):
        ligname = f.read(20).decode("utf-8")
        #print "Number of ligand %s molecules bound to states:" %ligname
    nbound = np.zeros((nlig,k), 'int32')
    for i in range(nlig):
        for j in range(k):
            ints.fromfile(f, 1)
            nbound[i,j] = ints.pop()
        #print "to state",j+1,":",nbound[i,j]

    # Read concentration dependent rates.
    # from state
    ix = []
    for i in range(0, ncdep):
        ints.fromfile(f,1)
        ix.append(ints.pop())
    # to state
    jx = []
    for j in range(0, ncdep):
        ints.fromfile(f,1)
        jx.append(ints.pop())
        #if verbose: print "jx[",j,"]=",jx[j]
    # ligand bound in that particular transition
    il = []
    for i in range(0, ncdep):
        ints.fromfile(f,1)
        il.append(ints.pop())
        #if verbose: print "il[", i, "]=", il[i]

    # Read open state conductance.
    dgamma = []
    for j in range(0, kA):
        doubles.fromfile(f,1)
        dgamma.append(doubles.pop())
#    print 'dgamma=', dgamma

    # Get number of states in each cycle and connections.
    nsc = np.zeros(50, 'int32')
    for i in range(0, ncyc):
        ints.fromfile(f,1)
        nsc[i] = ints.pop()
    #print "nsc[", i, "]=", nsc[i]
    im = np.zeros((50, 100), 'int32')
    for i in range(0, ncyc):
        for j in range(0, nsc[i]):
            ints.fromfile(f,1)
            im[i, j] = ints.pop()
            #print "im[",i,",",j,"]=",im[i,j]
    jm = np.zeros((50,100), 'int32')
    for i in range(0, ncyc):
        for j in range(0, nsc[i]):
            ints.fromfile(f,1)
            jm[i,j] = ints.pop()
            #print "jm[",i,",",j,"]=",jm[i,j]

    # Read voltage dependent rates.
    # from state
    iv = []
    for i in range(0, nvdep):
        ints.fromfile(f,1)
        iv.append(ints.pop())
        #print "iv[",i,"]=",iv[i]
    # to state
    jv = []
    for j in range(0, nvdep):
        ints.fromfile(f,1)
        jv.append(ints.pop())
        #print "jv[", j,"]=",jv[j]

    hpar = []
    for i in range(0, nvdep):
        floats.fromfile(f,1)
        hpar.append(floats.pop())
        #print "hpar[",i,"]=",hpar[i]

    pstar = []
    for j in range(0, 4):
        floats.fromfile(f,1)
        pstar.append(floats.pop())
        #print "pstar[",j,"]=",pstar[j]

    kmcon = []
    for i in range(0, 9):
        ints.fromfile(f,1)
        kmcon.append(ints.pop())
        #print "kmcon[",i,"]=",kmcon[i]

    ieq = []
    for i in range(0, nsetq):
        ints.fromfile(f,1)
        ieq.append(ints.pop())
        #print "ieq[",i,"]=",ieq[i]

    jeq = []
    for j in range(0, nsetq):
        ints.fromfile(f,1)
        jeq.append(ints.pop())
        #print "jeq[", j, "]=", jeq[j]

    ifq = []
    for i in range(0,nsetq):
        ints.fromfile(f,1)
        ifq.append(ints.pop())
        #print "ifq[",i,"]=",ifq[i]

    jfq = []
    for j in range(0, nsetq):
        ints.fromfile(f,1)
        jfq.append(ints.pop())
        #print "jfq[",j,"]=",jfq[j]

    efacq = []
    for i in range(0, nsetq):
        floats.fromfile(f,1)
        efacq.append(floats.pop())
        #print "efacq[",i,"]=",efacq[i]

    statenames = []
    for i in range(0, kstat):
        statename = f.read(10).decode("utf-8")
        statenames.append(statename.split()[0])
        #print "State name:", statename
    #print ("\n")

    ints.fromfile(f,1)
    nsub = ints.pop()
    ints.fromfile(f,1)
    kstat0 = ints.pop()
    ints.fromfile(f,1)
    npar0 = ints.pop()
    ints.fromfile(f,1)
    kcon = ints.pop()
    ints.fromfile(f,1)
    npar1 = ints.pop()
    ints.fromfile(f,1)
    ncyc0 = ints.pop()

    f.close()

    StateList = []
    j = 0
    for i in range(kA):
        StateList.append(mechanism.State('A', statenames[j], dgamma[j]))
        j += 1
    for i in range(kB):
        StateList.append(mechanism.State('B', statenames[j], 0))
        j += 1
    for i in range(kC):
        StateList.append(mechanism.State('C', statenames[j], 0))
        j += 1
    for i in range(kD):
        StateList.append(mechanism.State('D', statenames[j], 0))
        j += 1

    RateList = []
    for i in range(nrateq):
        cdep = False
        bound = None
        for j in range(ncdep):
            if ix[j] == irate[i] and jx[j] == jrate[i]:
                cdep = True
                bound = 'c'
        rate = QT[irate[i] - 1, jrate[i] - 1]
        # REMIS: please make sure the state indexing is correct
        RateList.append(mechanism.Rate(rate, StateList[irate[i]-1],
            StateList[jrate[i]-1], name=ratename[i], eff=bound))

    CycleList = []
    for i in range(ncyc):
#        mrconstrained = False
        CycleStates = []
        for j in range(nsc[i]):
            CycleStates.append(statenames[im[i, j]-1])
        CycleList.append(mechanism.Cycle(CycleStates))

    return mechanism.Mechanism(RateList, CycleList,
        mtitle=mectitle, rtitle=ratetitle)

def load_from_excel_sheet(filename, sheet=0, verbose=False):
    """
    Load mechanism from Excel file.
    Returns
    -------
    mec.Mechanism(RateList, StateList, ncyc) : instance of Mechanism class.
    """

    #TODO: implement constrain reading and setting
    #TODO: check microscopic reversibility 
    df = pd.read_excel(filename, sheet_name=sheet, index_col=None, header=None)

    mectitle = df.loc[df.iloc[:, 0] == "mectitle"].iloc[0][1]
    if verbose: print(mectitle)
    ratetitle = df.loc[df.iloc[:, 0] == "ratetitle"].iloc[0][1]
    if verbose: print(ratetitle)

    df_states = df.loc[df.iloc[:, 0] == "state"]
    states = {}
    for index, row in df_states.iterrows():
        states[row[1]] = mechanism.State(row[2], row[1], row[3])
        if verbose:
            print('found state:', row[1], row[2], row[3])

    df_cycles = df.loc[df.iloc[:, 0] == "cycle"]
    cycles = []
    for index, row in df_cycles.iterrows():
        cycle = []
        for i in range(2, int(row[1])+2):
            cycle.append(row[i])
        if verbose:
            print('found cycle:', cycle)
        cycles.append(mechanism.Cycle(cycle))

    df_rates = df.loc[df.iloc[:, 0] == "rate"]
    rates = []
    bound, cfunc, cargs = None, None, None
    fixed, mr, constrained = False, False, False
    for index, row in df_rates.iterrows():
        if verbose: print(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        if row[5] is 'c': bound = 'c'
        if row[8] is 'mr': 
            mr = True
            cycles[row[9]-1].mrconstr = [row[2], row[3]]
        if row[8] is 'fixed': fixed = True
#        if row[8] is 'multiply':
#            cfunc = mechanism.multiply

        if verbose: print(row[4], row[2], row[3], row[1], bound, row[6], row[7])
        rates.append(mechanism.Rate(row[4], states[row[2]], states[row[3]], name=row[1], 
                                    eff=bound, fixed=fixed, mr=mr, limits=[row[6], row[7]],
                                    is_constrained=constrained, constrain_func=cfunc, constrain_args=cargs))

    return mechanism.Mechanism(rates, cycles, mtitle=mectitle, rtitle=ratetitle)

    
def load_mec_from_prt(filename, verbose=False):    
    f = open(filename, 'r')
    linenum = 0
    
    
    ncyc, nsc, im2, jm2, mr = 0, [], [], [], []
    states, conductance, ligbound = [], [], []
    tres, tcrit, CHS = [], [], []
    rates, constrained = [], []
    k, kA, kB, kC, kD = 0, 0, 0, 0, 0

    f.readline()
    line = f.readline()
    if "Program HJCFIT Windows Version (Beta)" in line:
        version = 'win'
    else:
        version = 'dos'
    if verbose: print ('version=', version)

    while True:
        try:
            line = f.readline()
            #print 'line:', line
            if line == '':
                break
            line = line.strip("\r\n")
            linenum += 1
        except EOFError:
            print('Mecanism loading from PRT (or TXT) file finished.')
            
        if "HJCFIT: Fit of model to open-shut times with missed events" in line:
            if verbose: print ('This is possibly HJCFIT printout file.')
            
        if "No of states in each subset: kA, kB, kC, kD =" in line:
            kA = int(line.split()[-4].strip())
            kB = int(line.split()[-3].strip())
            kC = int(line.split()[-2].strip())
            kD = int(line.split()[-1].strip())
            k = kA + kB + kC + kD
            if verbose: print ("Number of states: kA, kB, kC, kD = {0:d}, {1:d}, {2:d}, {3:d}".
                format(kA, kB, kC, kD))

        if version == 'win' and "Number of open states =" in line and k == 0:
            kA = int(line.split()[-1].strip())
            if verbose:
                print("Number of open states = ", kA)
            line = f.readline()
            while (line != "\n") and ("conductance of state" in line):
                conductance.append(float(line.split()[-1].strip()))
                line = f.readline()
            k = len(conductance)
            kB = k - kA - 1
            kC = 1
            kD = 0
            if verbose: print ("Number of states: kA, kB, kC, kD = {0:d}, {1:d}, {2:d}, {3:d}".
                format(kA, kB, kC, kD))
            
        if "Number of ligands =" in line:
            nlig = int(line.split()[-1].strip())
            if verbose: print ("Number of ligands = {0:d}".format(nlig))
            f.readline() #'Concentration-dependent elements:'
            f.readline() #'  i   j     ligand #   Ligand name'
            line = f.readline()
            im, jm, lig = [], [], []
            while line != "\n":
                temp = line.split()
                im.append(int(temp[0]))
                jm.append(int(temp[1]))
                lig.append(temp[-1])
                line = f.readline()
            if verbose: print ("im=", im)
            if verbose: print ("jm=", jm)
            if verbose: print ("lig=", lig)
                
        
        if (version == 'dos') and ("Cycle #" in line) and (int(line[-1]) > ncyc):
            c1, c2 = [], []
            ncyc += 1
            line = f.readline()
            temp = line.split()
            c1.append(int(temp[1].strip(',')))
            c2.append(int(temp[2].strip(')')))
            mr.append([int(temp[1].strip(',')), int(temp[2].strip(')'))])
            line = f.readline()
            temp = line.split()
            while temp:
                c1.append(int(temp.pop(0).strip('-')))
                c2.append(int(temp.pop(0)))
            nsc.append(len(c1))
            im2.append(c1)
            jm2.append(c2)
            if verbose: print ("Cycle # ", ncyc)
            if verbose: print (nsc)
            if verbose: print (im2, jm2)

        if (version == 'win') and ("Microscopic reversibility" in line):
            line = f.readline()
            while "cycle=" in line:
                c1, c2 = [], []
                ncyc += 1
                temp = line.strip(';').split()
                nsc.append(len(temp[3].strip()))
                temp = temp[7:]
                for item in temp:
                    c1.append(int(item.strip(';')))
                im2.append(c1)
                line = f.readline()

        if (version == 'win') and ("by microscopic reversibility" in line):
            if len(mr) < ncyc:
                temp = []
                temp.append(int(line[3:5].strip()))
                temp.append(int(line[6:8].strip()))
                mr.append(temp)
            
#        if "state #    state name" in line: # in 'dos'
#            line = f.readline()
#            while line != "\n":
#                temp = line.split()
#                states.append(temp[1])
#                line = f.readline()
#            if len(states) == k:
#                if verbose: print states
#            else:
#                print "Warning: number of states does not correspond."
                
        if "Number of ligands bound" in line:
            line = f.readline()
            line = f.readline()     
            while len(states) < k:
                temp = line.split()
                if len(temp) == 3:
                    states.append(temp[1])
                    ligbound.append(int(temp[2]))
                line = f.readline()
            if verbose: print ('states: ', states)
            if verbose: print ('ligandsbound=', ligbound)
                
        if "Resolution for HJC calculations" in line:
            line = f.readline()
            while line != "\n":
                temp = line.split()
                if len(temp) == 4:
                    tres.append(float(temp[2]) * 1e-6)
                line = f.readline()
            if verbose: print ('resolution: ', tres)
            
        if "The following parameters are constrained:" in line:
            line = f.readline()
            while '-----' not in line:
                cnstr = []
                temp = line.split()
                r1 = int(temp[1]) - 1
                line = f.readline()
                temp = line.split()
                fac = float(temp[0])
                op = temp[1]
                r2 = int(temp[3]) - 1
                cnstr.append(r1)
                cnstr.append(op)
                cnstr.append(fac)
                cnstr.append(r2)
                line = f.readline()
                constrained.append(cnstr)
#            print constrained
            
        if (version == 'dos') and ("initial        final" in line):
            line = f.readline()
            while line != "\n":
                rate = []
                if 'q(' in line:
                    rate.append(int(line[8:10].strip()))
                    rate.append(int(line[11:13].strip()))
                    rate.append(line[18:29].strip())
                    rate.append(float(line[42:55].strip()))
                    if len(line) > 55:
                        rate.append(line[55:].strip().strip('(').strip(')'))
                rates.append(rate)
                line = f.readline()
            if constrained:
                for item in constrained:
                    if len(rates[item[0]]) > 4: # and rates[item[0, 4]] == 'constrained':
                        rates[item[0]].append(item[1])
                        rates[item[0]].append(item[2])
                        rates[item[0]].append(item[3])
            if verbose: print (rates)
            
        if 'Total number of rates' in line:
            numrates = int(line.split()[-1])
            numfixed = int(f.readline().split()[-1])
            numconstr = int(f.readline().split()[-1])
            nummr = int(f.readline().split()[-1])
            numec50 = int(f.readline().split()[-1])
            numfree = int(f.readline().split()[-1])
            if verbose: print ('\nTotal number of rates = {0:d}'.format(numrates) +
                '\nNumber that are fixed       = {0:d}'.format(numfixed) +
                '\nNumber that are constrained = {0:d}'.format(numconstr) +
                '\nNumber set by micro rev     = {0:d}'.format(nummr) +
                '\nNumber set by fixed EC50    = {0:d}'.format(numec50) +
                '\nNumber of free rates to be estimated = {0:d}'.format(numfree))

        if (version == 'win') and ("initial        final" in line):
            while len(rates) < numrates:
                rate = []
                line = f.readline()
                while line != "\n":
                    rate.append(int(line[8:10].strip()))
                    rate.append(int(line[11:13].strip()))
                    rate.append(line[18:29].strip())
                    rate.append(float(line[42:55].strip()))
                    rates.append(rate)
                    line = f.readline()
                    if line != "\n":
                        rates[-1].append(line.strip().strip('(').strip(')'))
                        line = f.readline()

    if verbose: print ('file contains {0:d} lines'.format(linenum))
    f.close()
    
    StateList = []
    j = 0
    for i in range(kA):
        StateList.append(mechanism.State('A', states[j], 50))
        j += 1
    for i in range(kB):
        StateList.append(mechanism.State('B', states[j], 0))
        j += 1
    for i in range(kC):
        StateList.append(mechanism.State('C', states[j], 0))
        j += 1
    for i in range(kD):
        StateList.append(mechanism.State('D', states[j], 0))
        j += 1
        
    RateList = []
    for i in range(numrates):
        bound = None
        mrc = False
        is_constr = False
        constr_func = None
        constr_args = None
        for j in range(len(im)):
            if im[j] == rates[i][0] and jm[j] == rates[i][1]:
                bound = 'c'
        for j in range(ncyc):
            if mr[j][0] == rates[i][0] and mr[j][0] == rates[i][1]:
                mrc = True
        if len(rates[i]) > 5 and rates[i][4] == 'constrained':
            is_constr = True
            if rates[i][5] == 'times':
                constr_func = mechanism.constrain_rate_multiple
                constr_args = [rates[i][7], rates[i][6]]
        
        rate = rates[i][3]
        RateList.append(mechanism.Rate(rate, StateList[rates[i][0]-1],
            StateList[rates[i][1]-1], name=rates[i][2], eff=bound, mr=mrc,
            is_constrained=is_constr, constrain_func=constr_func, constrain_args=constr_args))
            
    CycleList = []
    for i in range(ncyc):
#        mrconstrained = False
        CycleStates = []
        for j in range(nsc[i]):
            CycleStates.append(states[im2[i][j]-1])
        CycleList.append(mechanism.Cycle(CycleStates))

    return mechanism.Mechanism(RateList, CycleList)

#def set_load_from_prt(filename, verbose=False):    
#    f = open(filename, 'r')
#    linenum = 0
#    
#    scnfiles = []
#    conc, tres, tcrit, chsvec = [], [], [], []
#    
#    f.readline()
#    line = f.readline()
#    if "Program HJCFIT Windows Version (Beta)" in line:
#        version = 'win'
#    else:
#        version = 'dos'
#    if verbose: print 'version=', version
#
#    while True:
#        try:
#            line = f.readline()
#            #print 'line:', line
#            if line == '':
#                break
#            line = line.strip("\r\n")
#            linenum += 1
#        except EOFError:
#            print('Data set loading from PRT (or TXT) file finished.')
#            
#    return scnfiles, conc, tres, tcrit, chsvec

def mod_load(file):
    """
    Load mechanism from Channel Lab .mod file.

    Parameters
    ----------
    file : filename

    Returns
    -------
    mec.Mechanism(RateList, StateList, ncyc) : instance of Mechanism class.
    """

    # TODO: get cycles from mod file.
    f = open(file, 'r')	# open the .mec file as read only
    cl = f.readline().strip("\n")
    #print cl
    modtitle = f.readline().strip("\n")
    #print modtitle

    while True:
        try:
            line = f.readline()
            #print 'line:', line
            if line == '':
                break
            line = line.strip("\r\n")
        except EOFError:
            print('MOD reading finished.')

        if line == '[State1-20: labels]':
            statelabels = []
            for i in range(20):
                statelabels.append(f.readline().strip("\r\n"))
#            print statelabels

        if line == '[State1-20: onoff]':
            stateonoff = np.empty((20, 4))
            for i in range(20):
                onoff = f.readline().strip("\r\n")
                values = onoff.split(' ')
                stateonoff[i, 0] = int(values[0])
                stateonoff[i, 1] = int(values[1])
                stateonoff[i, 2] = float(values[2])
                stateonoff[i, 3] = float(values[3])
#            print stateonoff

        if line == '[Drug dependence labels]':
            druglabels = []
            for i in range(6):
                druglabels.append(f.readline().strip("\r\n"))
#            print druglabels

        if line == '[K1-31on]':
            kon = np.empty(31)
            for i in range(31):
                kon[i] = float(f.readline().strip("\r\n"))
#            print kon

        if line == '[K1-31off]':
            koff = np.empty(31)
            for i in range(31):
                koff[i] = float(f.readline().strip("\r\n"))
#            print koff

        if line == '[Kon concentration/voltage dependent]':
            konconc = np.empty((31, 7))
            for i in range(31):
                onconc = f.readline().strip("\r\n")
                values = onconc.split(' ')
                for j in range(7):
                    konconc[i, j] = int(values[j])
#            print konconc

        if line == '[Koff concentration/voltage dependent]':
            koffconc = np.empty((31, 7))
            for i in range(31):
                offconc = f.readline().strip("\r\n")
                values = offconc.split(' ')
                for j in range(7):
                    koffconc[i, j] = int(values[j])
#            print koffconc

        # [Rate constants for transitions between all states]
        # [Constraints for rate constants for transitions]
        # [Loop Constraints for rate constants for transitions]

        # [Kon steepness of voltage dependence]
        # [Kon activation range for voltage dependence]
        # [Koff steepness of voltage dependence]
        # [Koff activation range for voltage dependence]
        # [QMatrixMonteCarlo,#passes,#pts,#start,#channels,openstate,startstate,pApS,mVmM]
        # [extra storage space]
        # [adinterval,conductance,voltage,vrev,valence,kzero]
        # [Stimulus: Mode,AutoStep,ManualStep,Exp,Fc,pole,autofilt]
        # [Analysis: pksrch,pkave,basepts,basepos,startpos,risebeg,riseend]
        # [Analysis: expmodel,parse,numfitpts,numiter,numrestarts,tol,dual,startpts]
        # [Analysis: Fit limits hi,lo for vertex 1-72]
        # [Analysis: Fixed fit params for vertex 1-72]
        # [Analysis: Seeds for vertex 1-72, fitlims,autoseed,fixfree]
        # [FitWaveforms: start, stop positions for fit]
        # [Multi-stimulus file voltage / concentration parameters]
        # 20 [TState1-20: labels]

    f.close()

    StateList = []
    statesA = []
    statesB = []
    statesC = []
    for i in range(stateonoff.shape[0]):
        if stateonoff[i, 0] == 1:
            if stateonoff[i, 1] == 1:
                statesA.append(i)
            elif stateonoff[i, 3] == 1:
                statesC.append(i)
            else:
                statesB.append(i)
    newstates = []
    newstates.extend(statesA)
    newstates.extend(statesB)
    newstates.extend(statesC)
    k = len(newstates)

    for i in range(k):
            if stateonoff[newstates[i], 1] == 1:
                StateList.append(mechanism.State('A',
                    statelabels[newstates[i]], stateonoff[newstates[i], 2]))
            elif stateonoff[newstates[i], 3] == 1:
                StateList.append(mechanism.State('C',
                    statelabels[newstates[i]], 0))
            else:
                StateList.append(mechanism.State('B',
                    statelabels[newstates[i]], 0))

    RateList = []
    for i in range(k):
        li = (newstates[i]+1) // 5
        ci = (newstates[i]+1) % 5

        bound = None
        if newstates[i] != max(newstates):
            if (newstates[i] > 14) and (newstates[i] < 19):
                # States 16-19.
                ri = 9 * li + ci - 1
                if (kon[ri] != 0) or (koff[ri] != 0):
                    if konconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri1],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+1)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+1)),
                        eff=bound))
                    if koffconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri1],
                        StateList[newstates.index(newstates[i]+1)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+1))+str(i),
                        eff=bound))

            elif ci == 0:
                # States 5, 10, 15
                ri = 9 * li + ci + 4 - 1
                if (kon[ri] != 0) or (koff[ri] != 0):
                    if konconc[ri, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+5)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+5)),
                        eff=bound))
                    if koffconc[ri, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri],
                        StateList[newstates.index(newstates[i]+5)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+5))+str(i),
                        eff=bound))

            else:
                # States 1-4, 6-9, 11-14
                ri1 = 9 * li + ci - 1
                ri2 = 9 * li + ci + 4 - 1

                if (kon[ri1] != 0) or (koff[ri1] != 0):
                    if konconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri1],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+1)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+1)),
                        eff=bound))
                    if koffconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri1],
                        StateList[newstates.index(newstates[i]+1)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+1))+str(i),
                        eff=bound))

                if (((kon[ri2] != 0) or (koff[ri2] != 0)) and
                    ((newstates[i]+5) <= max(newstates))):
                    if konconc[ri2, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri2],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+5)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+5)),
                        eff=bound))
                    if koffconc[ri2, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri2],
                        StateList[newstates.index(newstates[i]+5)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+5))+str(i),
                        eff=bound))

    return mechanism.Mechanism(RateList, mtitle=modtitle), modtitle

def mec_save_to_yaml(mec, fname):
    import yaml
    stream = open(fname, 'w')
    yaml.dump(mec, stream)
