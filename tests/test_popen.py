import numpy.testing as npt

from scalcs import popen
from samples import samples
from scalcs import scalcslib as scl
from scalcs import scplotlib as scpl


def test_regression():
    mec = samples.CH82()
    tres = 0.0001 # 100 microsec
    # POPEN CURVE CALCULATIONS
    _, _, pi = scpl.Popen(mec, tres)
    assert (pi[-1]>0.967 and pi[-1]<0.969)

def test_import():
    from scalcs.popen import PopenCurve
    assert PopenCurve is not None

def test_PopenCurve_constructor():
    pc = popen.PopenCurve()
    assert pc.tres == 0.0
    assert pc.mec is None

def test_pilot_curve():
    
    mec = samples.CH82()
    tres = 0.0
    pc = popen.PopenCurve(mec, tres)
    
    assert len(pc._pilot_conc) == 1000
    assert len(pc._pilot_popen) == 1000
    
    npt.assert_almost_equal(pc.maxPopen, 0.9677419329868882, decimal=16)
    npt.assert_almost_equal(pc.EC50, 2.403813848843297e-06, decimal=16)
    npt.assert_almost_equal(pc.nH, 1.8920149941304116, decimal=12)    # check if possible to get working with decimal=16


    assert pc.is_not_decreasing
    assert not pc.is_not_increasing
    assert pc.is_monotonic
    assert not pc.is_decreasing
    assert pc.is_increasing
