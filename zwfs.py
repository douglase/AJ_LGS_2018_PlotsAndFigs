"""


"""

import poppy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from poppy.poppy_core import PlaneType

class CircularPhaseShift(poppy.AnalyticOpticalElement):

    """ Defines an ideal circular pupil aperture
    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the pupil, in meters. Default is 1.0
    shift: float
        Shift in radians.
    
    pad_factor : float, optional
        Amount to oversize the wavefront array relative to this pupil.
        This is in practice not very useful, but it provides a straightforward way
        of verifying during code testing that the amount of padding (or size of the circle)
        does not make any numerical difference in the final result.
    """

    @poppy.utils.quantity_input(radius=u.meter)
    def __init__(self, name=None, 
                 radius=1.0*u.meter,
                 shift=0.25,
                 pad_factor=1.0,
                 outside=0,
                 planetype=poppy.poppy_core.PlaneType.unspecified, **kwargs):

        if name is None:
            name = "Circle, radius={}".format(radius)
        super(CircularPhaseShift, self).__init__(name=name, planetype=planetype, **kwargs)
        self.radius=radius
        self.shift=shift
        self.outside=outside
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius

    def get_opd(self, wave, units='meters'):
        """ Compute the transmission inside/outside of the aperture.
        """
        if not isinstance(wave, poppy.Wavefront):  # pragma: no cover
            raise ValueError(" get_opd must be called with a Wavefront "
                             "to define the spacing")
        y, x = self.get_coordinates(wave)
        radius = self.radius.to(u.meter).value
        r = np.sqrt(x ** 2 + y ** 2)
        del x
        del y

        w_outside = np.where(r > radius)
        del r
        self.opd = (self.shift*wave.wavelength*np.ones(wave.shape)).to(u.m).value
        self.opd[w_outside] = self.outside
        return self.opd
    
def get_intensity(wf):
    return poppy.utils.removePadding(wf.intensity,oversample=wf.oversample)

def get_amplitude(wf):
    return poppy.utils.removePadding(wf.amplitude,oversample=wf.oversample)


def phi(I_C, b=0.5, P=1):
    """
     solution to N'Diaye et al 2013 eq. 8, if \theta=pi/2 
    
    $I_C=P^2 +2b+2Pb +2Pb(-\sqrt{2}(-\phi+\pi/4))$

    $\rightarrow \phi=-\sin^{-1}\left[-\frac{I_c-P^2-2b^2}{-2\sqrt{2}Pb}\right] +\frac{\pi}{4}$
    """
    return (-1)*np.arcsin((I_C - P**2 - 2*b**2)/(-2*np.sqrt(2)*b*P)) + np.pi/4.

def phi14(I_C, b=0.5):
    """
     solution to N'Diaye et al 2013 eq. 14
    
    """
    return -1+np.sqrt(3-2*b-(1-I_C/b))


def phi12(I_C, b=0.5, P=1):
    """
     solution to N'Diaye et al 2013 eq. 12, if \theta=pi/2 
    

=    """
    return -1+np.sqrt(3*P*b-2*b**2-(P**2-I_C)/(P*b))


