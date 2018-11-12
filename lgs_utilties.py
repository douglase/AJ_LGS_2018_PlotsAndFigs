"""
Utilities for calculating space laser guide star error budgets


"""

import astropy.units as u
import astropy.table as tabl
import astropy.constants as c

import numpy as np
import matplotlib.pyplot as plt
zero_mag_Vband = 9.1e9*u.photon/u.m**2/u.second #all the photons in Bessel V
zero_mag_zband =  4810*u.Jy*1.51e7*u.photon*u.second**(-1)*u.m**(-2)*(0.13/0.91)**(-1)/u.Jy
#https://www.astro.umd.edu/~ssm/ASTR620/mags.html

def quant_hist(param,**kwargs):
    plt.hist(param,normed=True,bins="auto",linewidth=.1,**kwargs)
    plt.xlabel(param.unit)
    plt.ylabel("Probability Density")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


    
def parse_table(pt,n=10000):
    """
    parse an astropy table into a list of kwargs.



    Example 
    ------
    
        pt=tabl.QTable()
        avg=0
        std=1
        pt["name"]=["mean","stdev"]
        pt["tx_laser_power"]=[10,.5]*u.milliwatt
        pt["tx_wavel"]=[980,10]*u.nm
        wl = pt["tx_wavel"][0]
        
        n=10000
        pt["tx_rx_separation"]=np.array([3e5,1000])*u.km
        pt["tx_divergence"]=[15,1]*u.arcsec
        table_link=lgs.link(**parse_table(pt,n=10000),magfilter="MagAO")


    """
    kwargs={}
    for col in pt.colnames:
        if col=="name":
            continue
        if pt[col][1] == 0:
            kwargs.update({col:pt[col][0]})
        else:
            #print(np.random.normal(pt[col][0].value,scale=pt[col][1].value,size=n),pt[col][0],pt[col][1],pt[col].unit,n)
            vals=np.random.normal(pt[col][0].value,scale=pt[col][1].value,size=n)*pt[col].unit
            kwargs.update({col:vals})
        
    return kwargs

class link:
    def __init__(self,
                tx_laser_power=np.random.normal(1,scale=5*0.1,size=1000)*u.watt,
                tx_wavel=980*u.nm,
                tx_aperture=85*u.mm,
                tx_rx_separation=np.random.normal(37000,scale=100,size=1000)*u.km,
                tx_jitter= 10*u.arcsec,
                tx_divergence=None,
                tx_fiber_MFD=5*u.um,
                rx_radius=9.8/2*u.m,
                rx_hexagon=True,
                rx_throughput=1,
                attenuation=None,
                tophat=False,
                magfilter="V",
                 force_band=None,
                 rx_exp_time=1.5*u.minute #stahl et al
                ):
        """
        Parameters:
                tx_laser_power=np.random.normal(1,scale=5*0.1,size=1000)*u.watt,
                tx_wavel=980*u.nm,
                tx_aperture=85*u.mm,
                tx_rx_separation=np.random.normal(37000,scale=100,size=1000)*u.km,
                tx_jitter= 10*u.arcsec,
                tx_divergence=None,#Half-angle divergence
                tx_fiber_MFD=5*u.um,
                rx_radius=8.4*u.m,
                rx_hexagon=True,
                rx_throughput=1,
                attenuation=None,
                tophat=False,
                magfilter="V",
                force_band=None, if not None, will attempt to force calculations with band set to given value.
        """
        print("initing")
        self.tx_laser_power = tx_laser_power
        self.tx_wavel = tx_wavel
        self.tx_aperture = tx_aperture
        self.R = tx_rx_separation
        self.tx_jitter = tx_jitter
        self.z_w0 = 0
        self.z=tx_rx_separation
        self.wavelength = tx_wavel
        self.tx_divergence = tx_divergence 
        if self.tx_divergence is None:
            raise ValueError("not implemented")
            #  to do, calculated from MFD and f.l.:
            #self.w_0 = tx_fiber_MFD
        else:
            self.w_0 = self.wavelength/(np.pi*self.tx_divergence.to(u.radian).value)
        if attenuation is not None:
            print("no detailed atmospheric transmission model implemented, assuming single float")
            self.attenuation = attenuation
        else:
            self.attenuation=1
        self.tx_rx_separation =tx_rx_separation 
        self.rx_radius  = rx_radius
        self.rx_hexagon = rx_hexagon
        self.tophat=tophat
        self.magfilter=magfilter
        self.rx_throughput=rx_throughput
        self.force_band=force_band
        
    @property
    def rx_power(self):
        '''
        '''
        rx = (self.tx_laser_power/(4*np.pi*self.R**2)*self.rx_Area).to(u.watt)
        return rx
    @property
    def z_r(self):
        """
        Rayleigh distance for the gaussian beam, based on
        current beam waist and wavelength.

        I.e. the distance along the propagation direction from the
        beam waist at which the area of the cross section has doubled.
        The depth of focus is conventionally twice this distance.
        """

        return np.pi * self.w_0 ** 2 / self.wavelength

    @property
    def divergence(self):
        """
        half angle divergence of the gaussian beam

        I.e. the angle between the optical axis and the beam radius at a large distance.
        Angle in radians.
        """
        if tx_divergence is not None:
            return self.wavelength / (np.pi * self.w_0)
        else:
            return self.tx_divergence

    def r_c(self, z=None):
        """
        The gaussian beam radius of curvature as a function of distance z

        Parameters
        -------------
        z : float, optional
            Distance along the optical axis.
            If not specified, the wavefront's current z coordinate will
            be used, returning the beam radius of curvature at the current position.

        Returns
        -------
        Astropy.units.Quantity of dimension length

        """
        if z is None:
            z = self.z
        dz = (z - self.z_w0)  # z relative to waist
        if dz.max() == 0:
            return np.inf * u.m
        return dz * (1 + (self.z_r / dz) ** 2)
    @property
    def I(self):
        """
        beam intensity distribution across aperture
        """
        return 
    def spot_radius(self, z=None):
        """
        radius of a propagating gaussian wavefront, at a distance z

        Parameters
        -------------
        z : float, optional
            Distance along the optical axis.
            If not specified, the wavefront's current z coordinate will
            be used, returning the beam radius at the current position.

        Returns
        -------
        Astropy.units.Quantity of dimension length
        """
        if z is None:
            z = self.z
        return self.w_0 * np.sqrt(1.0 + ((z - self.z_w0) / self.z_r) ** 2)
        
    @property
    def E_phot(self):
        return c.c*c.h/self.wavelength
    @property
    def tx_phot_sec(self):
        """
        corresponds to "Req. Photon Rate [photons/s]" in Table 5 of Marlow et al.
        
        """
        
        return self.tx_laser_power/self.E_phot*u.photon*self.attenuation
    @property
    def rx_phot_sec(self):
        """
        """
        
        return self.incident_phot_sec*self.rx_throughput
    @property
    def incident_phot_sec(self):
        """
        integrate the incident gaussian to the edge of the 
        telescope aperture
        
        if Tophat=True all energy is assumed to fall within 
        a tophat of radius equal to the spot radius.
        """
        y = self.rx_radius#*u.radian
         #sigma as defined in Douglas et al. 2015, DOI: 10.1109/LGRS.2014.2361812, is approx. same as spot radius
        #sigma = (self.tx_divergence*self.tx_rx_separation).decompose()
        eff_factor = 1-np.exp(-y**2/(2*self.spot_radius()**2))
        #print("gaussian eff.:"+str(eff_factor))
    
        if self.tophat:
            #not physical
            eff_factor = (y**2/(self.spot_radius()**2)).decompose().value
            print("tophat eff.:"+str(eff_factor))
        if self.rx_hexagon:
            eff_factor = eff_factor *3*np.sqrt(3)/(2*np.pi) #hexagon has ~82.7% area of circumscribed circle
        return (self.tx_phot_sec).to(u.photon/u.second)*eff_factor
    
    #  methods supporting coordinates,
    #including switching between distance and angular units
    @property
    def band(self, fudge=False):
        """
        Return the filter identifier, given the wavelength of the tx beam
        """
        if self.force_band is not None:
            return self.force_band
        if( self.magfilter=="MagAO")&(self.wavelength.mean() >600*u.nm) & \
                (self.wavelength.mean() <1000*u.nm):
            return "MAG_WFS"
        elif (self.wavelength.mean() >(5448 - 840/2.)*u.angstrom) & \
                (self.wavelength.mean() <(5448 + 840/2.)*u.angstrom):
            return "V"
        elif (self.wavelength.mean() >(6410 - 1500/2.)*u.angstrom) & \
                (self.wavelength.mean() <(6410 + 1500/2.)*u.angstrom):
            return "R"       
        elif (self.wavelength.mean() >(9100 - 1248/2.)*u.angstrom) & \
                 (self.wavelength.mean() <(9100 + 1248/2.)*u.angstrom):
            return "z"
        #elif (self.wavelength.mean() >600*u.nm) & (self.wavelength.mean() <1000*u.nm):
        #    return "MAG_WFS" #catchall
        else:
            print("wavelength (" +str(self.wavelength.mean()) +") has no specified filter")
            return "unknown band"
        
    @property
    def rx_Area(self):
        return (np.pi*self.rx_radius**2)
    @property
    def magnitude(self):
        print(self.band)
        #if self.band=="AB":
        #    fv=3631*u.jansky*self.rx_Area
        #    flux=self.rx_phot_sec*c.h/self.wavelength.mean()/u.photon
        #    print(flux,fv)
        #    return (-2.5*np.log10((flux/fv)))
        if self.band=="V":
            zero_mag= zero_mag_Vband 
            stellar_photons = zero_mag*self.rx_Area
            #print("stellar photons", stellar_photons.decompose())
            return (-2.5*np.log10((self.incident_phot_sec/stellar_photons)))
        if self.band=="z":
            zero_mag= zero_mag_zband
            stellar_photons = zero_mag*self.rx_Area
            #print("stellar photons", stellar_photons.decompose())
            return (-2.5*np.log10((self.incident_phot_sec/stellar_photons)))
        if self.band=="R":
            #http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
            zero_mag = 702*u.photon/u.cm**2/u.second/u.angstrom*150*u.nm
            stellar_photons = zero_mag*self.rx_Area
            #print("stellar photons", stellar_photons.decompose())
            return (-2.5*np.log10((self.incident_phot_sec/stellar_photons)))
        if self.band=="z\'":
            #http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
            zero_mag = 602*u.photon/u.cm**2/u.second/u.angstrom*124.8*u.nm
            stellar_photons = zero_mag*self.rx_Area
            return (-2.5*np.log10((self.incident_phot_sec/stellar_photons)))

        if self.band=="MAG_WFS":
            zero_mag = 5e9*u.photon/u.m**2/u.second #marlow et al p.19
            stellar_photons = zero_mag*self.rx_Area 
            #print("stellar photons", stellar_photons.decompose())
            return (-2.5*np.log10((self.incident_phot_sec/stellar_photons)))

