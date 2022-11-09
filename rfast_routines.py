import lblabc_input
import math
import os
import numpy    as     np
from   scipy    import interpolate
from astropy.io import ascii
from rfast_opac_routines import opacities_read
from rfast_opac_routines import cia_read
from rfast_opac_routines import rayleigh
#
#
# flux adding routine for non-emitting, inhomogeneous, scattering atmosphere
#
# inputs:
#
#     dtau    - layer extinction optical depth [Nlam,Nlay]
#        g    - layer asymmetry parameter [Nlam,Nlay]
#    omega    - layer single scattering albedo [Nlam,Nlay]
#       As    - surface albedo
#
# optional:
#
#    r,t,a    - return layer reflectivity, transmissivity, and/or absorptivity if layp = True [Nlam,Nlay]
#
# outputs:
#
#       Ag    - planetary geometric albedo [Nlam]
#
def flxadd(dtau,g,omega,As,layp=-1):

# determine number of atmospheric layers, wavelength points
  Nlay = dtau.shape[1]
  Nlam = dtau.shape[0]

# initialize variables
  r    = np.zeros([Nlam,Nlay])
  t    = np.zeros([Nlam,Nlay])
  Ru   = np.zeros([Nlam,Nlay+1])

# special case for single-scattering albedo of unity
  ic = np.where(omega == 1)
  if ic[0].size !=0:
    r[ic] = 3/4*np.divide(np.multiply(1-g[ic],dtau[ic]),1 + np.multiply(3/4*(1-g[ic]),dtau[ic]))
    t[ic] = np.divide(1,1 + np.multiply(3/4*(1-g[ic]),dtau[ic]))

# more general case
  ic = np.where(omega != 1)
  if (ic[0].size !=0):

#   intermediate quantities for computing layer radiative properties
    a     = np.zeros([Nlam,Nlay])
    b     = np.zeros([Nlam,Nlay])
    d     = np.zeros([Nlam,Nlay])
    Ainf  = np.zeros([Nlam,Nlay])
    a[ic] = np.sqrt(1-omega[ic])
    b[ic] = 3/2*np.sqrt((1-np.multiply(omega[ic],g[ic])))
    d     = a + 2/3*b # convenient definition
    id    = np.where(d != 0)
    if id[0].size != 0:
      Ainf[id] = np.divide(2/3*b[id] - a[id],d[id])
    d     = 1 - np.multiply(np.exp(-2*np.multiply(np.multiply(a[ic],b[ic]),dtau[ic])),np.power(Ainf[ic],2)) # another convenient definition
    r[ic] = np.divide(np.multiply(Ainf[ic],(1 - np.exp(-2*np.multiply(np.multiply(a[ic],b[ic]),dtau[ic])))),d)
    t[ic] = np.divide(np.multiply(1 - np.power(Ainf[ic],2),np.exp(-np.multiply(np.multiply(a[ic],b[ic]),dtau[ic]))),d)

# lower boundary condition
  Ru[:,Nlay] = As

# add reflectivity upwards from surface
  for i in range(0,Nlay):
    j = Nlay - 1 - i
    Ru[:,j] = r[:,j] + t[:,j]**2*Ru[:,j+1]/(1 - r[:,j]*Ru[:,j+1])

# planetary albedo
  Ag = Ru[:,0]

# return additional quantities, if requested
  if layp != -1:
    if layp:
      return Ag, r, t, a

  return Ag
#
#
# flux adding routine for emitting, inhomogeneous, scattering atmosphere; w/o solar sources
#
# inputs:
#
#     dtau    - layer extinction optical depth [Nlam,Nlay]
#        g    - layer asymmetry parameter [Nlam,Nlay]
#    omega    - layer single scattering albedo [Nlam,Nlay]
#      lam    - wavelength grid (um) [Nlam]
#        T    - temperature profile (K) [Nlev]
#       Ts    - surface temperature (K)
#       em    - surface emissivity
#
# optional:
#
#    r,t,a    - return layer reflectivity, transmissivity, and/or absorptivity if layp = True [Nlam,Nlay]
#
# outputs:
#
#      Fu     - upwelling specific flux at toa (W m**-2 um**-2)
#
def flxadd_em(dtau,g,omega,lam,T,Ts,em,layp=-1):

# small dtau to prevent divide by zero
  small_tau = 1.e-7

# determine number of atmospheric layers, wavelength points
  Nlay = dtau.shape[1]
  Nlam = dtau.shape[0]
  Nlev = Nlay + 1

# initialize variables
  r     = np.zeros([Nlam,Nlay])
  t     = np.zeros([Nlam,Nlay])
  a     = np.zeros([Nlam,Nlay])
  su    = np.zeros([Nlam,Nlay])
  sd    = np.zeros([Nlam,Nlay])
  id    = np.zeros([Nlam,Nlay])
  dBlam = np.zeros([Nlam,Nlay])
  Rd    = np.zeros([Nlam,Nlev])
  Sd    = np.zeros([Nlam,Nlev])
  Ru    = np.zeros([Nlam,Nlev])
  Su    = np.zeros([Nlam,Nlev])

# identitity matrix
  id[:,:] = 1.

# special case for single-scattering albedo of unity
  ic = np.where(omega == 1)
  if ic[0].size !=0:
    r[ic] = 3/4*np.divide(np.multiply(1-g[ic],dtau[ic]),1 + np.multiply(3/4*(1-g[ic]),dtau[ic]))
    t[ic] = np.divide(1,1 + np.multiply(3/4*(1-g[ic]),dtau[ic]))

# more general case
  ic = np.where(omega != 1)
  if (ic[0].size !=0):

#   intermediate quantities for computing layer radiative properties
    f     = np.zeros([Nlam,Nlay])
    b     = np.zeros([Nlam,Nlay])
    d     = np.zeros([Nlam,Nlay])
    Ainf  = np.zeros([Nlam,Nlay])
    f[ic] = np.sqrt(1-omega[ic])
    b[ic] = 3/2*np.sqrt((1-np.multiply(omega[ic],g[ic])))
    d     = f + 2/3*b # convenient definition
    id0   = np.where(d != 0)
    if id0[0].size != 0:
      Ainf[id0] = np.divide(2/3*b[id0] - f[id0],d[id0])
    d     = 1 - np.multiply(np.exp(-2*np.multiply(np.multiply(f[ic],b[ic]),dtau[ic])),np.power(Ainf[ic],2)) # another convenient definition
    r[ic] = np.divide(np.multiply(Ainf[ic],(1 - np.exp(-2*np.multiply(np.multiply(f[ic],b[ic]),dtau[ic])))),d)
    t[ic] = np.divide(np.multiply(1 - np.power(Ainf[ic],2),np.exp(-np.multiply(np.multiply(f[ic],b[ic]),dtau[ic]))),d)

# layer absorptivity
  a   = id - r - t
  ia0 = np.where(a <= id - np.exp(-small_tau))
  if ia0[0].size != 0:
    a[ia0] = 1 - np.exp(-small_tau)
  ia0 = np.where(a >= 1)
  if ia0[0].size != 0:
    a[ia0] = np.exp(-small_tau)

# level Planck function (W m**-2 um**-1 sr**-1) [Nlam,Nlev]
  Blam = planck2D(lam,T)

# difference across layer in Planck function
  dBlam[:,:] = Blam[:,1:] - Blam[:,:-1]

# repeated term in source def'n
  corr = np.multiply(dBlam,id-a) + np.multiply(a,np.divide(dBlam,np.log(id-a)))

# source terms
  su   = np.pi*(np.multiply(a,Blam[:,0:Nlay])   - corr)
  sd   = np.pi*(np.multiply(a,Blam[:,1:Nlay+1]) + corr)

# upper boundary condition: no downwelling ir flux at TOA
  Rd[:,0] = 0.
  Sd[:,0] = 0.

# add reflectivity and source terms downwards
  for j in range(1,Nlev):
    Rd[:,j] = r[:,j-1]  + t[:,j-1]**2*Rd[:,j-1]/(1 - r[:,j-1]*Rd[:,j-1])
    Sd[:,j] = sd[:,j-1] + t[:,j-1]*(Sd[:,j-1] + su[:,j-1]*Rd[:,j-1])/(1 - r[:,j-1]*Rd[:,j-1])

# lower boundary condition
  Ru[:,Nlev-1] = 1 - em
  Su[:,Nlev-1] = np.pi*em*planck(lam,Ts)

# add reflectivity and source terms upwards
  for i in range(1,Nlev):
    j       = Nlev - i - 1
    Ru[:,j] = r[:,j]  + t[:,j]**2*Ru[:,j+1]/(1 - r[:,j]*Ru[:,j+1])
    Su[:,j] = su[:,j] + t[:,j]*(Su[:,j+1] + sd[:,j]*Ru[:,j+1])/(1 - r[:,j]*Ru[:,j+1])

# upwelling flux at top of atmosphere
  Fu = (Su[:,0] + Ru[:,0]*Sd[:,0])/(1 - Ru[:,0]*Rd[:,0])

# return additional quantities, if requested
  if layp != -1:
    if layp:
      return Fu, r, t, a

# code here would give up/down fluxes at each level
#    Fu = np.zeros([Nlam,Nlev])
#    Fd = np.zeros([Nlam,Nlev])
#    for i in range(0,Nlev):
#      Fu[:,i] = (Su[:,i] + Ru[:,i]*Sd[:,i])/(1 - Ru[:,i]*Rd[:,i])
#      Fd[:,i] = (Sd[:,i] + Rd[:,i]*Su[:,i])/(1 - Ru[:,i]*Rd[:,i])

  return Fu
#
#
# flux adding routine for inhomogeneous, scattering atmosphere with treatment for direct beam
#
# inputs:
#
#     dtau    - layer extinction optical depth [Nlam,Nlay]
#        g    - layer asymmetry parameter [Nlam,Nlay]
#    omega    - layer single scattering albedo [Nlam,Nlay]
# dtau_ray    - layer rayleigh scattering optical depth [Nlam,Nlay]
# dtau_cld    - layer cloud extinction optical depth [Nlam,Nlay]
#       gc    - packaged 1st, 2nd, and 3rd cloud scattering moments, each [Nlam,Nlay]
#     phfc    - 0 -> henyey-greenstein | 1-> two-term hg
#       As    - surface albedo
#    alpha    - phase angle (deg)
#   threeD    - contains points/weights for gauss-tchebyshev integration
#
# optional:
#
#    r,t,a    - return layer reflectivity, transmissivity, and/or absorptivity if layp = True [Nlam,Nlay]
#
# outputs:
#
#      Ap     - planetary reflectivity (geometric albedo x phase function)
#
def flxadd_3d(dtau,g,omega,dtau_ray,dtau_cld,gc,phfc,As,alpha,threeD,layp=-1):

# small dtau to prevent divide by zero
  small_tau = 1.e-7

# unpack cloud moments
  gc1,gc2,gc3 = gc

# determine number of atmospheric layers, wavelength points
  Nlay = dtau.shape[1]
  Nlam = dtau.shape[0]
  Nlev = Nlay + 1

# unpack gauss and tchebyshev points and weights
  thetaG,thetaT,wG,wT = threeD

# phase angle in radians
  ar = alpha*np.pi/180.

# diffusivity factor (3/2 for underlying derivation)
#  D  = 1.5

# initialize variables
  r     = np.zeros([Nlam,Nlay])
  t     = np.zeros([Nlam,Nlay])
  a     = np.zeros([Nlam,Nlay])
  su    = np.zeros([Nlam,Nlay])
  sd    = np.zeros([Nlam,Nlay])
  id    = np.zeros([Nlam,Nlay])
  dBlam = np.zeros([Nlam,Nlay])
  Rd    = np.zeros([Nlam,Nlev])
  Sd    = np.zeros([Nlam,Nlev])
  Ru    = np.zeros([Nlam,Nlev])
  Su    = np.zeros([Nlam,Nlev])

# identitity matrix
  id[:,:] = 1.

# special case for single-scattering albedo of unity
  ic = np.where(omega == 1)
  if ic[0].size !=0:
    r[ic] = 3/4*np.divide(np.multiply(1-g[ic],dtau[ic]),1 + np.multiply(3/4*(1-g[ic]),dtau[ic]))
    t[ic] = np.divide(1,1 + np.multiply(3/4*(1-g[ic]),dtau[ic]))

# more general case
  ic = np.where(omega != 1)
  if (ic[0].size !=0):

#   intermediate quantities for computing layer radiative properties
    f     = np.zeros([Nlam,Nlay])
    b     = np.zeros([Nlam,Nlay])
    d     = np.zeros([Nlam,Nlay])
    Ainf  = np.zeros([Nlam,Nlay])
    f[ic] = np.sqrt(1-omega[ic])
    b[ic] = 3/2*np.sqrt((1-np.multiply(omega[ic],g[ic])))
    d     = f + 2/3*b # convenient definition
    id0   = np.where(d != 0)
    if id0[0].size != 0:
      Ainf[id0] = np.divide(2/3*b[id0] - f[id0],d[id0])
    d     = 1 - np.multiply(np.exp(-2*np.multiply(np.multiply(f[ic],b[ic]),dtau[ic])),np.power(Ainf[ic],2)) # another convenient definition
    r[ic] = np.divide(np.multiply(Ainf[ic],(1 - np.exp(-2*np.multiply(np.multiply(f[ic],b[ic]),dtau[ic])))),d)
    t[ic] = np.divide(np.multiply(1 - np.power(Ainf[ic],2),np.exp(-np.multiply(np.multiply(f[ic],b[ic]),dtau[ic]))),d)

# layer absorptivity
  a   = id - r - t
  ia0 = np.where(a == 0)
  if ia0[0].size != 0:
    a[ia0] = 1 - np.exp(-small_tau)

# fraction of scattering in each layer due to Rayleigh [Nlam x Nlay]
  fray      = np.zeros([Nlam,Nlay])
  fray[:,:] = 1.
  ic0       = np.where(dtau_cld > small_tau)
  cld       = False
  if (ic0[0].size != 0):
    cld  = True
    fray = np.divide(dtau_ray,np.multiply(omega,dtau))

# scattering angle (trivial here -- general logic commented out below)
  cosTh = np.cos(np.pi - ar)

# integrated optical depths
  tau        = np.zeros([Nlam,Nlev])
  tau[:,1:]  = np.cumsum(dtau,axis=1)
  taum       = 0.5*(tau[:,1:] + tau[:,:-1]) # integrated optical depth to mid-levels [Nlam x Nlay]
  ftau       = dtau/2*np.exp(-dtau)         # direct beam correction term

# pixel geometry-insensitive cloud scattering terms
  if cld:
    if (phfc == 0): # single-term henyey-greenstein
      pc = pHG(gc1,cosTh)
    if (phfc == 1): # improved two-term henyey-greenstein; zhang & li (2016) JQSRT 184:40
      # defn following Eqn 6 of Z&L2016; note: g = gc1; h = gc2; l = gc3
      w   = np.power(np.multiply(gc1,gc2)-gc3,2) - np.multiply(4*(gc2-np.power(gc1,2)),np.multiply(gc1,gc3)-np.power(gc2,2))
      iw0 = np.where(w < 0)
      if (iw0[0].size != 0):
        w[iw0] = 1.e-5
      de  = 2*(gc2-np.power(gc1,2))
      g1  = np.divide(gc3 - np.multiply(gc1,gc2) + np.power(w,0.5),de) # eqn 7a
      g2  = np.divide(gc3 - np.multiply(gc1,gc2) - np.power(w,0.5),de) # eqn 7b
      al  = 0.5*id + np.divide(3*np.multiply(gc1,gc2)-2*np.power(gc1,3)-gc3,2*np.power(w,0.5)) # eqn 7c
      iw0 = np.where(w < 1.e-4) # avoiding divide by zero in extreme forward scattering case
      if (iw0[0].size != 0):
        g1[iw0] = gc1[iw0]
        g2[iw0] = 0.
        al[iw0] = 1.
      ip0 = np.where(np.abs(g2) > np.abs(g1))
      if (ip0[0].size != 0):
        g2[ip0] = -g1[ip0]
      pc = np.multiply(al,pHG(g1,cosTh)) + np.multiply(id-al,pHG(g2,cosTh))

# loop over gauss and tchebyshev points
  F0  = 1.             # normal incidence flux
  Ap  = np.zeros(Nlam) # planetary reflectivity
  for i in range(len(thetaG)):
    nu     = 0.5*(thetaG[i] - (np.cos(ar)-1)/(np.cos(ar)+1))*(np.cos(ar)+1) # H&L Eqn 9
    for j in range(len(thetaT)):

#     compute solar and observer zenith angles; horak & little (1965)
      mu0    = np.sin(np.arccos(thetaT[j]))*np.cos(np.arcsin(nu)-ar)        # solar incidence; H&L Eqn 1
      mu1    = np.sin(np.arccos(thetaT[j]))*np.cos(np.arcsin(nu))           # observer zenith; H&L Eqn 2

#     convoluted, unneeded geometry to get scattering angle
      #fac    = ((1-mu0**2)**0.5)*((1-mu1**2)**0.5)
      #if (fac > 0):
      #  cosphi = (mu0*mu1 - np.cos(ar))/fac                                 # H&L Eqn 3
      #  cosTh  = -(mu0*mu1 - fac*cosphi)                                    # scattering angle
      #else:
      #  cosTh  = 0.

#     direct beam treatment
      Idir0  = F0*np.exp(-(tau[:,0:-1]+ftau)/mu0)                                           # solar intensity that reaches a given layer
      dIdir  = np.multiply(fray,Idir0)*pray(cosTh)/(4*np.pi)                  # intensity rayleigh scattered to observer in each layer
      if cld:
        dIdir = dIdir + np.multiply(np.multiply(id-fray,pc),Idir0)/(4*np.pi)  # intensity cloud scattered to observer in each layer
      dFdi   = np.multiply(dIdir,np.multiply(omega,id-np.exp(-dtau)))         # layer upwelling flux going into direct beam
      dIdir  = np.multiply(dIdir,np.multiply(omega,id-np.exp(-dtau/mu1)))     # scale with single scattering albedo, layer optical depth

#     direct beam total scattered flux in each layer [Nlam x Nlay]
      dF = np.multiply(Idir0,np.multiply(omega,id-np.exp(-dtau)))

#     portions that enter diffuse up and down streams
      sd = np.multiply(dF,fray)/2
      if cld:
        if (phfc == 0): # single-term henyey-greenstein
          fdc = (pHG_int(gc1,(1-mu0**2)**0.5) + pHG_int(gc1,-(1-mu0**2)**0.5))/4
        if (phfc == 1): # improved two-term henyey-greenstein
          fr   = (pHG_int(g1,(1-mu0**2)**0.5) + pHG_int(g1,-(1-mu0**2)**0.5))/4
          bk   = (pHG_int(g2,(1-mu0**2)**0.5) + pHG_int(g2,-(1-mu0**2)**0.5))/4
          fdc  = np.multiply(al,fr) + np.multiply(id-al,bk)
        sd  = sd + np.multiply(fdc,np.multiply(dF,id-fray))
      su = dF - sd - dFdi # from conservation

#     upper boundary condition: no downwelling ir flux at TOA
      Rd[:,0] = 0.
      Sd[:,0] = 0.

#     add reflectivity and source terms downwards
      for k in range(1,Nlev):
        Rd[:,k] = r[:,k-1]  + t[:,k-1]**2*Rd[:,k-1]/(1 - r[:,k-1]*Rd[:,k-1])
        Sd[:,k] = sd[:,k-1] + t[:,k-1]*(Sd[:,k-1] + su[:,k-1]*Rd[:,k-1])/(1 - r[:,k-1]*Rd[:,k-1])

#     lower boundary condition
      Ru[:,Nlev-1] = As
      Su[:,Nlev-1] = As*mu0*F0*np.exp(-tau[:,-1]/mu0)

#     add reflectivity and source terms upwards
      for kk in range(1,Nlev):
        k       = Nlev - kk - 1
        Ru[:,k] = r[:,k]  + t[:,k]**2*Ru[:,k+1]/(1 - r[:,k]*Ru[:,k+1])
        Su[:,k] = su[:,k] + t[:,k]*(Su[:,k+1] + sd[:,k]*Ru[:,k+1])/(1 - r[:,k]*Ru[:,k+1])

#     attenuate direct beam as it exits atmosphere; sum intensities from each layer
      dIdir   = np.multiply(dIdir,np.exp(-(tau[:,0:-1]+ftau)/mu1))
      Idir    = np.sum(dIdir,axis=1)

#     upwelling flux at top of atmosphere
      Fu = (Su[:,0] + Ru[:,0]*Sd[:,0])/(1 - Ru[:,0]*Rd[:,0])

#     sum over quadrature points
      Ap    = Ap + (Idir + Fu/np.pi)*wG[i]*wT[j]

# return additional quantities, if requested
  if layp != -1:
    if layp:
      return Ap*(np.cos(ar) + 1), r, t, a

  return Ap*(np.cos(ar) + 1)
#
#
# planetary spectrum model
#
# inputs:
#
#      Nlev   - number of vertical levels to use in model
#        Rp   - planetary radius (R_Earth)
#         a   - orbital distance (au)
#        As   - grey surface albedo
#        em   - surface emissivity
#         p   - pressure profile (Pa)
#         t   - temperature profile (K)
#        t0   - "surface" temperature (K)
#         m   - atmospheric mean molecular weight (kg/molecule)
#         z   - altitude profile (m)
#      grav   - gravitational acceleration profile (m/s/s)
#        Ts   - host star temperature (K)
#        Rs   - host star radius (Rsun)
#       ray   - True -> do rayleigh scattering ; False -> no rayleigh scattering
#      ray0   - gas rayleigh cross sections at 0.4579 um (m**2/molec) (see set_gas_info)
#      rayb   - background gas rayleigh cross section relative to Ar
#         f   - vector of molecular mixing ratios; order: Ar, CH4, CO2, H2, H2O, He, N2, O2, O3
#        fb   - background gas mixing ratio
#       mmr   - if true, interprets mixing ratios as mass mixing ratios; vmrs otherwise
#        nu   - refractivity at STP (averaged over column)
#    threeD   - disk integration quantities (Gauss/Tchebyshev points and weights)
#     gasid   - names of radiatively active gases (see set_gas_info)
# species_l   - names of species to include as line absorbers; options: ch4, co2, h2o, o2, o3
# species_c   - names of species to include as cia; options: co2, h2, n2, o2
#       ref   - include refraction in transit case if true
#       cld   - True -> include cloud ; False -> no cloud
#       sct   - include forward scattering correction in transit case if true
#        fc   - cloud fractional coverage
#        pt   - top-of-cloud pressure (Pa)
#       dpc   - cloud thickness (Pa)
#      tauc   - cloud optical thickness (extinction)
#       src   - source type (diff -> diffuse reflectance; thrm -> thermal sources)
#        g0   - cloud asymmetry parameter (Nmom x len(lam))
#        w0   - cloud sungle scattering albedo (len(lam))
# sigma_interp- line absorber pressure interpolation function
#   cia_interp- cia coefficient temperature interpolation function
#       lam   - wavelength grid for output (um)
#
# options:
#
#        pf   - if set, line opacities are interpolated to this fixed pressure (Pa)
#        tf   - if set, cia coefficients are interpolated to this fixed temperature
#
# outputs:
#
#       Ap    - planetary albedo, akin to geometric albedo [Nlam]
#     FpFs    - planet-to-star flux ratio [Nlam]
#
def gen_spec(Nlev,Rp,a,As,em,p,t,t0,m,z,grav,Ts,Rs,ray,ray0,rayb,f,fb,
             mmw0,mmr,ref,nu,alpha,threeD,
             gasid,ncia,ciaid,species_l,species_c,
             cld,sct,phfc,fc,pt,dpc,g0,w0,tauc,
             src,sigma_interp,cia_interp,lam,pf=-1,tf=-1):

  # constants
  Re   = 6.378e6        # earth radius (m)
  Rsun = 6.957e8        # solar radius (m)
  au   = 1.496e11       # au (m)
  kB   = 1.38064852e-23 # m**2 kg s**-2 K**-1
  Na   = 6.0221408e23   # avogradro's number

  # small optical depth to prevent divide by zero
  small_tau = 1e-10

  # number of wavelength points
  Nlam = len(lam)

  # number of layers
  Nlay = Nlev - 1

  # midpoint grids and pressure change across each layer
  dp    = p[1:] - p[:-1]             # pressure difference across each layer
  pm    = 0.5*(p[1:] + p[:-1])       # mid-layer pressure
  tm    = 0.5*(t[1:] + t[:-1])       # mid-layer temperature
  nm    = pm/kB/tm                   # mid-layer number density, ideal gas law (m**-3)
  gravm = 0.5*(grav[1:] + grav[:-1]) # mid-layer gravity
  fm    = 0.5*(f[:,1:] + f[:,:-1])   # mid-layer mixing ratios
  fbm   = 0.5*(fb[1:] + fb[:-1])     # mid-layer background gas mixing ratios
  mm    = 0.5*(m[1:] + m[:-1])       # mid-layer mean molecular weight

  # layer column number density (molecules/m**2) or mass density (kg/m**2)
  if mmr:
    dMc = dp/gravm
  else:
    dNc = dp/gravm/mm

  # interpolate line opacities onto p/T grid, cannot be less than zero
  if ( np.any( pf == -1) and np.any( tf == -1) ): # varying p/t case
    sigma = np.power(10,sigma_interp(np.log10(pm),1/tm))
  elif ( np.any( pf != -1) and np.any( tf == -1) ): # fixed p case
    sigma = np.power(10,sigma_interp(1/tm))
  elif ( np.any( pf == -1) and np.any( tf != -1) ): # fixed t case
    sigma = np.power(10,sigma_interp(np.log10(pm)))
  else: # fixed p and t
    sigma0 = np.power(10,sigma_interp())
    sigma  = np.repeat(sigma0[:,np.newaxis,:], Nlay, axis=1)

  izero        = np.where(sigma < 0)
  sigma[izero] = 0

  # interpolate cia coefficients onto temperature grid, cannot be less than zero
  if ( np.any( tf == -1) ): # varying temp case
    kcia = cia_interp(1/tm)
  else:
    kcia0 = cia_interp(1/tf)
    kcia  = np.zeros([kcia0.shape[0],Nlay,Nlam])
    for isp in range(0,kcia0.shape[0]):
      kcia[isp,:,:] = np.repeat(kcia0[isp,np.newaxis,:], Nlay, axis=0)

  izero       = np.where(kcia < 0)
  kcia[izero] = 0

  # rayleigh scattering opacities
  if ray:
    if mmr:
      sigma_ray = rayleigh(lam,ray0/(mmw0/1.e3/Na),fm,fbm,rayb) # m**2/kg
    else:
      sigma_ray = rayleigh(lam,ray0,fm,fbm,rayb)                # m**2/molec
  else:
    sigma_ray = np.zeros([Nlay,Nlam])

  # line absorber opacity
  sigma_gas = np.zeros([Nlay,Nlam])
  for isp in range(0,len(species_l)):
    idg       = gasid.index(np.char.lower(species_l[isp]))
    if mmr:
      sigma_gas = sigma_gas + fm[idg,:,np.newaxis]*sigma[isp,:,:]/(mmw0[idg]/1.e3/Na) # m**2/kg
    else:
      sigma_gas = sigma_gas + fm[idg,:,np.newaxis]*sigma[isp,:,:]                     # m**2/molec

  # cia opacity (m**2/molec, or m**2/kg if mmr is true)
  sigma_cia = np.zeros([Nlay,Nlam])
  icia      = 0
  for isp in range(0,len(species_c)):
    idg1  = gasid.index(np.char.lower(species_c[isp]))
    mmw1  = mmw0[idg1]
    for ipar in range(0,ncia[isp]):
      # case where background is everything but absorber
      if (np.char.lower(ciaid[ipar+1,isp]) == 'x'):
        ff   = fm[idg1,:]*(1-fm[idg1,:])
        mmw2 = mm*Na*1e3
      # case where partner is a specified gas
      else:
        idg2 = gasid.index(np.char.lower(ciaid[ipar+1,isp]))
        ff   = fm[idg1,:]*fm[idg2,:]
        mmw2 = mmw0[idg2]/Na/1e3
      if mmr:
        fac       = mm/mmw1/mmw2 # factor so that sigma is in units of m**2/kg
        sigma_cia = sigma_cia + ff[:,np.newaxis]*np.transpose(np.multiply(np.transpose(kcia[icia,:,:]),nm*fac))
      else:
        sigma_cia = sigma_cia + ff[:,np.newaxis]*np.transpose(np.multiply(np.transpose(kcia[icia,:,:]),nm))
      icia = icia + 1

  # total opacity (m**2/molec, or m**2/kg if mmr is true)
  sigma_tot = sigma_gas + sigma_ray + sigma_cia

  # clearsky layer optical depth, single scattering albedo, and asymmetry parameter
  if mmr:
    dtau     = np.multiply(np.transpose(sigma_tot),dMc)
    dtau_ray = np.multiply(np.transpose(sigma_ray),dMc)
  else:
    dtau     = np.multiply(np.transpose(sigma_tot),dNc)
    dtau_ray = np.multiply(np.transpose(sigma_ray),dNc)
  g        = np.zeros([Nlam,Nlay]) # note that g=0 for Rayleigh scattering
  if (not ray):
    dtau_ray    = np.zeros([Nlam,Nlay])
    dtau_ray[:] = small_tau
    dtau        = dtau + dtau_ray
    omega       = np.zeros([Nlam,Nlay])
  else:
    omega    = np.divide(dtau_ray,dtau)

  # transit refractive floor correction (robinson et al. 2017)
  pref = max(p)
  if (src == 'trns' and ref):
    pref    = refract_floor(nu,t,Rs,a,Rp,m,grav)
  
  # call radiative transfer model
  if (src == 'diff' or src == 'cmbn'):
    Ap       = flxadd(dtau,g,omega,As)*2/3 # factor of 2/3 converts to geometric albedo, for Lambert case
  if (src == 'thrm' or src == 'scnd' or src == 'cmbn'):
    Flam     = flxadd_em(dtau,g,omega,lam,t,t0,em)
  if (src == 'trns'):
    td       = transit_depth(Rp,Rs,z,dtau,p,ref,pref)
  if (src == 'phas'):
    dtau_cld = np.zeros([Nlam,Nlay])
    gc1      = np.zeros([Nlam,Nlay]) # first moment
    gc2      = np.zeros([Nlam,Nlay]) # second moment
    gc3      = np.zeros([Nlam,Nlay]) # second moment
    gc       = gc1,gc2,gc3
    Ap       = flxadd_3d(dtau,g,omega,dtau_ray,dtau_cld,gc,phfc,As,alpha,threeD)

  # if doing clouds
  if cld:
    # unpack first, second and third moments of phase function
    g1,g2,g3       = g0

    # cloudy optical properties
    dtaug          = np.copy(dtau)
    dtau_cld       = np.zeros([Nlam,Nlay])
    ip             = np.argwhere( (p < pt+dpc) & (p >= pt) )
    #
    # logic for cloud uniformly distributed in pressure
    dtau_cld[:,ip[np.where(ip < Nlay)]] = dp[ip[np.where(ip < Nlay)]]/dpc
    # rough logic for exponentially-distributed cloud
#    logs           = np.logspace(np.log10(1.e-3*max(tauc)),np.log10(max(tauc)),len(ip))
#    logs           = logs/np.sum(logs)
#    dtau_cld[:,ip[:,0]] = logs
    #
    dtau_cld       = np.multiply(dtau_cld,np.repeat(tauc[:,np.newaxis], Nlay, axis=1))
    wc             = np.transpose(np.repeat(w0[np.newaxis,:], Nlay, axis=0))
    gc1            = np.transpose(np.repeat(g1[np.newaxis,:], Nlay, axis=0)) # first moment
    gc2            = np.transpose(np.repeat(g2[np.newaxis,:], Nlay, axis=0)) # second moment
    gc3            = np.transpose(np.repeat(g3[np.newaxis,:], Nlay, axis=0)) # second moment
    dtau_cld_s     = np.multiply(wc,dtau_cld)
    omega          = np.divide(dtau_ray + dtau_cld_s,dtau + dtau_cld)
    g              = np.divide(np.multiply(gc1,dtau_cld_s),dtau_ray + dtau_cld_s)
    dtau           = dtaug + dtau_cld
    gc             = gc1,gc2,gc3

    # transit forward scattering correction (robinson et al. 2017)
    if (src == 'trns' and sct):
      f       = np.zeros([Nlam,Nlay])
      ifor    = np.where(gc1 >= 1 - 0.1*(1-np.cos(Rs*Rsun/a/au)))
      f[ifor] = np.multiply(np.divide((1 - np.power(gc1[ifor],2)),2*gc1[ifor]),np.power(1-gc1[ifor],-1)-np.power(1+np.power(gc1[ifor],2)-2*gc1[ifor]*np.cos(Rs*Rsun/a/au),-0.5))
      dtau[ifor] = np.multiply(1-np.multiply(f[ifor],wc[ifor]),dtau_cld[ifor]) + dtaug[ifor]

    # call radiative transfer model
    if (src == 'diff' or src == 'cmbn'):
      Apc    = flxadd(dtau,g,omega,As)*2/3 # factor of 2/3 converts to geometric albedo, for Lambert case
    if (src == 'thrm' or src == 'scnd' or src == 'cmbn'):
      Flamc  = flxadd_em(dtau,g,omega,lam,t,t0,em)
    if (src == 'trns'):
      td      = transit_depth(Rp,Rs,z,dtau,p,ref,pref)
    if (src == 'phas'):
      Apc     = flxadd_3d(dtau,g,omega,dtau_ray,dtau_cld,gc,phfc,As,alpha,threeD)

    # weighted albedo or flux
    if (src == 'diff' or src == 'cmbn' or src == 'phas'):
      Ap   = (1-fc)*Ap + fc*Apc
    if (src == 'thrm' or src == 'scnd' or src == 'cmbn'):
      Flam = (1-fc)*Flam + fc*Flamc

  # planet-to-star flux ratio, brightness temp, or effective transit altitude
  if (src == 'diff' or src == 'phas'):
    FpFs = Ap*(Rp*Re/a/au)**2
  if (src == 'thrm' or src == 'scnd'):
    Tbrt = Tbright(lam,Flam)
  if (src == 'cmbn'):
    FpFs = Ap*(Rp*Re/a/au)**2 + Flam/(np.pi*planck(lam,Ts))*(Rp/Rs)**2*((Re/Rsun)**2)
  if (src == 'scnd'):
    FpFs = Flam/(np.pi*planck(lam,Ts))*(Rp/Rs)**2*((Re/Rsun)**2)
  if (src == 'trns'):
    zeff = Rs*Rsun*td**0.5 - Rp*Re

  # return quantities
  if (src == 'diff' or src == 'phas'):
    ret = Ap,FpFs
  if (src == 'thrm'):
    ret = Tbrt,Flam
  if (src == 'scnd'):
    ret = Tbrt,FpFs
  if (src == 'cmbn'):
    ret = Ap,FpFs
  if (src == 'trns'):
    ret = zeff,td

  return ret
#
#
# spectral grid routine, pieces together multiple wavelength
# regions with differing resolutions
#
# inputs:
#
#     res     - spectral resolving power (x/dx)
#     x_min   - minimum spectral cutoff
#     x_max   - maximumum spectral cutoff
#
# outputs:
#
#         x   - center of spectral gridpoints
#        Dx   - spectral element width
#
def gen_spec_grid(x_min,x_max,res,Nres=0):
  if ( len(x_min) == 1 ):
    x_sml = x_min/1e3
    x_low = max(x_sml,x_min - x_min/res*Nres)
    x_hgh = x_max + x_max/res*Nres
    x,Dx  = spectral_grid(x_low,x_hgh,res=res)
  else:
    x_sml = x_min[0]/1e3
    x_low = max(x_sml,x_min[0] - x_min[0]/res[0]*Nres)
    x_hgh = x_max[0] + x_max[0]/res[0]*Nres
    x,Dx  = spectral_grid(x_low,x_hgh,res=res[0])
    for i in range(1,len(x_min)):
      x_sml  = x_min[i]/1e3
      x_low  = max(x_sml,x_min[i] - x_min[i]/res[i]*Nres)
      x_hgh  = x_max[i] + x_max[i]/res[i]*Nres
      xi,Dxi = spectral_grid(x_low,x_hgh,res=res[i])
      x      = np.concatenate((x,xi)) 
      Dx     = np.concatenate((Dx,Dxi))
    Dx = [Dxs for _,Dxs in sorted(zip(x,Dx))]
    x  = np.sort(x)
  return np.squeeze(x),np.squeeze(Dx)
#
#
# routine to read in inputs from script
#
# inputs:
#
#    filename_scr  - filename containing input parameters and calues
#
#
# outputs:
#
#    (a long collection of all parameters needed to run models)
#
def inputs(filename_scr):

  # flags for planet mass and gravity inputs
  mf = False
  gf = False

  # read inputs
  with open(filename_scr) as f:
    for line in f:
      line  = line.partition('#')[0] # split line at '#' symbol
      line  = line.strip()          # trim whitespace
      vn    = line.partition('=')[0] # variable name
      vn    = vn.strip()
      vv    = line.partition('=')[2] # variable value
      vv    = vv.strip()
      if (vn.lower() == 'fnr' ):
        fnr  = vv
      elif (vn.lower() == 'fns' ):
        fns  = vv
      elif (vn.lower() == 'fnn' ):
        fnn  = vv
      elif (vn.lower() == 'dirout' ):
        dirout  = vv
      elif (vn.lower() == 'fnatm' ):
        fnatm = vv
      elif (vn.lower() == 'fntmp' ):
        fntmp = vv
      elif (vn.lower() == 'skpatm'):
        skpatm = int(vv)
      elif (vn.lower() == 'skptmp'):
        skptmp = int(vv)
      elif (vn.lower() == 'colpr'):
        colpr = int(vv)
      elif (vn.lower() == 'colpt'):
        colpt = int(vv)
      elif (vn.lower() == 'colt'):
        colt = int(vv)
      elif (vn.lower() == 'psclr'):
        psclr = float(vv)
      elif (vn.lower() == 'psclt'):
        psclt = float(vv)
      elif (vn.lower() == 'imix'):
        imix = int(vv)
      elif (vn.lower() == 'pmin'):
        pmin = float(vv)
      elif (vn.lower() == 'pmax'):
        pmax = float(vv)
      elif (vn.lower() == 't0'):
        t0   = float(vv)
      elif (vn.lower() == 'rp'):
        Rp   = float(vv)
      elif (vn.lower() == 'mp'):
        Mp   = float(vv)
        mf   = True
      elif (vn.lower() == 'gp'):
        gp   = float(vv)
        gf   = True
      elif (vn.lower() == 'a'):
        a    = float(vv)
      elif (vn.lower() == 'as'):
        As   = float(vv)
      elif (vn.lower() == 'em'):
        em   = float(vv)
      elif (vn.lower() == 'phfc'):
        phfc = int(vv)
      elif (vn.lower() == 'w'):
        w   = float(vv)
      elif (vn.lower() == 'g1'):
        g1   = float(vv)
      elif (vn.lower() == 'g2'):
        g2   = float(vv)
      elif (vn.lower() == 'g3'):
        g3   = float(vv)
      elif (vn.lower() == 'pt'):
        pt   = float(vv)
      elif (vn.lower() == 'dpc'):
        dpc  = float(vv)
      elif (vn.lower() == 'tauc0'):
        tauc0 = float(vv)
      elif (vn.lower() == 'lamc0'):
        lamc0 = float(vv)
      elif (vn.lower() == 'fc'):
        fc   = float(vv)
      elif (vn.lower() == 'pf'):
        pf   = float(vv)
      elif (vn.lower() == 'tf'):
        tf   = float(vv)
      elif (vn.lower() == 'smpl'):
        vv = vv.partition(',')        
        if (len(vv[2]) > 0):
          smpl = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            smpl  = np.concatenate((smpl,[float(vv[0])]))
        else:
          smpl    = np.zeros(1)
          smpl[:] = float(vv[0])
      elif (vn.lower() == 'opdir' ):
        opdir  = vv
      elif (vn.lower() == 'snr0'):
        vv = vv.partition(',')        
        if (len(vv[2]) > 0):
          snr0 = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            snr0  = np.concatenate((snr0,[float(vv[0])]))
        else:
          snr0    = np.zeros(1)
          snr0[:] = float(vv[0])
      elif (vn.lower() == 'lam0'):
        vv = vv.partition(',')        
        if (len(vv[2]) > 0):
          lam0 = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            lam0  = np.concatenate((lam0,[float(vv[0])]))
        else:
          lam0    = np.zeros(1)
          lam0[:] = float(vv[0])
      elif (vn.lower() == 'ts'):
        Ts   = float(vv)
      elif (vn.lower() == 'rs'):
        Rs   = float(vv)
      elif (vn.lower() == 'p10'):
        p10  = float(vv)
      elif (vn.lower() == 'nlev'):
        Nlev = int(vv)
      elif (vn.lower() == 'alpha'):
        alpha   = float(vv)
      elif (vn.lower() == 'ntg'):
        ntg = int(vv)
      elif (vn.lower() == 'bg'):
        bg = vv.strip()
      elif (vn.lower() == 'ray'):
        ray  = True
        if (vv == 'False'):
          ray   = False
      elif (vn.lower() == 'cld'):
        cld  = True
        if (vv == 'False'):
          cld   = False
      elif (vn.lower() == 'ref'):
        ref  = True
        if (vv == 'False'):
          ref   = False
      elif (vn.lower() == 'sct'):
        sct  = True
        if (vv == 'False'):
          sct   = False
      elif (vn.lower() == 'fixp'):
        fixp = True
        if (vv == 'False'):
          fixp   = False
      elif (vn.lower() == 'fixt'):
        fixt = True
        if (vv == 'False'):
          fixt   = False
      elif (vn.lower() == 'rnd'):
        rnd = True
        if (vv == 'False'):
          rnd   = False
      elif (vn.lower() == 'ntype'):
        ntype = vv.strip()
      elif (vn.lower() == 'src'):
        src = vv.strip()
      elif (vn.lower() == 'lams'):
        vv = vv.partition(',')
        if (len(vv[2]) > 0):
          lams = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            lams = np.concatenate((lams,[float(vv[0])]))
        else:
          lams    = np.zeros(1)
          lams[:] = float(vv[0])
      elif (vn.lower() == 'laml'):
        vv = vv.partition(',')
        if (len(vv[2]) > 0):
          laml = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            laml = np.concatenate((laml,[float(vv[0])]))
        else:
          laml    = np.zeros(1)
          laml[:] = float(vv[0])
      elif (vn.lower() == 'res'):
        vv = vv.partition(',')
        if (len(vv[2]) > 0):
          res = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            res = np.concatenate((res,[float(vv[0])]))
        else:
          res     = np.zeros(1)
          res[:]  = float(vv[0])
      elif (vn.lower() == 'f0'):
        vv = vv.partition(',')
        if (len(vv[2]) > 0):
          f0 = [float(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            f0 = np.concatenate((f0,[float(vv[0])]))
        else:
          f0     = np.zeros(1)
          f0[:]  = float(vv[0])
      elif (vn.lower() == 'colr'):
        vv = vv.partition(',')
        if (len(vv[2]) > 0):
          colr = [int(vv[0])]
          while (len(vv[2]) > 0):
            vv = vv[2].partition(',')
            colr = np.concatenate((colr,[int(vv[0])]))
        else:
          colr     = np.zeros(1)
          colr[:]  = int(vv[0])
      elif (vn.lower() == 'regrid'):
        regrid  = True
        if (vv == 'False'):
          regrid   = False
      elif (vn.lower() == 'species_r'):
        if (vv.isspace() or len(vv) == 0):
          species_r = []
        else:
          vv = vv.partition(',')
          species_r = [vv[0].strip()]
          if (len(vv[2]) > 0):
            while (len(vv[2]) > 0):
              vv = vv[2].partition(',')
              species_r = np.concatenate((species_r,[vv[0].strip()]))
      elif (vn.lower() == 'species_l'):
        if (vv.isspace() or len(vv) == 0):
          species_l = []
        else:
          vv = vv.partition(',')
          species_l = [vv[0].strip()]
          if (len(vv[2]) > 0):
            while (len(vv[2]) > 0):
              vv = vv[2].partition(',')
              species_l = np.concatenate((species_l,[vv[0].strip()]))
      elif (vn.lower() == 'species_c'):
        if (vv.isspace() or len(vv) == 0):
          species_c = []
        else:
          vv = vv.partition(',')
          species_c = [vv[0].strip()]
          if (len(vv[2]) > 0):
            while (len(vv[2]) > 0):
              vv = vv[2].partition(',')
              species_c = np.concatenate((species_c,[vv[0].strip()]))
      elif (vn.lower() == 'restart'):
        restart  = True
        if (vv == 'False'):
          restart   = False
      elif (vn.lower() == 'fp10'):
        fp10  = True
        if (vv == 'False'):
          fp10   = False
      elif (vn.lower() == 'rdgas'):
        rdgas  = True
        if (vv == 'False'):
          rdgas   = False
      elif (vn.lower() == 'rdtmp'):
        rdtmp  = True
        if (vv == 'False'):
          rdtmp   = False
      elif (vn.lower() == 'mmr'):
        mmr  = False
        if (vv == 'True'):
          mmr   = True
      elif (vn.lower() == 'clr'):
        clr  = False
        if (vv == 'True'):
          clr   = True
      elif (vn.lower() == 'fmin'):
        fmin   = float(vv)
      elif (vn.lower() == 'nwalkers'):
        nwalkers = int(vv)
      elif (vn.lower() == 'nstep'):
        nstep = int(vv)
      elif (vn.lower() == 'nburn'):
        nburn = int(vv)
      elif (vn.lower() == 'thin'):
        thin = int(vv)
      elif (vn.lower() == 'grey'):
        grey  = True
        if (vv == 'False'):
          grey   = False
      elif (vn.lower() == 'progress'):
        progress  = True
        if (vv == 'False'):
          progress   = False

  # set pf to -1 if user does not want iso-pressure opacities
  if (not fixp):
    pf = -1

  # set tf to -1 if user does not want iso-temperature opacities
  if (not fixt):
    tf = -1

  # check for consistency between wavelength grid and resolution grid
  if (lams.shape[0] > 1 and lams.shape[0] != res.shape[0]):
    print("rfast warning | major | smpl length inconsistent with wavelength grid")
    quit()

  # check for consistency between resolution grid and over-sample factor
  if (smpl.shape[0] > 1 and smpl.shape[0] != res.shape[0]):
    print("rfast warning | major | smpl length inconsistent with resolution grid")
    quit()

  # check for consistency between resolution grid and snr0 parameter
  if (snr0.shape[0] > 1 and snr0.shape[0] != res.shape[0]):
    print("rfast warning | major | snr0 length inconsistent with wavelength grid")
    quit()

  # check for consistency between resolution grid and lam0 parameter
  if (lam0.shape[0] > 1 and lam0.shape[0] != res.shape[0]):
    print("rfast warning | major | lam0 length inconsistent with wavelength grid")
    quit()

  # check that snr0 is within applicable wavelength range
  if (lam0.shape[0] > 1):
    for i in range(lam0.shape[0]):
      if (lam0[i] < min(lams) or lam0[i] > max(laml)):
        print("rfast warning | major | lam0 outside wavelength grid")
        quit()
  else:
    if (lam0[0] < min(lams) or lam0[0] > max(laml)):
      print("rfast warning | major | lam0 outside wavelength grid")
      quit()

  # complete directory path if '/' is omitted
  if (len(opdir) > 0 and opdir[-1] != '/'):
    opdir = opdir + '/'
  if (len(opdir) == 0):
    opdir = './hires_opacities/'

  # check if opacities directory exists
  if (not os.path.isdir(opdir)):
    print("rfast warning | major | opacities directory does not exist")
    quit()

  # check if output directory exist; create if it does not
  if (len(dirout) > 0 and dirout[-1] != '/'):
    dirout = dirout + '/'
  if (len(dirout) > 0 and not os.path.isdir(dirout) and sys.argv[0] == 'rfast_genspec.py'):
    print("rfast warning | minor | output directory does not exist, attempting to create")
    os.makedirs(dirout)
  elif (len(dirout) > 0 and not os.path.isdir(dirout)):
    print("rfast warning | minor | output directory does not exist, use current directory")
    dirout = os.getcwd() + '/'

  # check for mixing ratio issues
  if (np.sum(f0) - 1 > 1.e-6 and not rdgas):
    if (np.sum(f0) - 1 < 1.e-3):
      print("rfast warning | minor | input gas mixing ratios sum to slightly above unity")
    else:
      print("rfast warning | major | input gas mixing ratios sum to much above unity")
      quit()

  # set gravity if gp not set
  if (not gf):
    gp  = 9.798*Mp/Rp**2

  # set planet mass if not set
  if (not mf):
    Mp = (gp/9.798)*Rp**2

  # cannot have both Mp and gp
  if (mf and gf):
    print("rfast warning | major | cannot independently set planet mass and gravity in inputs")
    quit()

  # cloud base cannot be below bottom of atmosphere
  if (pt+dpc > pmax):
    print("rfast warning | major | cloud base below bottom of atmosphere")
    quit()

  # transit radius pressure cannot be larger than max pressure
  if (p10 > pmax):
    print("rfast warning | major | transit radius pressure below bottom of atmosphere")
    quit()

  return fnr,fnn,fns,dirout,Nlev,pmin,pmax,bg,\
         species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,imix,\
         t0,rdtmp,fntmp,skptmp,colt,colpt,psclt,\
         species_l,species_c,\
         lams,laml,res,regrid,smpl,opdir,\
         Rp,Mp,gp,a,As,em,\
         grey,phfc,w,g1,g2,g3,pt,dpc,tauc0,lamc0,fc,\
         ray,cld,ref,sct,fixp,pf,fixt,tf,p10,fp10,\
         src,\
         alpha,ntg,\
         Ts,Rs,\
         ntype,snr0,lam0,rnd,\
         clr,fmin,mmr,nwalkers,nstep,nburn,thin,restart,progress
#
#
# initializes opacities and convolution kernels
#
# inputs:
#
#    lam_lr   - low-resolution wavelength grid midpoints (um)
#   dlam_lr   - low-resolution wavelength grid widths (um)
#    lam_hr   - high-resolution wavelength grid midpoints (um)
#     mode    - optional, indicates photometry vs. spectroscopy
#  species_l  - list of line absorbers to include
#  species_c  - list of cia absorbers to include
#    opdir    - directory where hi-res opacities are located (string)
#     pf      - pressure for iso-pressure case (Pa)
#     tf      - temperature for iso-temperature case (K)
#
# outputs:
#
# sigma_interp- line absorber opacity interpolation function (m**2/molecule)
#    sigma_cia- cia coefficient interpolation function (m**-1 m**-6)
#    kern     - convolution kernel
#
def init(lam_lr,dlam_lr,lam_hr,species_l,species_c,opdir,pf,tf,mode=-1):

  # gaussian kernel for later degrading
  kern = kernel(lam_lr,lam_hr,Dx = dlam_lr,mode = mode)

  # read in line absorbers
  press,temp,sigma = opacities_read(species_l,lam_hr,opdir)

  # setup up opacities interpolation routine
  x  = np.log10(press)
  y  = 1/temp
  z  = np.log10(sigma)

  # gradients in log-pressure, inverse temperature
  gradx = np.gradient(z,x,axis=1)
  grady = np.gradient(z,y,axis=2)

  # define function that interpolates opacities to p, T grid
  # x0 is log10(pressure), y0 is inverse temperature
  # x0, y0 are vectors of identical length (e.g., atm. model)
  # returns log10(opacity) in m**2/molecule
  def sigma_interp_2D(x0,y0):

    # finds gridpoints nearest to interpolation points
    dx    = np.subtract.outer(x,x0)
    ix    = np.argmin(np.absolute(dx),axis=0)
    dy    = np.subtract.outer(y,y0)
    iy    = np.argmin(np.absolute(dy),axis=0)

    # matrices of distances from interpolation points
    dx    = x[ix]-x0
    dx    = np.repeat(dx[np.newaxis,:], gradx.shape[0], axis=0)
    dx    = np.repeat(dx[:,:,np.newaxis], gradx.shape[3], axis=2)
    dy    = y[iy]-y0
    dy    = np.repeat(dy[np.newaxis,:], grady.shape[0], axis=0)
    dy    = np.repeat(dy[:,:,np.newaxis], grady.shape[3], axis=2)

    # interpolate using gradients, distances
    z0    = z[:,ix,iy,:] - np.multiply(gradx[:,ix,iy,:],dx) - np.multiply(grady[:,ix,iy,:],dy)

    return z0

  # set up 1-D interpolation, if needed
  if ( np.any( pf != -1) or np.any( tf != -1) ): # fixed p or t case
    if ( np.any( pf != -1) and np.any( tf == -1) ): # fixed p case
      p0    = np.zeros(len(temp))
      p0[:] = pf
      sigma = np.power(10,sigma_interp_2D(np.log10(p0),1/temp))
      sigma_interp_1D = interpolate.interp1d(1/temp,np.log10(sigma),axis=1,assume_sorted=True,fill_value="extrapolate")
      sigma_interp = sigma_interp_1D
    elif ( np.any( pf == -1) and np.any( tf != -1) ): # fixed t case
      t0    = np.zeros(len(press))
      t0[:] = tf
      sigma = np.power(10,sigma_interp_2D(np.log10(press),1/t0))
      sigma_interp_1D = interpolate.interp1d(np.log10(press),np.log10(sigma),axis=1,assume_sorted=True,fill_value="extrapolate")
      sigma_interp = sigma_interp_1D
    else: # fixed p and t case
      p0     = np.zeros(len(temp))
      p0[:]  = pf
      sigma  = np.power(10,sigma_interp_2D(np.log10(p0),1/temp))
      sigma_interp_1D = interpolate.interp1d(1/temp,np.log10(sigma),axis=1,assume_sorted=True,fill_value="extrapolate")
      sigma0 = np.power(10,sigma_interp_1D(1/tf))
      def sigma_interp_0D():
        return np.log10(sigma0)
      sigma_interp = sigma_interp_0D
  else: # general variable p and t case
    sigma_interp = sigma_interp_2D

  # read in and down-sample collision-induced absorbers
  tempcia,kcia,ncia,ciaid = cia_read(species_c,lam_hr,opdir)
  cia_interp              = interpolate.interp1d(1/tempcia,kcia,axis=1,assume_sorted=True,fill_value="extrapolate")

  return sigma_interp,cia_interp,ncia,ciaid,kern
#
#
# initialize quantities for disk integration
#
# inputs:
#
#       src   - model type flag
#       ntg   - number of Tchebyshev and Gauss integration points
#
# outputs:
#
#    threeD   - if src is phas -> Gauss and Tchebyshev points and weights
#
def init_3d(src,ntg):

  threeD = -1

  # if doing 3-D model
  if (src == 'phas'):

    # set Gauss / Tchebyshev points and weights
    thetaG, wG = np.polynomial.legendre.leggauss(ntg)
    thetaT, wT = tchebyshev_pts(ntg)

    # take advantage of symmetry about illumination equator
    thetaT = thetaT[0:math.ceil(ntg/2)] # only need 1/2 Tchebyshev points
    wT     = wT[0:math.ceil(ntg/2)]     # also only need 1/2 weights
    if (ntg % 2 != 0):
        wT[-1] = 0.5*wT[-1]           # if n is odd, must halve equatorial weight

    threeD = thetaG,thetaT,wG,wT

  return threeD
#
#
# set weights (kernel) for spectral convolution
#
# inputs:
#
#          x  - low-resolution spectral grid
#       x_hr  - high-resolution spectral grid (same units as x)
#
# outputs:
#
#       kern  - array describing wavelength-dependent kernels for
#               convolution (len(x) x len(x_hr))
#
# options:
#
#         Dx  - widths of low-resolution gridpoints (len(x))
#       mode  - vector (len(x)) of integers indicating if 
#               x_i is a spectroscopic point (1) or photometric 
#               point.  if not set, assumes all are spectroscopic 
#               and applies gaussian lineshape.
#
# notes:
#
#   designed to pair with kernel_convol function.  heavily modified 
#   and sped-up from a version originated by Mike Line.
#
def kernel(x,x_hr,Dx = -1,mode = -1):

  # number of points in lo-res grid
  Nx= len(x)

  # compute widths if not provided
  if ( np.any( Dx == -1) ):
    dx  = np.zeros(Nx)
    xm  = 0.5*(x[1:] + x[:-1])
    dx1 = xm[1:] - xm[:-1]
    dx[1:-1]    = dx1[:]
    res_interp  = interpolate.interp1d(x[1:-1],x[1:-1]/dx1,fill_value="extrapolate")
    dx[0]    = x[0]/res_interp(x[0])
    dx[Nx-1] = x[Nx-1]/res_interp(x[Nx-1])
  else:
    dx    = np.zeros(Nx)
    dx[:] = Dx

  # initialize output array
  kern = np.zeros([Nx,len(x_hr)])

  # loop over lo-res grid and compute convolution kernel
  fac = (2*(2*np.log(2))**0.5) # ~= 2.355

  # case where mode is not specified
  if ( np.any( mode == -1) ):
    for i in range(Nx):

      # FWHM = 2.355 * standard deviation of a gaussian
      sigma=dx[i]/fac

      # kernel
      kern[i,:]=np.exp(-(x_hr[:]-x[i])**2/(2*sigma**2))
      kern[i,:]=kern[i,:]/np.sum(kern[i,:])

  # case where mode is specified
  else:
    for i in range(Nx):
      if (mode[i] == 1): # spectroscopic point
        # FWHM = 2.355 * standard deviation of a gaussian
        sigma=dx[i]/fac

        # kernel
        kern[i,:] = np.exp(-(x_hr[:]-x[i])**2/(2*sigma**2))
        sumk      = np.sum(kern[i,:])
        if (sumk != 0):
          kern[i,:] = kern[i,:]/np.sum(kern[i,:])
        else:
          kern[i,:] = 0

      elif (mode[i] == 0): # photometric point
        j         = np.squeeze(np.where(np.logical_and(x_hr >= x[i]-Dx[i]/2, x_hr <= x[i]+Dx[i]/2)))
        if ( len(j) > 0 ):
          kern[i,j] = 1
          # edge handling
          jmin      = j[0]
          jmax      = j[-1]
          if (jmin == 0):
            Dxmin = abs(x_hr[jmin+1]-x_hr[jmin])
          else:
            Dxmin = abs( 0.5*(x_hr[jmin]+x_hr[jmin+1]) -  0.5*(x_hr[jmin]+x_hr[jmin-1]) )
          if (jmax == len(x_hr)-1):
            Dxmax = abs(x_hr[jmax]-x_hr[jmax-1])
          else:
            Dxmax = abs( 0.5*(x_hr[jmax]+x_hr[jmax+1]) -  0.5*(x_hr[jmax]+x_hr[jmax-1]) )
          xb = (x[i]-Dx[i]/2) - (x_hr[jmin]-Dxmin/2)
          xa = (x_hr[jmax]-Dxmax/2) - (x[i]+Dx[i]/2)
          if (xb >= 0):
            fb = 1 - xb/Dxmin
          else:
            fb = 1
          if (xa >= 0):
            fa = 1 - xa/Dxmax
          else:
            fa = 1
          kern[i,jmin] = fb
          kern[i,jmax] = fa
          kern[i,:] = kern[i,:]/np.sum(kern[i,:]) #re-normalize

  return kern
#
#
# convolve spectrum with general kernel
#
# inputs:
#
#      kern   - kernel matrix from kernel (len(low-res) x len(high-res))
#   spec_hr   - hi-res spectrum (len(hi-res))
#
# outputs:
#
#   spec_lr   - degraded spectrum (len(low-res))
#
def kernel_convol(kern,spec_hr):

  spec_lr    = np.zeros(kern.shape[0])
  conv       = np.multiply(kern,spec_hr)
  spec_lr[:] = np.sum(conv,axis=1)

  return spec_lr
#
#
# simple wavelength-dependent noise model
#
# inputs:
#
#      lam0   - wavelength where snr0 is normalized to (um)
#      snr0   - signal-to-noise at lam0
#       lam   - wavelength (um)
#      dlam   - wavelength bin width (um)
#      FpFs   - planet-to-star flux ratio
#        Ts   - host star effective temperature
#     ntype   - noise type, options are:
#                 'csnr' = constant snr
#                 'cnse' = constant noise
#                 'cerr' = constant error in FpFs
#                 'plan' = noise dominated by planetary counts
#                 'ezod' = noise dominated by exozodiacal light
#                 'detr' = noise dominated by detector
#                 'leak' = noise dominated by stellar leakage
#
# outputs:
#
#       err   - 1-sigma uncertainty in planet-to-star flux ratio
#
# notes:
#
#   assumes transmission, quantum efficiency, raw contrast are all grey.
#
def noise(lam0,snr0,lam,dlam,FpFs,Ts,ntype):

  # scalings based on dominant noise type
  if (ntype == 'csnr'):
    snr    = np.zeros(lam.shape[0])
    snr[:] = snr0
    err    = FpFs/snr
  elif (ntype == 'cnse'):
    FpFs_interp  = interpolate.interp1d(lam,FpFs)
    res_interp   = interpolate.interp1d(lam,lam/dlam)
    snr          = snr0*(FpFs/FpFs_interp(lam0))*((lam/lam0)**2)*(res_interp(lam0)/(lam/dlam))*(planck(lam,Ts)/planck(lam0,Ts))
    err          = FpFs/snr
  elif (ntype == 'cerr'):
    FpFs_interp  = interpolate.interp1d(lam,FpFs)
    err          = np.zeros(lam.shape[0])
    err[:]       = FpFs_interp(lam0)/snr0
  elif (ntype == 'plan'):
    FpFs_interp  = interpolate.interp1d(lam,FpFs)
    res_interp   = interpolate.interp1d(lam,lam/dlam)
    snr          = snr0*((FpFs/FpFs_interp(lam0))**0.5)*(lam/lam0)*((res_interp(lam0)/(lam/dlam))**0.5)*((planck(lam,Ts)/planck(lam0,Ts))**0.5)
    err          = FpFs/snr
  elif (ntype == 'ezod'):
    FpFs_interp  = interpolate.interp1d(lam,FpFs)
    res_interp   = interpolate.interp1d(lam,lam/dlam)
    snr          = snr0*(FpFs/FpFs_interp(lam0))*((res_interp(lam0)/(lam/dlam))**0.5)*((planck(lam,Ts)/planck(lam0,Ts))**0.5)
    err          = FpFs/snr
  elif (ntype == 'detr'):
    FpFs_interp  = interpolate.interp1d(lam,FpFs)
    res_interp   = interpolate.interp1d(lam,lam/dlam)
    snr          = snr0*(FpFs/FpFs_interp(lam0))*(lam/lam0)*(res_interp(lam0)/(lam/dlam))*(planck(lam,Ts)/planck(lam0,Ts))
    err          = FpFs/snr
  elif (ntype == 'leak'):
    FpFs_interp  = interpolate.interp1d(lam,FpFs)
    res_interp   = interpolate.interp1d(lam,lam/dlam)
    snr          = snr0*(FpFs/FpFs_interp(lam0))*(lam/lam0)*((res_interp(lam0)/(lam/dlam))**0.5)*((planck(lam,Ts)/planck(lam0,Ts))**0.5)
    err          = FpFs/snr

  return err
#
#
# planck function
#
# inputs:
#
#       lam   - wavelength (um)
#        T    - temperature (K)
#
# outputs:
#
#      Blam   - planck intensity (W m**-2 um**-1 sr**-1)
#
def planck(lam,T):

  # constants
  kB    = 1.38064852e-23 # m**2 kg s**-2 K**-1
  h     = 6.62607015e-34 # kg m**2 s**-1
  c     = 2.99792458e8   # m s**-1

  # convert to m
  lam0  = lam/1.e6

  # planck function in W m**-2 m**-1 sr**-1
  Blam  = 2*h*c**2/lam0**5/(np.exp(h*c/kB/T/lam0) - 1)

  return Blam/1.e6
#
#
# planck function, w/matrix ops for all lam / T combos
#
# inputs:
#
#       lam   - wavelength (um)
#        T    - temperature (K)
#
# outputs:
#
#      Blam   - planck intensity (W m**-2 um**-1 sr**-1) [Nlam,Ntemp]
#
def planck2D(lam,T):

  # number of elements
  Nlam  = lam.shape[0]
  Ntemp = T.shape[0]

  # constants
  kB    = 1.38064852e-23 # m**2 kg s**-2 K**-1
  h     = 6.62607015e-34 # kg m**2 s**-1
  c     = 2.99792458e8   # m s**-1

  # combined constants
  a      = np.zeros([Nlam,Ntemp])
  b      = np.zeros([Nlam,Ntemp])
  a[:,:] = 2*h*c**2 # m**4 kg s**-3
  b[:,:] = h*c/kB   # m K

  # identity matrix
  id      = np.zeros([Nlam,Ntemp])
  id[:,:] = 1.

  # 2D matrices
  lam0  = np.repeat(lam[:,np.newaxis], Ntemp, axis=1)/1.e6 # m
  T0    = np.repeat(T[np.newaxis,:], Nlam, axis=0)

  # planck function in W m**-2 m**-1 sr**-1
  Blam  = np.divide(np.divide(a,np.power(lam0,5)),(np.exp(np.divide(b,np.multiply(T0,lam0)))-id))

  return Blam/1.e6
#
#
# brightness temperature
#
# inputs:
#
#       lam   - wavelength (um)
#      Flam   - specific flux density (W m**-2 um**-1)
#
# outputs:
#
#        Tb   - brightness temperature (K)
#
def Tbright(lam,Flam):

  # constants
  kB    = 1.38064852e-23 # m**2 kg s**-2 K**-1
  h     = 6.62607015e-34 # kg m**2 s**-1
  c     = 2.99792458e8   # m s**-1

  # conversions to mks
  lam0  = lam/1.e6
  Ilam0 = Flam*1.e6/np.pi

  # brightness temperature
  Tb    = h*c/kB/lam0/np.log(1 + 2*h*c**2/Ilam0/lam0**5)

  return Tb
#
#
# compute transit depth spectrum using Robinson (2017) formalism
#
# inputs:
#
#        Rp   - planetary radius (Rearth)
#        Rs   - stellar radius (Rsun)
#         z   - vertical altitude grid (m)
#      dtau   - vertical differential optical depths [Nlam x Nlay]
#         p   - pressure grid
#       ref   - include refractive floor? (T/F)
#      pref   - pressure location of refractive floor
#
#
# outputs:
#
#        td   - transit depth spectrum
#
def transit_depth(Rp,Rs,z,dtau,p,ref,pref=-1):

  Nlev = dtau.shape[1] + 1 # number of levels
  Nlam = dtau.shape[0]     # number of wavelengths
  Re   = 6.378e6           # earth radius (m)
  Rsun = 6.957e8           # solar radius (m)
  Nlay = Nlev-1

  # grid of impact parameters
  b    = Rp*Re + z
  bm   = 0.5*(b[1:] + b[:-1])

  # geometric path distribution
  bs  = b*b
  bms = bm*bm
  dn  = bms[None,:] - bs[0:Nlay,None]
  Pb  = np.divide(4*np.power(np.repeat(bm[:,np.newaxis],Nlay,axis=1),2),dn)
  izero = np.where(dn < 0)
  Pb[izero] = 0
  Pb  = np.power(Pb,0.5)
  
  # integrate along slant paths, compute transmission
  tau = np.transpose(np.dot(Pb,np.transpose(dtau)))
  t   = np.exp(-tau)

  # refractive floor
  if ref:
    iref = np.where(p[0:Nlay] >= pref)
    t[:,iref] = 0.

  # integrate over annuli
  A   = b[:-1]**2 - b[1:]**2 # annulus area
  td  = np.dot(1-t,A) + (Rp*Re)**2
  td  = td/(Rs*Rsun)**2

  return td
#
#  refractive floor location from analytic expression in robinson et al. (2017)
#
def refract_floor(nu,t,Rs,a,Rp,m,grav):
  Re   = 6.378e6        # earth radius (m)
  Rjup = 6.991e7        # ju[iter radius (m)
  Na   = 6.0221408e23   # avogradro's number
  return 23e2*(1.23e-4/nu)*(np.mean(t)/130)**1.5*(Rs)*(5.2/a)*(Rjup/Rp/Re)**0.5*(2.2/Na/np.mean(m)/1e3)**0.5*(24.8/np.mean(grav))**0.5
#
#
# spectral grid routine, general
#
# inputs:
#
#      x_min  - minimum spectral cutoff
#      x_max  - maximumum spectral cutoff
#         dx  - if set, adopts fixed spacing of width dx
#        res  - if set, uses fixed or spectrally varying resolving power
#       lamr  - if set, uses spectrally varying resolving power and
#               lamr must have same size as res
#
# outputs:
#
#          x  - center of spectral gridpoints
#         Dx  - spectral element width
#
# notes:
#
#   in case of spectrally-varying resolving power, we use the 
#   derivative of the resolving power at x_i to find the resolving 
#   power at x_i+1.  this sets up a quadratic equation that relates 
#   the current spectral element to the next.
#
def spectral_grid(x_min,x_max,res = -1,dx = -1,lamr = -1):

  # constant resolution case
  if ( np.any(dx != -1) ): 
    x     = np.arange(x_min,x_max,dx)
    if (max(x) + dx == x_max):
      x = np.concatenate((x,[x_max]))
    Dx    = np.zeros(len(x))
    Dx[:] = dx
  # scenarios with constant or non-constant resolving power
  if ( np.any( res != -1) ):
    if ( np.any( lamr == -1) ): # constant resolving power
      x,Dx = spectral_grid_fixed_res(x_min,x_max,res)
    else: #spectrally-varying resolving power
      # function for interpolating spectral resolution
      res_interp = interpolate.interp1d(lamr,res,fill_value="extrapolate")

      # numerical derivative and interpolation function
      drdx = np.gradient(res,lamr)
      drdx_interp = interpolate.interp1d(lamr,drdx,fill_value="extrapolate")

      # initialize
      x  = [x_min]
      Dx = [x_min/res_interp(x_min)]
      i  = 0

      # loop until x_max is reached
      while (x[i] < x_max):
        resi =  res_interp(x[i])
        resp = drdx_interp(x[i])
        a = 2*resp
        b = 2*resi - 1 - 4*x[i]*resp - resp/resi*x[i]
        c = 2*resp*x[i]**2 + resp/resi*x[i]**2 - 2*x[i]*resi - x[i]
        if (a != 0 and resp*x[i]/resi > 1.e-6 ):
          xi1 = (-b + np.sqrt(b*b - 4*a*c))/2/a
        else:
          xi1 = (1+2*resi)/(2*resi-1)*x[i]
        Dxi = 2*(xi1 - x[i] - xi1/2/res_interp(xi1))
        x  = np.concatenate((x,[xi1]))
        i  = i+1
      if (max(x) > x_max):
        x  = x[0:-1]

      Dx = x/res_interp(x)

  return np.squeeze(x),np.squeeze(Dx)
#
#
# spectral grid routine, fixed resolving power
#
# inputs:
#
#     res     - spectral resolving power (x/dx)
#     x_min   - minimum spectral cutoff
#     x_max   - maximumum spectral cutoff
#
# outputs:
#
#         x   - center of spectral gridpoints
#        Dx   - spectral element width
#
def spectral_grid_fixed_res(x_min,x_max,res):
#
  x    = [x_min]
  fac  = (1 + 2*res)/(2*res - 1)
  i    = 0
  while (x[i]*fac < x_max):
    x = np.concatenate((x,[x[i]*fac]))
    i  = i + 1
  Dx = x/res
#
  return np.squeeze(x),np.squeeze(Dx)
#
#
# tchebyshev points and weights (Webber et al., 2015)
#
# inputs:
#
#     n       - number of tchebyshev points
#
# outputs:
#
#         x   - points
#         w   - weights
#
def tchebyshev_pts(n):
  i = np.arange(1,n+1,step=1)
  x = np.cos(np.pi*i/(n+1))
  w = np.pi/(n+1)*(np.sin(np.pi*i/(n+1)))**2

  return x,w
#
#
# rayleigh phase function (normalized so that integral over dcosTh from -1 to 1 is 2)
#
# inputs:
#
#     cosTh   - cosine of scattering angle
#
# outputs:
#
#     phase function at cosTh
#
def pray(cosTh):
  return 3/4*(1+cosTh**2)
#
#
# henyey-greenstein phase function (normalized so that integral over dcosTh from -1 to 1 is 2)
#
# inputs:
#
#         g   - asymmetry parameter
#     cosTh   - cosine of scattering angle
#
# outputs:
#
#     phase function at cosTh
#
def pHG(g,cosTh):
  id      = np.copy(g)
  id[:]   = 1.
  return np.divide((id-np.multiply(g,g)),np.power(id+np.multiply(g,g)-2*cosTh*g,1.5))
#
#
# henyey-greenstein phase function integrated from y to +1
#
# inputs:
#
#         g   - asymmetry parameter
#     cosTh   - cosine of scattering angle
#
# outputs:
#
#     integral of HG phase function from cosTh to +1
#
def pHG_int(g,cosTh):
  id      = np.copy(g)
  id[:]   = 1.
  soln    = np.copy(g)
  soln[:] = 0.
  iz = np.where(g == 0)
  if iz[0].size !=0:
    soln[iz] = 1-cosTh
  iz = np.where(g != 0)
  if iz[0].size !=0:
    soln[iz] = np.divide((id[iz]-np.multiply(g[iz],g[iz])),np.multiply(g[iz],np.power(id[iz]+np.multiply(g[iz],g[iz])-2*g[iz],0.5)))
    soln[iz] = soln[iz] - np.divide((id[iz]-np.multiply(g[iz],g[iz])),np.multiply(g[iz],np.power(id[iz]+np.multiply(g[iz],g[iz])-2*cosTh*g[iz],0.5)))  
  return soln
#
#
# three-moment expansion of HG phase function (normalized so that integral from -1 to 1 is 2)
#
# inputs:
#
#         g   - asymmetry parameter
#     cosTh   - cosine of scattering angle
#
# outputs:
#
#     phase function at cosTh
#
def pHG3(g,cosTh):
  id      = np.copy(g)
  id[:,:] = 1.
  return id + 3*g*cosTh + 5/2*np.multiply(g,g)*(3*cosTh**2 - 1)
#
#
# three-moment expansion of HG phase function integrated from y to +1
#
# inputs:
#
#         g   - asymmetry parameter
#     cosTh   - cosine of scattering angle
#
# outputs:
#
#     integral of three-moment HG phase function from cosTh to +1
#
def pHG3_int(g,cosTh):
  id      = np.copy(g)
  id[:]   = 1.
  soln = -0.5*(cosTh-1)*(5*np.multiply(g,g)*cosTh*(cosTh+1) + 3*g*(cosTh+1) + 2*id)
  return soln