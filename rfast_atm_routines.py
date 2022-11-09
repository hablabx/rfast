import numpy             as     np
import math
import time
from   scipy.interpolate import interp1d
#
# set up hard-coded gas parameters
#
# inputs:
#
#       bg   - background gas identifier ('ar','ch4','co2','h2','he','n2','o2')
#
def set_gas_info(bg):

  # number of absorbing or scattering gases
  Ngas  = 11 # Ar, CH4, CO2, H2, H2O, He, N2, O2, O3

  # gas names
  gasid = ['ar','ch4','co2','h2','h2o','he','n2','o2','o3','n2o','co']

  # in gasid order:
  #   molar weight (g/mole)
  #   cross section (m**2/molecule) at 0.4579 um for
  #   STP refractivity
  #   note: zero is used as placeholder in ray0 and nu0 for gases that will "always" be trace
  #   note: see sneep & ubacks 2005, jqsrt 92:293--310
  mmw0  =  np.float32([  39.948,    16.04,    44.01, 2.0159,   18.015,  4.0026,   28.013,  31.999, 48.000,   44.013,   28.01])
  ray0  =  np.float32([8.29e-31,2.035e-30,2.454e-30,2.16e-31,8.93e-31,1.28e-32,10.38e-31,8.39e-31, 0.0000,2.903e-30,11.3e-31])
  nu0   =  np.float32([0.000281,0.000444, 0.000449, 0.000132,0.000261,0.000035, 0.000298,0.000271, 0.0000, 0.000483,0.000325])

  # set background gas parameters, noting that rayb is relative to argon
  ib    = gasid.index(bg.lower())
  mb    = mmw0[ib]
  rayb  = ray0[ib]/ray0[gasid.index('ar')]

  return Ngas,gasid,mmw0,ray0,nu0,mb,rayb
#
#
# set up atmospheric model
#
# inputs:
#
#      Nlev   - number of atmospheric levels
#      pmin   - top of atmosphere pressure (Pa)
#      pmax   - bottom of atmosphere pressure (Pa)
#        t0   - temperature for isothermal profile (K)
#     rdtmp   - if true, read gas mixing ratios from file
#     fntmp   - filename for thermal structure
#    skptmp   - lines to skip for header in fntmp
#      colt   - column of temperature in fntmp
#     colpt   - pressure column in fntmp
#     psclt   - factor to convert pressure to Pa
#      Ngas   - number of potential radiatively active gases, from set_gas_info
# species_r   - identifiers of user-requested radiatively active gases
#        f0   - gas mixing ratios for vertically-constant case
#     rdgas   - if true, read gas mixing ratios from file
#     fnatm   - filename for gas mixing ratios
#    skpatm   - lines to skip for header in fnatm
#      colr   - columns of gas mixing ratios corresponding to species_r
#     colpr   - pressure column in fnatm
#     psclr   - factor to convert pressure to Pa
#      mmw0   - gas mean molar weights (see set_gas_info)
#       mmr   - if true, mixing ratios are interpreted as mass mixing ratios
#       cld   - flag that indicates if clouds are included
#        pt   - cloud top pressure (Pa)
#       dpc   - cloud thickness (dpc)
#     tauc0   - cloud optical depth at user-specified wavelength
#       ref   - flag to indicate if refraction is included
#         t   - temperature profile (K)
#        mb   - background gas mean molar weight (g/mole)
#        Mp   - planetary mass (Mearth)
#        Rp   - planetary radius (Rearth)
#       src   - model type (diff,thrm,cmbn,scnd,trns)
#       p10   - pressure for fixed-pressure planetary radius (Pa)
#      fp10   - if true, Rp is interpreted as being given at p10
#       ref   - flag to indicate if refraction is included for transit case
#       nu0   - gas STP refractivities (see set_gas_info)
#        gp   - (optional) planetary surface gravity (m s**-2)
#
# outputs:
#
#         p   - pressure grid (Pa) (Nlev)
#         t   - temperature profile (K) (Nlev)
#         z   - altitude profile (m) (Nlev)
#      grav   - gravity profile (m/s/s) (Nlev)
#         f   - gas mixing ratio profiles (Ngas x Nlev)
#         m   - atmospheric mean molecular weight (Nlev)
#       nu0   - atmospheric refractivity at STP (float)
#
def setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
              t0,rdtmp,fntmp,skptmp,colt,colpt,psclt,
              species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
              mmr,mb,Mp,Rp,cld,pt,dpc,tauc0,p10,fp10,src,ref,nu0,gp=-1):

  # small mixing ratio for non-active gases
  rsmall = 1e-10

  # set pressure grid (ie, vertical grid)
  p = set_press_grid(Nlev,pmin,pmax,cld,pt,dpc,tauc0,src)

  # ensure p10 is in pressure grid, for transit spectra and if used
  if (src == "trns" and fp10):
    ip    = np.where(abs(p-p10) == min(abs(p-p10)))[0]
    #if( ip[0] == Nlev-1 ):
    #  ip[0] = Nlev-2
    p[ip] = p10    

  # set temperature profile -- currently assumes isothermal
  if not rdtmp:
    t     = np.zeros(Nlev)
    t[:]  = t0
  else:
    dat  = readdat(fntmp,skptmp) # [ levels x input columns]
    pd   = dat[:,colpt-1]*psclt
    td   = dat[:,colt-1]
    td_interp = interp1d(pd,td,fill_value="extrapolate")
    t    = td_interp(p)
    t[-1]= t0 # set surface / lower boundary temperature   

  # set gas profiles -- either vertically constant or read in
  f    = np.zeros([Ngas,Nlev])
  f[:] = rsmall
  if not rdgas:
    i = 0
    for id in species_r:
      ig = gasid.index(id.lower())
      f[ig,:] = f0[i]
      i  = i + 1
  else:
    dat  = readdat(fnatm,skpatm) # [ levels x input columns]
    pd   = dat[:,colpr-1]*psclr
    i    = 0
    for id in species_r:
      fd        = dat[:,colr[i]-1]
      fd_interp = interp1d(pd,np.log10(fd),fill_value="extrapolate")
      fi = np.power(10,fd_interp(p))
      ig = gasid.index(id.lower())
      f[ig,:] = fi
      i  = i + 1    

  # mixing ratios summed across all species
  ft = np.sum(f,axis=0)

  # fill amount, cannot be less than zero
  fb         = 1 - ft
  fb[fb < 0] = 0

  # set mean molecular weight (kg/molec)
  m = mmw(f,fb,mb,mmw0,mmr)

  # set refractivity at STP
  nu = 0.
  if (ref and src == 'trns'):
    nu = refractivity(f,nu0,mmw0,mmr,m)

  # do hydrostatic calculation
  #tstart = time.time()
  if (src == "trns" and fp10):
    z,grav = hydrostat(Nlev,p,t,m,Mp,Rp,gp,ip=ip[0])
  else:
    z,grav = hydrostat(Nlev,p,t,m,Mp,Rp,gp)
  #print('Hydrostatic calculation timing (s): ',time.time()-tstart)

  return p,t,z,grav,f,fb,m,nu
#
#
# hydrostatic calculation
#
# inputs:
#
#      Nlev   - number of atmospheric levels
#         p   - pressure grid (Pa)
#         t   - temperature profile (K)
#         m   - mean molecular weight profile (kg/molec)
#        Mp   - planetary mass (Mearth)
#        Rp   - planetary radius (Rearth)
#        gp   - planetary surface gravity (m s**-2; optional; supersedes Mp if used)
#        ip   - if set, pressure index where assumed Rp, gp apply
#
# outputs:
#
#         z   - altitude profile (m)
#      grav   - gravity profile (m/s/s)
#
def hydrostat(Nlev,p,t,m,Mp,Rp,gp,ip=-1):

  # earth radius (m)
  Re   = 6.378e6

  # surface gravity (m/s/s)
  if (gp == -1):
    grav0 = 9.798*Mp/Rp**2
  else:
    grav0 = gp

  # universal gas constant
  kB = 1.38064852e-23 # m**2 kg s**-2 K**-1

  # altitude and gravity grids
  z    = np.zeros(Nlev)
  grav = np.zeros(Nlev)

  # mean layer mean molecular weight
  mm = 0.5*(m[1:] + m[:-1])

  # case where Rp, gp apply at bottom of pressure profile
  if (ip == -1):

    # surface values
    z[Nlev-1]    = 0.
    grav[Nlev-1] = grav0

    # iterate upward from surface
    for i in range(1,Nlev):
      k    = Nlev-i-1
      a    = kB/grav0/mm[k]
      fac  = a*((t[k+1]-(t[k]-t[k+1])/(p[k]-p[k+1])*p[k+1])*np.log(p[k+1]/p[k])-(t[k]-t[k+1]))
      fac  = (Rp*Re)/(1+(z[k+1]/Rp/Re)) - fac
      z[k] = (Rp*Re/fac - 1)*Rp*Re
      grav[k] = grav0*(Rp*Re)**2/(Rp*Re + z[k])**2

  # case where Rp, gp apply somewhere else in profile
  else:

    # values at ip
    z[ip]    = 0.
    grav[ip] = grav0

    # iterate upward from ip
    for i in range(1,ip+1):
      k    = ip-i
      a    = kB/grav0/mm[k]
      fac  = a*(t[k+1]-(t[k]-t[k+1])/(p[k]-p[k+1]))*np.log(p[k+1]/p[k])
      fac  = fac - a*((t[k]-t[k+1])/(p[k]-p[k+1])*(p[k]-p[k+1]))
      fac  = (Rp*Re)/(1+(z[k+1]/Rp/Re)) - fac
      z[k] = (Rp*Re/fac - 1)*Rp*Re
      grav[k] = grav0*(Rp*Re)**2/(Rp*Re + z[k])**2

    # iterate downward from ip
    for i in range(ip+1,Nlev):
      k    = i
      a    = kB/grav0/mm[k-1]
      fac  = a*(t[k]-(t[k-1]-t[k])/(p[k-1]-p[k]))*np.log(p[k]/p[k-1])
      fac  = fac - a*((t[k-1]-t[k])/(p[k-1]-p[k])*(p[k-1]-p[k]))
      fac  = (Rp*Re)/(1-(z[k-1]/Rp/Re)) - fac
      z[k] = -(Rp*Re/fac - 1)*Rp*Re
      grav[k] = grav0*(Rp*Re)**2/(Rp*Re + z[k])**2

  return z,grav
#
#
# mean molecular weight calculation
#
# inputs:
#
#        f    - gas mixing ratios, order is: Ar, CH4, CO2, H2, H2O, He, N2, O2, O3
#       fb    - background gas volume mixing ratio
#       mb    - background gas mean molar weight (g/mole)
#     mmw0    - gas molar weights (see set_gas_info)
#      mmr    - if true mixing ratios are mass mixing ratios; volume mixing ratios otherwise
#
# outputs:
#
#        m    - mean molecular weight (kg/molecule)
#
def mmw(f,fb,mb,mmw0,mmr):

  # constants
  Na    = 6.0221408e23   # avogradro's number

  # ordering is: Ar, CH4, CO2, H2, H2O, He, N2, O2, O3
  if mmr:
    de    = np.sum(f/mmw0[:,np.newaxis],axis=0) + fb/mb
    id    = np.copy(de)
    id[:] = 1
    m     = np.divide(id,de)/1.e3/Na
  else:
    id    = np.copy(fb)
    id[:] = 1
    m     = (np.sum(f*mmw0[:,np.newaxis],axis=0) + fb*mb)/1.e3/Na

  return m
#
#
# refractivity at STP
#
#   inputs:
#
#     f       -  mixing ratios of [Ar,CH4,CO2,H2O,N2,O2]
#   nu0       -  species refractivity at STP
#  mmw0       -  species mean molar weight (grams per mole)
#   mmr       -  indicates if mixing ratios are mass vs. volume
#     m       -  atmospheric mean molecular weight (kg per molec)
#
#   outputs:
#
#     nu0     -  refractivity at STP
#
def refractivity(f,nu0,mmw0,mmr,m):

  # constants
  Na    = 6.0221408e23   # avogradro's number

  # volume mixing ratio-weighted refractivity
  if mmr:
    fm  = np.divide(np.mean(f,axis=1),mmw0)*np.mean(m*Na*1.e3)
  else:
    fm  = np.mean(f,axis=1)
  nu  = np.sum(fm*nu0)

  return nu
#
#
# set vertical pressure grid
#
# inputs:
#
#       Nlev  - number of vertical levels
#       pmax  - top-of-model pressure
#       pmax  - bottom-of-model pressure
#        cld  - boolean to indicate if clouds are requested
#         pt  - cloud top pressure
#      tauc0  - cloud optical depth at user-provided wavelength
#        src  - model type (e.g., thermal, transit, diffuse reflectance)
#
# outputs:
#
#          p  - pressure grid
#
# notes:
#        all input pressures must have same units
#
def set_press_grid(Nlev,pmin,pmax,cld,pt,dpc,tauc0,src):

  Nlay = Nlev - 1

  # "small" cloud optical depth
  taus  = 0.22

  # large-enough optical depth that direct beam is small
  taul  = 2.0

  # large optical depth in opaque cloud portion
  #taull = 10.0

  # if doing a cloudy calc, accomodate cloud in grid
  if cld:

  # approach for diffuse rt solver is easier
    if (src != 'phas'):
      p     = np.logspace(np.log10(pmin),np.log10(pmax),Nlev)

      # force layers nearest to top/bottom of cloud to coincide with cloud
      it    = np.where(abs(p-pt) == min(abs(p-pt)))[0]
      if (it == Nlay):
        it = Nlay-1
      p[it] = pt
      pb    = pt + dpc
      ib    = np.where(abs(p-pb) == min(abs(p-pb)))[0]
      if (it == ib):
        ib  = it + 1
      p[ib] = pb

    else:
      # if cloud is thin, only need one layer
      if (tauc0 <= taus):

        Npc = 1

        # pressure grid from top of atmosphere to surface
        p0    = np.logspace(np.log10(pmin),np.log10(pmax),Nlev-Npc-1)
        if (pmin == pt):
          p0[0] = 0.5*(p0[0] + p0[1])
        if (pmax == pt + dpc):
          p0[-1] = 0.5*(p0[-1] + p0[-2])

        # pressure grid within cloud
        pc    = np.logspace(np.log10(pt),np.log10(pt+dpc),Npc+1)

        # combined pressure grid
        p     = np.unique(sorted(np.append(p0,pc)))

      # if cloud is thick, logic for avoiding sharp boundary
      else:

        # number of taus to reach optical depth large enough to extinct direct beam
        Npc = math.ceil(min(taul,tauc0)/taus)

        # ensure the upper grid doesn't extend beneath the cloud
        if (Npc*dpc*taus/tauc0 > pt + dpc):
          Npc = math.floor((pt + dpc)/(dpc*taus/tauc0))

        # pressure grid through cloud transition zone
        pc = pt + np.arange(Npc)*dpc*taus/tauc0

        # include cloud base
        pc = np.append(pc,pt+dpc)

        # pressure grid from top of atmosphere to surface
        p0    = np.logspace(np.log10(pmin),np.log10(pmax),Nlev-Npc-1)
        if (pmin == pt):
          p0[0] = 0.5*(p0[0] + p0[1])
        if (pmax == pt + dpc):
          p0[-1] = 0.5*(p0[-1] + p0[-2])

        # combined pressure grid
        p     = np.unique(sorted(np.append(p0,pc)))


  else: # otherwise, simple logarithmic gridpoint spacing

    p     = np.logspace(np.log10(pmin),np.log10(pmax),Nlev)

  return p
#
#
# function for reading ascii-formatted, space-separated data
#
# inputs:
#
#       fn    - filename
#       lskp  - header lines to skp
#
# outputs:
#
#       dat   - matrix of read-in data
#
def readdat(fn,lskp):

  f      = open(fn, 'r')
  lcount = 0
  for line in f: # count number of input data lines
    if (lcount == lskp+1):
      line = line.strip()
      columns = line.split()
      ndat = len(columns)     # number of input columns
    lcount += 1               # counts number of lines
  f.seek(0)                   # rewind file
  dat    = np.zeros([lcount-lskp,ndat]) # for storing data
  for i in range(lskp):       # read header
    hdr = f.readline()
  il = 0
  for line in f:
    line = line.strip()
    columns = line.split()
    for i in range(ndat):
      dat[il,i] = float(columns[i])
    il += 1
  f.close()

  return dat