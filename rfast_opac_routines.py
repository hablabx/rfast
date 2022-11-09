import lblabc_input
import numpy    as     np
from   scipy    import interpolate
from astropy.io import ascii
#
#
# routine to read-in cia data
#
#   inputs:
#
#   species   -  list of strings, choosing from 'co2','h2','n2','o2',
#              which will determine species included and ordering of species
#   lam       -  wavelength grid to place data on (um)
#    opdir    - directory where hi-res opacities are located (string)
#
#   outputs:
#
#      temp   -  temperature grid for cia data (K)
#       cia   -  cia coefficients [Ncases,Ntemp,Nwavelength] (m**-6 m**-1)
#
#   notes:
#
#   to add a new CIA instance, either increment the number of CIA 
#   cases (if gas already included) or add a new entry to ncia (if adding
#   a new gas). for the already-included gas scenario, go to the appropriate 
#   "if" check and repeat the logic for the new CIA file. for a wholly new 
#   species, and a new "elif" check and update the logic.
#
def cia_read(species,lam,opdir):

  # uniform temperature grid to interpolate onto
  temp = [100.,200.,300.,400.,500.,750.,1000.]
  temp = np.float_(temp)

  # number of cia cases associated with co2, h2, n2, o2
  ngas  = 4
  ncia0 = [1,2,1,3]
  gasn  = ['co2','h2','n2','o2']

  # gas and partner info (must agree with read-in order below)
  # first entry is the absorber and subsequent entries are partners
  ciaid0  = np.transpose(np.array([ ['     ']*(max(ncia0)+1) for i in range(ngas)]))
  ciaid0[0:2,0] = ['co2','co2']
  ciaid0[0:3,1] = ['h2','h2','he']
  ciaid0[0:2,2] = ['n2','n2']
  ciaid0[0:4,3] = ['o2','o2','n2','x']

  # determine number of cia cases
  icia    = 0
  ncia    = [0]*len(species)
  ciaid   = np.transpose(np.array([ ['     ']*(max(ncia0)+1) for i in range(len(species))]))
  for isp in range(len(species)):
    idg  = gasn.index(np.char.lower(species[isp]))
    ncia[isp] = ncia0[idg]
    ciaid[:,isp] = ciaid0[:,idg]
    icia = icia + ncia0[idg]

  # variable to store cia data
  kcia  = np.zeros([max(icia,1),len(temp),len(lam)])

  # loop over included species, interpolate onto uniform grid
  icia = 0
  for isp in range(len(species)):

    if (species[isp].lower() == 'co2'):

      # read data
      fn              = 'CO2-CO2_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)

      # interpolate onto lam
      cia_interp    = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)

      # update cia counter
      icia = icia + ncia0[0]

    elif (species[isp].lower() == 'h2'):

      # read data
      fn              = 'H2-H2_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)

      # interpolate onto lam
      cia_interp     = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)

      # read data
      fn              = 'H2-He_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia+1,:,:] = cia_interp(lam)

      # update cia counter
      icia = icia + ncia0[1]

    elif (species[isp].lower() == 'n2'):

      # read data
      fn              = 'N2-N2_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)

      # interpolate onto lam
      cia_interp     = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)

      # update cia counter
      icia = icia + ncia0[2]

    elif (species[isp].lower() == 'o2'):

      # read data
      fn              = 'O2-O2_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)

      # interpolate onto lam
      cia_interp     = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)

      # read data
      fn              = 'O2-O2_abs_Herzberg.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # copy onto temp (only one temperature point)
      cia0 = np.squeeze(cia0)
      cia0 = np.repeat(cia0[np.newaxis,:], len(temp), axis=0)

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:]   = kcia[icia,:,:] + cia_interp(lam)

      # read data
      fn              = 'O2-N2_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # copy onto temp (only one temperature point)
      cia0 = np.squeeze(cia0)
      cia0 = np.repeat(cia0[np.newaxis,:], len(temp), axis=0)

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia+1,:,:] = cia_interp(lam)

      # read data
      fn              = 'O2-X_abs.cia'
      temp0,lam0,cia0 = read_cia(opdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia+2,:,:] = cia_interp(lam)

      # update cia counter
      icia = icia + ncia0[3]

  # convert to m**-6 m**-1
  amg = 2.6867774e25
  kcia = kcia/(amg**2)*1.e2

  return temp,kcia,ncia,ciaid
#
#
# routine to read-in opacity database
#
#   inputs:
#
#   species   - list of strings, choosing from 'ch4','co2','h2o','o2','o3',
#                which will determine species included and ordering of species
#      lam    - desired output opacity wavelength grid (um)
#    opdir    - directory where hi-res opacities are located (string)
#
#   outputs:
#
#     press   -  opacity pressure grid (Pa)
#     sigma   -  opacities (m**2/molec) with size [Nspecies,Npressure,Nwavelength]
#
#   notes:
#
#   to add a new absorber, first ensure the relevant .abs file exists in the 
#   hires_opacities folder. add the species id/name to spc, add to mmw, indicate 
#   if a uv xsec file exists in the hires folder and the lines to skip at the top 
#   of this file.
#
def opacities_read(species,lam,opdir):

  form  = '_hitran2016_10_100000cm-1_1cm-1.abs'
  forx  = 'xsec.dat'
  fn    = [opdir + s.lower() + form for s in species]
  fnx   = [opdir + s.lower() + forx for s in species]
  Nspec = len(species) # number of species included

  # species names and mean molar weight, if they have xsec file & lines to skip in header
  spc   = ['ch4','co2','h2o','o2','o3','n2o','co']
  mmw   = [16.04,44.01,18.01,16.0,48.0,44.01,28.0]
  xsc   = [ 'y',  'y',  'y',  'y', 'y',  'y', 'y']
  lskp  = [  8,    8,    8,    8,   8 ,   8 ,  8 ]
  Na    = 6.0221408e23 # avogadro's number

  # small cross section (cm**2/molec) to replace all zeroes
  smallsig = 1e-40

  # fortran i/o id
  iuabc = 101

  # if there are >0 species included
  if (Nspec > 0):
    # initialize I/O variables for fortran LBLABC input - must agree with lblabc_input.f95
    mxp   = 20
    mxt   = 20
    mxwn  = 100000
    p     = np.zeros(mxp,order='F')
    tatm  = np.zeros([mxp,mxt],order='F')
    wn    = np.zeros(mxwn,order='F')
    abc   = np.zeros([mxp,mxt,mxwn],order='F')
    npg   = 0
    ntg   = 0
    nwn   = 0

    # read .abs file
    lblabc_input.open_lblabc(iuabc,fn[0])
    lblabc_input.read_lblabc(iuabc,npg,ntg,nwn,p,tatm,wn,abc)

    # number of p, T, lam points & associated grids
    press = p[np.nonzero(p)]
    Np    = len(press)
    temp  = tatm[0,0:Np]
    temp  = temp[np.nonzero(temp)]
    Nt    = len(temp)
    wn0   = wn[np.nonzero(wn)]      # cm**-1
    lam0  = 1.e4/wn[np.nonzero(wn)] # um
    kap   = np.squeeze(abc[0:Np,0:Nt,np.nonzero(wn)]) # cm**2/g

    # convert to cm**2/molec, remove <=0 values
    isp   = spc.index(species[0].lower())
    mw    = mmw[isp]
    sig   = kap*mw/Na # cm**2/molec
    sig   = np.nan_to_num(sig)
    sig[np.where(sig<=0)] = smallsig

    # initialize output opacities array
    sigma = np.zeros([Nspec,Np,Nt,len(lam)])

    # interpolate onto output wavelength grid
    sig_interp      = interpolate.interp1d(wn0[:-2],np.log10(sig[:,:,:-2]),axis=2,assume_sorted=True,fill_value="extrapolate")
    sigma[0,:,:,:]  = np.power(10,sig_interp(1.e4/lam))

    # read xsec file, if applicable
    if (xsc[isp] == 'y'):
      data  = ascii.read(fnx[0],data_start=lskp[isp])
      lamx  = data['col1']
      xsecx = data['col2'] # cm**2/molecule
      xsecx[np.where(xsecx<=0)] = smallsig
      xsec_interp = interpolate.interp1d(lamx,np.log10(xsecx),assume_sorted=True,fill_value="extrapolate")
      xsec        = np.power(10,xsec_interp(lam))
      xsec[np.where(xsec<smallsig)] = smallsig
      xsec        = np.repeat(xsec[np.newaxis,:],   sigma.shape[1], axis=0)
      xsec        = np.repeat(xsec[:,np.newaxis,:], sigma.shape[2], axis=1)
      sigma[0,:,:,:] = sigma[0,:,:,:] + xsec

    # loop over other species and store opacities
    for j in range(1,Nspec):
      # re-initialize storage arrays
      p     = np.zeros(mxp,order='F')
      tatm  = np.zeros([mxp,mxt],order='F')
      wn    = np.zeros(mxwn,order='F')
      abc   = np.zeros([mxp,mxt,mxwn],order='F')
      npg   = 0
      ntg   = 0
      nwn   = 0

      # read .abs file
      lblabc_input.open_lblabc(iuabc,fn[j])
      lblabc_input.read_lblabc(iuabc,npg,ntg,nwn,p,tatm,wn,abc)

      # wavelength grid, opacity
      wn0   = wn[np.nonzero(wn)]      # cm**-1
      lam0  = 1.e4/wn[np.nonzero(wn)] # um
      kap   = np.squeeze(abc[0:Np,0:Nt,np.nonzero(wn)]) # cm**2/g

      # convert to cm**2/molec, remove <=0 values
      isp   = spc.index(species[j].lower())
      mw    = mmw[isp]
      sig   = kap*mw/Na # cm**2/molec
      sig   = np.nan_to_num(sig)
      sig[np.where(sig<=0)] = smallsig

      # interpolate onto output wavelength grid
      sig_interp      = interpolate.interp1d(wn0[:-2],np.log10(sig[:,:,:-2]),axis=2,assume_sorted=True,fill_value="extrapolate")
      sigma[j,:,:,:]  = np.power(10,sig_interp(1.e4/lam))

      # read xsec file, if applicable
      if (xsc[isp] == 'y'):
        data  = ascii.read(fnx[j],data_start=lskp[isp])
        lamx  = data['col1']
        xsecx = data['col2'] # cm**2/molecule
        xsecx[np.where(xsecx<=0)] = smallsig
        xsec_interp = interpolate.interp1d(lamx,np.log10(xsecx),assume_sorted=True,fill_value="extrapolate")
        xsec        = np.power(10,xsec_interp(lam))
        xsec[np.where(xsec<smallsig)] = smallsig
        xsec        = np.repeat(xsec[np.newaxis,:],   sigma.shape[1], axis=0)
        xsec        = np.repeat(xsec[:,np.newaxis,:], sigma.shape[2], axis=1)
        sigma[j,:,:,:] = sigma[j,:,:,:] + xsec

  else: # case with no species -> zero opacities across all press, temp
    press = np.float32([1.,1.e7])
    temp  = np.float32([50.,650.])
    Np    = len(press)
    Nt    = len(temp)
    sigma = np.zeros([1,Np,Nt,len(lam)])
    sigma[:,:,:,:] = smallsig

  # convert to m**2/molecule
  sigma = sigma*1.e-4

  return press,temp,sigma
#
#
# rayleigh scattering cross section (m**2/molecule)
#
#   inputs:
#
#     lam     -  wavelength (um)
#     ray0    -  xsec (m**2/molec) at 0.4579 (see set_gas_info)
#     f       -  gas vmr profiles (ordered as set_gas_info)
#     fb      -  background gas vmr
#     rayb    -  background gas rayleigh cross section relative to Ar
#
#   outputs:
#
#     sigma   -  cross section (m**2/molecule) at each lam point
#
def rayleigh(lam,ray0,f,fb,rayb):

  # cross section at all wavelengths
  sigma = np.multiply.outer(np.sum(f*ray0[:,np.newaxis],axis=0),(0.4579/lam)**4.)
  sigma = sigma + np.multiply.outer(fb*rayb*ray0[0,np.newaxis],(0.4579/lam)**4.)

  return sigma
#
#
# cloud optical properties routine
#
#   inputs:
#
#     lam     -  wavelength (um)
#     g1      -  grey cloud asymmetry parameter (first phase function moment), if grey
#     g2      -  grey cloud second phase function moment
#     g3      -  grey cloud third phase function moment
#     w       -  grey cloud single scattering albedo, if grey
#     lamc0   -  wavelength where cloud extinction optical depth is normalized to
#     grey    -  boolean, assume grey cloud if true
#    opdir    - directory where hi-res opacities are located (string)
#
#   outputs:
#
#     gc      -   cloud asymmetry parameter (len(lam))
#     wc      -   cloud single scattering albedo (len(lam))
#     Qc      -   cloud extinction efficiency, normalized at lamc
#
def init_cloud_optics(lam,g1,g2,g3,w,lamc0,grey,cld,opdir):

  # if doing clouds
  if cld:
    # if grey, set properties as constant
    if grey:
      gc1 = np.zeros(len(lam))
      gc2 = np.zeros(len(lam))
      gc3 = np.zeros(len(lam))
      wc  = np.zeros(len(lam))
      Qc  = np.zeros(len(lam))
      gc1[:] = g1
      gc2[:] = g2
      gc3[:] = g3
      wc[:]  = w
      Qc[:]  = 1
    else:
      # liquid
      data     = ascii.read(opdir+'strato_cum.mie',data_start=20,delimiter=' ')
      lam_in   = data['col1']
      w_in     = data['col10']
      g_in     = data['col11']
      q_in     = data['col7']
      w_interp = interpolate.interp1d(lam_in,w_in,assume_sorted=True,fill_value="extrapolate")
      g_interp = interpolate.interp1d(lam_in,g_in,assume_sorted=True,fill_value="extrapolate")
      q_interp = interpolate.interp1d(lam_in,q_in,assume_sorted=True,fill_value="extrapolate")
      wcl      = w_interp(lam)
      gcl      = g_interp(lam)
      qcl      = q_interp(lam)/q_interp(lamc0)
      # ice
      data     = ascii.read(opdir+'baum_cirrus_de100.mie',data_start=2,delimiter=' ')
      lam_in   = data['wl']
      w_in     = data['omega']
      g_in     = data['g']
      q_in     = data['Qe']
      w_interp = interpolate.interp1d(lam_in,w_in,assume_sorted=True,fill_value="extrapolate")
      g_interp = interpolate.interp1d(lam_in,g_in,assume_sorted=True,fill_value="extrapolate")
      q_interp = interpolate.interp1d(lam_in,q_in,assume_sorted=True,fill_value="extrapolate")
      wci      = w_interp(lam)
      gci      = g_interp(lam)
      qci      = q_interp(lam)/q_interp(lamc0)
      # 50/50 mixture
      f      = 0.5
      wc     = f*wcl + (1-f)*wci
      Qc     = f*qcl + (1-f)*qci
      gc     = np.zeros(len(lam))
      gc1    = f*gcl + (1-f)*gci
      gc2    = np.zeros(len(lam))
      gc2[:] = g2
      gc3    = np.zeros(len(lam))
      gc3[:] = g3
  else: # no cloud case, zero some stuff out
    gc1 = np.zeros(len(lam))
    gc2 = np.zeros(len(lam))
    gc3 = np.zeros(len(lam))
    wc   = np.zeros(len(lam))
    Qc   = np.zeros(len(lam))

  # package together first and second moments
  gc = gc1,gc2,gc3

  return gc,wc,Qc
#
#
# read a *_abs.cia file
#
#   inputs:
#
#   fn        -  filename
#
#   outputs:
#
#        t    -  temperature grid for cia data (K)
#      cia    -  cia coefficients (cm**-1 amagat**-2)
#
def read_cia(fn):

  # open and read data file
  data  = ascii.read(fn,data_start=0,delimiter=' ')

  # number of temperature, wavelength points
  Ntemp = int(data['col1'][0])
  Nlam  = len(data['col1'][1:-1])

  # variables for storing temperature, cia
  temp  = np.zeros(Ntemp)
  lam   = np.zeros(Nlam)
  cia   = np.zeros([Ntemp,Nlam])

  # store wavelength points
  lam[:] = 1.e4/data['col1'][1:-1]

  # loop over temperature points and store data
  for i in range(Ntemp):
    temp[i]  = data['col'+str(i+2)][0]
    cia[i,:] = data['col'+str(i+2)][1:-1]

  return temp,lam,cia