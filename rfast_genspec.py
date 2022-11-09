# import statements
import time
import shutil
import os
import sys
import numpy             as np
import matplotlib.pyplot as plt
from astropy.table       import Table, Column, MaskedColumn
from astropy.io          import ascii
from scipy.interpolate   import interp1d
from rfast_routines      import gen_spec
from rfast_routines      import gen_spec_grid
from rfast_routines      import kernel_convol
from rfast_routines      import init
from rfast_routines      import init_3d
from rfast_routines      import inputs
from rfast_atm_routines  import set_gas_info
from rfast_atm_routines  import setup_atm
from rfast_opac_routines import init_cloud_optics

# get input script filename
if len(sys.argv) >= 2:
  filename_scr = sys.argv[1] # if script name provided at command line
else:
  filename_scr = input("rfast inputs script filename: ") # otherwise ask for filename

# obtain input parameters from script
fnr,fnn,fns,dirout,Nlev,pmin,pmax,bg,\
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
clr,fmin,mmr,nwalkers,nstep,nburn,thin,restart,progress = inputs(filename_scr)

# set info for all radiatively active gases, including background gas
Ngas,gasid,mmw0,ray0,nu0,mb,rayb = set_gas_info(bg)

# generate wavelength grids
Nres           = 3 # no. of res elements to extend beyond grid edges to avoid edge sensitivity (2--4)
lam,dlam       = gen_spec_grid(lams,laml,np.float_(res),Nres=0)
lam_hr,dlam_hr = gen_spec_grid(lams,laml,np.float_(res)*smpl,Nres=np.rint(Nres*smpl))

# initialize opacities and convolution kernels
sigma_interp,cia_interp,ncia,ciaid,kern = init(lam,dlam,lam_hr,species_l,species_c,opdir,pf,tf)

# initialize cloud asymmetry parameter, single scattering albedo, extinction efficiency
gc,wc,Qc = init_cloud_optics(lam_hr,g1,g2,g3,w,lamc0,grey,cld,opdir)
tauc     = tauc0*Qc

# initialize disk integration quantities
threeD   = init_3d(src,ntg)

# interpret mixing ratios as mass or volume, based on user input
mmr = False
if (imix == 1): mmr = True

# timing
tstart = time.time()

# initialize atmospheric model
#tstarts = time.time()
p,t,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                 t0,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                 species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                 mmr,mb,Mp,Rp,cld,pt,dpc,tauc0,p10,fp10,src,ref,nu0)
#print('Atmospheric setup timing (s): ',time.time()-tstarts)

# call forward model
#tstartg = time.time()
F1_hr,F2_hr = gen_spec(Nlev,Rp,a,As,em,p,t,t0,m,z,grav,Ts,Rs,ray,ray0,rayb,f,fb,
                       mmw0,mmr,ref,nu,alpha,threeD,
                       gasid,ncia,ciaid,species_l,species_c,
                       cld,sct,phfc,fc,pt,dpc,gc,wc,tauc,
                       src,sigma_interp,cia_interp,lam_hr,pf=pf,tf=tf)
#print('Spectral modeling timing (s): ',time.time()-tstartg)

# degrade resolution
F1   = kernel_convol(kern,F1_hr)
F2   = kernel_convol(kern,F2_hr)

# timing
tend = time.time()
print('Total setup and forward model timing (s), spectral points: ',tend-tstart,lam_hr.shape[0])

# write data file
if (src == 'diff' or src == 'cmbn'):
  names = ['wavelength (um)','d wavelength (um)','albedo','flux ratio']
if (src == 'thrm'):
  names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux (W/m**2/um)']
if (src == 'scnd'):
  names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux ratio']
if (src == 'trns'):
  names = ['wavelength (um)','d wavelength (um)','zeff (m)','transit depth']
if (src == 'phas'):
  names = ['wavelength (um)','d wavelength (um)','reflect','flux ratio']
data_out = Table([lam,dlam,F1,F2], names=names)
ascii.write(data_out,dirout+fns+'.raw',format='fixed_width',overwrite=True)

# document parameters to file
shutil.copy(filename_scr,dirout+fns+'.log')
if not os.path.isfile(dirout+filename_scr):
  shutil.copy(filename_scr,dirout+filename_scr)

# useful to indicate radius at p10
if (src == 'trns'):
  Re = 6.378e6 # Earth radius (m)
  r  = Rp + z/Re
  r_interp = interp1d(np.log(p),r,fill_value="extrapolate")
  print("Radius at p10 (Re): ",r_interp(np.log(p10)))

# plot raw spectrum
if (src == 'diff' or src == 'scnd' or src == 'cmbn' or src == 'phas'):
  ylab = 'Planet-to-Star Flux Ratio'
if (src == 'thrm'):
  ylab = r'Specific flux (W/m$^2$/${\rm \mu}$m)'
if (src == 'trns'):
  ylab = r'Transit depth'
plt.plot(lam, F2, drawstyle='steps-mid')
plt.ylabel(ylab)
plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
plt.savefig(dirout+fns+'.png',format='png',bbox_inches='tight')
plt.close()