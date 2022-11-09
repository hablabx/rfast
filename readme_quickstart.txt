The rfast package includes software to generate synthetic spectra of worlds (via the 
rfast_gendata.py routine) and software to retrieve on spectral observations, be they 
real or fake (via rfast_retrieve.py). Nearly all aspects of these utilities are managed 
via the rfast_inputs.scr runscript.

To get started, a user must compile a limited amount of Fortran software that is required 
for rfast to ingest tables of opacities stored as binary. Within the downloaded (and 
intact) rfast folder, enter at the command line:

  python -m numpy.f2py -c lblabc_input.f95 -m lblabc_input

This command uses the f2py python package to create a Python module version of the 
lblabc_input.f95 routine.

Following successful creation of the lblabc_input module, the forward model is run with:

  python rfast_genspec.py rfast_inputs.scr

This should create a simulated reflected-light spectrum of Earth, stored in a ".raw" file.

To add instrument noise, do:

  python rfast_noise.py rfast_inputs.scr

This creates a ".dat" file that, using the "fnn" extension parameter in the runscript, has
the name "earth_refl_demo_snr20.dat".

Retrievals are managed, in part, by the rfast_rpars.txt file. This file indicates every 
parameter that can be retrieved on. For each parameter, the user can toggle on/off whether
a parameter is retrieved, if it is retrieved in log versus linear space, if a Gaussian or 
flat prior is adopted, and parameters that define the prior. For a flat prior, "prior 1" 
is the parameter lower limit and "prior 2" is the parameter upper limit. For a Gaussian 
prior, "prior 1" is the central value of the Gaussian and "prior 2" is the 1-sigma width 
of the Gaussian. Units are the same as in rfast_inputs.

To set up a retrieval, and after adjusting rfast_rpars to your needs, simply do:

  python rfast_initret.py rfast_inputs.scr rfast_rpars.txt

Which creates files rfast_retrieve.py and rfast_analyze.py from templates. Now, to execute
your retrieval, do:

  python rfast_retrieve.py rfast_inputs.scr

This should create a .h5 file that stores the results of the spectral retrieval.

Rudimentary analyses of the emcee results (stored in the .h5 file) is provided with:

  python rfast_analyze.py rfast_inputs.scr