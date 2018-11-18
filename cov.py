# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# amalie
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###############################################################################
# MODULES
###############################################################################
import os
import numpy as np
from astropy.table import Table, Column, join, vstack
import astropy.units as u
import astropy.coordinates as coord
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 as mw
from galpy.actionAngle import actionAngleStaeckel


# Don't print Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)

# Make some plot settings
sns.set_palette('colorblind')
sns.set_color_codes('colorblind')
sns.set_style({"xtick.direction": "inout","ytick.direction": "inout",
               "xtick.top": True, "ytick.right": True})

###############################################################################
# FUNCTIONS
###############################################################################
# Convert the angles
def convert_angles(angles):
    angles /= (1000. * 3600.)
    return angles


def make_sigmamatrix(ra_e, dec_e, plx_e, pmra_e, pmdec_e, rv_e,
        ra_dec_cor, ra_plx_cor, ra_pmra_cor, ra_pmdec_cor,
        dec_plx_cor, dec_pmra_cor, dec_pmdec_cor, plx_pmra_cor,
        plx_pmdec_cor, pmra_pmdec_cor):

    ra_e = convert_angles(ra_e)
    dec_e = convert_angles(dec_e)

    # Build entries
    s00 = ra_e ** 2
    s11 = dec_e ** 2
    s22 = plx_e ** 2
    s33 = pmra_e ** 2
    s44 = pmdec_e ** 2
    s55 = rv_e ** 2

    s01 = ra_e * dec_e * ra_dec_cor
    s02 = ra_e * plx_e * ra_plx_cor
    s03 = ra_e * pmra_e * ra_pmra_cor 
    s04 = ra_e * pmdec_e * ra_pmdec_cor
    s05 = 0

    s12 = dec_e * plx_e * dec_plx_cor
    s13 = dec_e * pmra_e * dec_pmra_cor
    s14 = dec_e * pmdec_e * dec_pmdec_cor
    s15 = 0

    s23 = plx_e * pmra_e * plx_pmra_cor
    s24 = plx_e * pmdec_e * plx_pmdec_cor
    s25 = 0

    s34 = pmra_e * pmdec_e * pmra_pmdec_cor
    s35 = 0

    s45 = 0

    sigma = np.array([
        [s00, s01, s02, s03, s04, s05],
        [s01, s11, s12, s13, s14, s15],
        [s02, s12, s22, s23, s24, s25],
        [s03, s13, s23, s33, s34, s35],
        [s04, s14, s24, s34, s44, s45],
        [s05, s15, s25, s35, s45, s55]
        ])
    return sigma


def sample_gaiaphase(mu, ra_e, dec_e, plx_e, pmra_e, pmdec_e, rv_e,
        ra_dec_cor, ra_plx_cor, ra_pmra_cor, ra_pmdec_cor,
        dec_plx_cor, dec_pmra_cor, dec_pmdec_cor, plx_pmra_cor,
        plx_pmdec_cor, pmra_pmdec_cor, n_sample=10000):
    # Get sigma
    sigma = make_sigmamatrix(ra_e, dec_e, plx_e, pmra_e, pmdec_e, rv_e,
            ra_dec_cor, ra_plx_cor, ra_pmra_cor, ra_pmdec_cor,
            dec_plx_cor, dec_pmra_cor, dec_pmdec_cor, plx_pmra_cor,
            plx_pmdec_cor, pmra_pmdec_cor)

    # Sample from distibution
    sample = np.random.multivariate_normal(mu, sigma, size=n_sample)
    return sample


# Read data
kittenfilename = 'testsmallredgiantcat.vot'
fillvalue = -9999.0
n_samples = 10000
redgiantcat = Table.read(kittenfilename, format='votable')

# Unpack data for this example
ra = float(redgiantcat['RA_GAIA'].data.flatten())
dec = float(redgiantcat['DEC_GAIA'].data.flatten())
ra_e = float(redgiantcat['ERROR_RA_GAIA'].data.flatten())
dec_e = float(redgiantcat['ERROR_DEC_GAIA'].data.flatten())
plx = float(redgiantcat['PARALLAX_GAIA'].data.flatten())
plx_e = float(redgiantcat['ERROR_PARALLAX_GAIA'].data.flatten())
pmra = float(redgiantcat['PMRA_GAIA'].data.flatten())
pmra_e = float(redgiantcat['ERROR_PMRA_GAIA'].data.flatten())
pmdec = float(redgiantcat['PMDEC_GAIA'].data.flatten())
pmdec_e = float(redgiantcat['ERROR_PMDEC_GAIA'].data.flatten())
ra_dec_cor = float(redgiantcat['RA_DEC_CORR_GAIA'].data.flatten())
ra_plx_cor = float(redgiantcat['RA_PARALLAX_CORR_GAIA'].data.flatten())
ra_pmra_cor = float(redgiantcat['RA_PMRA_CORR_GAIA'].data.flatten())
ra_pmdec_cor = float(redgiantcat['RA_PMDEC_CORR_GAIA'].data.flatten())
dec_plx_cor = float(redgiantcat['DEC_PARALLAX_CORR_GAIA'].data.flatten())
dec_pmra_cor = float(redgiantcat['DEC_PMRA_CORR_GAIA'].data.flatten())
dec_pmdec_cor = float(redgiantcat['DEC_PMDEC_CORR_GAIA'].data.flatten())
plx_pmra_cor = float(redgiantcat['PARALLAX_PMRA_CORR_GAIA'].data.flatten())
plx_pmdec_cor = float(redgiantcat['PARALLAX_PMDEC_CORR_GAIA'].data.flatten())
pmra_pmdec_cor = float(redgiantcat['PMRA_PMDEC_CORR_GAIA'].data.flatten())
rv = float(redgiantcat['RADIAL_VELOCITY_GAIA'].data.flatten())
rv_e = float(redgiantcat['ERROR_RADIAL_VELOCITY_GAIA'].data.flatten())

mu = np.array([ra, dec, plx, pmra, pmdec, rv]).flatten()

sample = sample_gaiaphase(mu, ra_e, dec_e, plx_e, pmra_e, pmdec_e, rv_e,
            ra_dec_cor, ra_plx_cor, ra_pmra_cor, ra_pmdec_cor,
            dec_plx_cor, dec_pmra_cor, dec_pmdec_cor, plx_pmra_cor,
            plx_pmdec_cor, pmra_pmdec_cor)
sample = sample[sample[:, 2] > 0]

# Plot sample
pp = PdfPages('hist.pdf')
fig, axes = plt.subplots(2, 3, sharey=True)
fig.tight_layout()

axes[0, 0].hist(sample[:, 0], bins='fd')
axes[0, 0].set_xlabel('Ra')

axes[0, 1].hist(sample[:, 1], bins='fd')
axes[0, 1].set_xlabel('Dec')

axes[0, 2].hist(sample[:, 2], bins='fd')
axes[0, 2].set_xlabel('Parallax');

axes[1, 0].hist(sample[:, 3], bins='fd')
axes[1, 0].set_xlabel('Pmra');
axes[1, 1].hist(sample[:, 4], bins='fd')
axes[1, 1].set_xlabel('Pmdec');
axes[1, 2].hist(sample[:, 5], bins='fd')
axes[1, 2].set_xlabel('RV');
pp.savefig(bbox_inches='tight')
pp.close()

assert np.all(sample[:, 2] > 0)
# Calculate dynamics
# This follows the tutorial at:
# http://dfm.io/astropy/coordinates/apply_space_motion.html
# v_sun from Schonrich et al 2010
print('Get dynamics')
v_sun = coord.CartesianDifferential([11.1, 240+12.24, 7.25] * u.km/u.s)
# NB: Here I just use the inverse parallax as dist
cs = coord.SkyCoord(ra=sample[:, 0] * u.deg,
        dec=sample[:, 1] * u.deg,
        distance=coord.Distance(parallax=sample[:, 2] * u.mas),
        pm_ra_cosdec=sample[:, 3] * u.mas / u.yr,
        pm_dec=sample[:, 4] * u.mas / u.yr,
        radial_velocity=sample[:, 5] * u.km / u.s,
        galcen_distance=8.34*u.kpc,  # Reid et al 2014
        galcen_v_sun=v_sun,  # 240 is from Reid et al 2014
        z_sun=27*u.pc)  # Default, Chen et al 2001

# Define Galactocentric frame
gc = coord.Galactocentric(galcen_distance=8.34*u.kpc,  # Reid et al 2014
        galcen_v_sun=v_sun,  # 240 is from Reid etal 2014
        z_sun=27*u.pc)  # Default, Chen et al 2001

aAS = actionAngleStaeckel(
        pot=pot,
        delta=0.45,
        c=False
        )

cs = cs.transform_to(gc)
cs.representation_type = 'cylindrical'
rho = cs.rho  # pc
phi = cs.phi  # pc
z = cs.z  # pc
for i, c in enumerate(cs):
    o = Orbit(c) 
    Jr = o.jr()
    Lz = o.jp()

# Plot sample
pp = PdfPages('velocityhist.pdf')
fig, axes = plt.subplots(2, 2, sharey=True)
fig.tight_layout()

axes[0, 0].hist(rho, bins='fd')
axes[0, 0].set_xlabel('rho (pc)')

axes[0, 1].hist(z, bins='fd')
axes[0, 1].set_xlabel('z (pc)')

axes[1, 0].hist(Lz, bins='fd')
axes[1, 0].set_xlabel('Lz');

axes[1, 1].hist(Jr, bins='fd')
axes[1, 1].set_xlabel(r'J_r');

fig.xlabel('RUWE')
fig.ylabel('Density')
pp.savefig(bbox_inches='tight')
pp.close()

