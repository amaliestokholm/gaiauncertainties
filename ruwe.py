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
import scipy.interpolate


# Don't print Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)

# Make some plot settings
sns.set_palette('colorblind')
sns.set_color_codes('colorblind')
sns.set_style({"xtick.direction": "inout","ytick.direction": "inout",
               "xtick.top": True, "ytick.right": True})

# Read data
#kittenfilename = 'testsmallredgiantcat.vot'
kittenfilename = 'gestalt_apogeerv_BASTA.vot'
fillvalue = -9999.0
redgiantcat = Table.read(kittenfilename, format='votable')
ruwefile = 'ruwe.txt'
gmag = redgiantcat['PHOT_G_MEAN_MAG_GAIA'].data
bprp = redgiantcat['BP_RP_GAIA'].data
chi2 = redgiantcat['ASTROMETRIC_CHI2_AL_GAIA'].data 
ngoodobs = redgiantcat['ASTROMETRIC_N_GOOD_OBS_AL_GAIA'].data 
print('gmag', gmag)
print('bp-rp', bprp)
print('chi2', chi2)
print('ngoodobs', ngoodobs)

def compute_tdtable(u0table):
    # Compute u0 using With colors and gmag
    for i, col in enumerate(u0table.columns[2:]):
        if i == 0:
            tdtable = u0table[col].data.reshape(len(u0table[col].data), 1)
        else:
            coldata = u0table[col].data.reshape(len(u0table[col].data), 1)
            tdtable = np.concatenate((tdtable, coldata), axis=1)
    tdtable = tdtable.T
    return tdtable


def compute_ruwe(gmag, bprp, chi2, ngoodobs, tdtable, u0table):
    if (bprp > -1.0) & (bprp < 10.0):
        # Read table from DR2_RUWE_V1.zip
        c = np.arange(-1.0, 10+0.1, 0.1)

        m = c.size
        n = u0table['g_mag'].data.size
        assert tdtable.shape == (m, n), (m, n)

        f = scipy.interpolate.interp2d(u0table['g_mag'].data, c, tdtable, kind='linear')
        u0 = f(gmag, bprp)[0]
        #print('using color and gmag u0=', u0)
    else:
        #print('bp-rp is outside the grid!')
        if (np.isfinite(gmag)) & (gmag > 3.6) & (gmag < 21.0):
            #print('u0 is computed using only gmag')

            # Read table from DR2_RUWE_V1.zip
            f = scipy.interpolate.interp1d(u0table['g_mag'].data, u0table['u0g'].data)
            u0 = f(gmag)
            #print('using only gmag u0=', f(gmag))
        else:
            #print('gmag is outside the grid!')
            return fillvalue

    # Compute RUWE
    u = np.sqrt(chi2 / (ngoodobs - 5))
    ruwe = u / u0
    #print('RUWE=', ruwe)
    return ruwe


if os.path.isfile(ruwefile):
    print('Load ruwefile')
    ruwe = np.loadtxt(ruwefile)
    ruwecol = Column(ruwe, name='RUWE_GAIA')
    redgiantcat.add_column(ruwecol)
else:
    print('Compute new ruwefile')
    mask = (bprp > -1.0) & (bprp < 10.0)
    ruwe = np.zeros(len(gmag[mask]))
    u0table = Table.read('table_u0_2D.txt', format='ascii')
    tdtable = compute_tdtable(u0table)
    for i, (gmag, bprp, chi2, ngoodobs) in enumerate(zip(gmag[mask], bprp[mask], chi2[mask], ngoodobs[mask])):
        ruwe[i] = compute_ruwe(gmag, bprp, chi2, ngoodobs, tdtable, u0table)
    print(len(redgiantcat), len(redgiantcat[mask]))
    ruwecol = Column(ruwe, name='RUWE_GAIA')
    redgiantcat.add_column(ruwecol)
    np.savetxt(ruwefile, ruwe)

pp = PdfPages('ruwehist.pdf')
fig, axes = plt.subplots(1, 2, sharey=True)
fig.tight_layout()
axes[0].hist(ruwe, bins='fd')
axes[0].set_xlim([-0.1, 2])
axes[1].hist(ruwe, bins='fd')
axes[0].set_xlabel('RUWE')
axes[1].set_xlabel('RUWE')
axes[0].set_ylabel('Density')
pp.savefig(bbox_inches='tight')
pp.close()

pp = PdfPages('colormag.pdf')
fig, axes = plt.subplots(1, 2)
ruwecut = (ruwe < 1.4)
axes[0].scatter(bprp[ruwecut], gmag[ruwecut] - 5 * np.log10(redgiantcat['DISTANCE_BASTA'].data[ruwecut] / 10), s=1, alpha=0.1, cmap='viridis')
axes[0].set_xlabel('bp-rp')
axes[0].set_ylabel('G mag')
axes[0].invert_yaxis()
axes[1].hist(ruwe, bins='fd')
axes[1].axvline(x=1.4, linestyle='--', color='0.6')
axes[1].set_xlim([-0.1, 2])
axes[1].set_xlabel('RUWE')
axes[1].set_ylabel('Density')
pp.savefig(bbox_inches='tight')
pp.close()
