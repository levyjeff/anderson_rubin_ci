"""
Copyright (C) 2005 Jeffrey A Levy

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

def anderson_rubin_ci(model, conflevel=0.95):
    """
    Implemented based on the ipack library from R
    https://github.com/cran/ivpack/blob/master/R/anderson.rubin.ci.R

    Arguments:
    model -- a fit Statsmodels results container (required)
    conflevel -- the confidence level to calculate (default 0.95)

    Returns: Tuple of length two, bottom value followed by top value. However,
             note that tuples with numeric results are not guaranteed, and may 
             be a string or inf in some cases.
    """
    y1 = model.model.endog
    n = len(y1)
    regressor_cols = model.model.exog_names
    instrument_cols = model.model.instrument_names
    regressors = model.model.exog
    W = model.model.instrument.to_numpy()

    y2 = regressors[:,[c not in instrument_cols for c in regressor_cols]]
    Z = regressors[:,[c in instrument_cols for c in regressor_cols]]

    l = W.shape[1]
    k = Z.shape[1]
    q = f.ppf(conflevel, l-k, n-l)
    cval = q * (l-k) / (n-l)

    y2hat_given_w = sm.OLS(endog=y2, exog=W).fit().fittedvalues
    y2hat_given_z = sm.OLS(endog=y2, exog=Z).fit().fittedvalues
    y1hat_given_w = sm.OLS(endog=y1, exog=W).fit().fittedvalues
    y1hat_given_z = sm.OLS(endog=y1, exog=Z).fit().fittedvalues

    coef_beta0sq = (cval * sum(y2**2) - (cval+1) * sum(y2.flatten()*y2hat_given_w) + sum(y2.flatten() * y2hat_given_z))[0]
    coef_beta0 = -2 * cval * sum(y1.flatten() * y2.flatten()) + 2*(cval+1)*sum(y1.flatten()*y2hat_given_w) - 2*sum(y1.flatten()*y2hat_given_z)
    coef_constant = cval * sum(y1.flatten()**2) - (cval+1) * sum(y1.flatten() * y1hat_given_w) + sum(y1.flatten() * y1hat_given_z)
    D = coef_beta0**2 - 4*coef_constant * coef_beta0sq

    if coef_beta0sq == 0:
        if coef_beta0 > 0:
            ci = [-coef_constant/coef_beta0, np.inf]
        elif coef_beta0 < 0:
            ci = [-np.inf, -coef_constant/coef_beta0]
        elif coef_beta0 == 0:
            if coef_constant >= 0:
                ci = "Whole Real Line"
            else:
                ci = "Empty Set"
    else:
        if D > 0:
            #roots of quadratic equation
            root1 = (-coef_beta0 + np.sqrt(D)) / (2*coef_beta0sq)
            root2 = (-coef_beta0 - np.sqrt(D)) / (2*coef_beta0sq)
            upper_root = max(root1, root2)
            lower_root = min(root1, root2)
            if coef_beta0sq < 0:
                ci = [lower_root, upper_root]
            elif coef_beta0sq > 0:
                ci = f"[-Infinity, {lower_root}] union [{upper_root}, Infinity]"
        elif D == 0:
            ci = "Whole Real Line"
    return ci

if __name__ == '__main__':
    #Test
    from statsmodels.sandbox.regression.gmm import IV2SLS
    df = pd.read_csv(r'data\card_data.csv')
    
    endog = df['lwage']
    exog = df[['educ', 'exper','expersq','black','south','smsa','reg661','reg662',
               'reg663','reg664','reg665','reg666','reg667','reg668','smsa66']]
    instrument = df[['nearc4', 'exper','expersq','black','south','smsa','reg661',
                     'reg662','reg663','reg664','reg665','reg666','reg667','reg668',
                     'smsa66']]
    exog = sm.add_constant(exog)
    instrument = sm.add_constant(instrument)

    model = IV2SLS(endog, exog, instrument)
    results = model.fit()
    ar_cis = anderson_rubin_ci(results)

    # Compare to values from "Using Geographic Variation in College Proximity to
    # Estimate the Return from Schooling, Card (1995), Table 3, Panel A, column (5)"
    assert(round(results.params['educ'], 3) == 0.132)
    assert(round(results.bse['educ'], 3) == 0.055)
    # Compare to values from ivpack test 
    # (https://rdrr.io/cran/ivpack/man/anderson.rubin.ci.html):
    assert(round(ar_cis[0], 8) == 0.02480484)
    assert(round(ar_cis[1], 8) == 0.28482359)

    