
import numpy as np
import itertools
import sys
import time
import matplotlib.pyplot as plt


def enumerate_progress(list, label=''):
    """Simple progress bar.

    """
    t0 = time.time()
    ni = len(list)
    for i, v in enumerate(list):
        yield i, v
        ppct = int(100. * (i - 1) / ni)
        cpct = int(100. * (i + 0) / ni)
        if cpct > ppct:
            dt = time.time() - t0
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            sys.stdout.write("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                             label + " " + int(10. * cpct / 100) * "-" + "> " + ("%02d" % cpct) + r"%")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_cov(ells, dell, cls_fid, cl_keys, sig_eps, ng_i, fsky = 0.363609919):
    cov_tot = []
    invcov_l = []
    for il, l in enumerate(ells):
        cl_dict_l = {}
        nl_dict = {}

        #TODO : Add noise 
        for key in cl_keys:
            cl_dict_l[key] = cls_fid[key][l]
            
            nl_dict[key] = 0
            if (key[:2] == key[-2:]):
                nl_dict[key] = sig_eps**2/ng_i
        ncl = len(cl_keys)
        cov_l = np.zeros([ncl, ncl])

        # Get symetric Cls for permutations
        for key in cl_keys:
            probeA, probeB = key.split('x')
            cl_dict_l['x'.join([probeB, probeA])] = cl_dict_l[key]
            nl_dict['x'.join([probeB, probeA])] = nl_dict[key]

        for (idx1, key1), (idx2, key2) in itertools.combinations_with_replacement(enumerate(cl_keys), 2):
            # print(key1, key2)
            probeA, probeB = key1.split('x')
            probeC, probeD = key2.split('x')
    #         if il == 0:
    #             print(probeA, probeB, probeC, probeD)
            cov_l[idx1, idx2] = 1. / (fsky * (2. * l + 1.) * dell[il]) \
                * (
                    (cl_dict_l['x'.join([probeA, probeC])] + nl_dict['x'.join([probeA, probeC])]) *
                    (cl_dict_l['x'.join([probeB, probeD])] + nl_dict['x'.join([probeB, probeD])]) +
                    (cl_dict_l['x'.join([probeA, probeD])] + nl_dict['x'.join([probeA, probeD])]) *
                    (cl_dict_l['x'.join([probeB, probeC])] + nl_dict['x'.join([probeB, probeC])])
            )

        # Get the symmetric matrix
        cov_l = cov_l + cov_l.T - np.diag(cov_l.diagonal())
        cov_tot.append(cov_l)
        invcov_l.append(np.linalg.inv(cov_l))
    return cov_tot, invcov_l


def plot_pn(ells, cl, ax=None, **kwargs):
    if ax is None: ax = plt.gca()
    is_pos = np.where(cl[ells]>0)
    is_neg =np.where(cl[ells]<0)
    p = ax.plot(ells[is_pos], cl[ells][is_pos],  **kwargs)
    ax.plot(ells[is_neg], -cl[ells][is_neg], ls='--', color=p[0].get_color())


def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def get_fisher(ells, params, cls_dpar_p, cls_dpar_m, delta_par, invcov_l, cl_keys):
    print(cl_keys)
    npar = len(params)
    dcls_dpar = {}
    for par in params:
        dcls_dpar[par] = {}
        for il, ell in enumerate(ells):
            dcls_dpar[par][ell] = np.zeros(len(cl_keys))
            for ik, key in enumerate(cl_keys):
                dcls_dpar[par][ell][ik] = (cls_dpar_p[par][key][ell] - cls_dpar_m[par][key][ell])/(delta_par[par])
    
    fisher = np.zeros([npar, npar])
    # invcov_w = like_euclid_wl.invcovmat_l
#     print(cl_keys)
    for (i1, par1), (i2, par2) in itertools.combinations_with_replacement(enumerate(params), 2):
        for il, l in enumerate(ells):
            fisher[i1, i2] += np.dot(np.dot(dcls_dpar[par1][ell], invcov_l[il]), dcls_dpar[par2][ell])  
    fisher = fisher + fisher.T - np.diag(fisher.diagonal())    
    return fisher, dcls_dpar