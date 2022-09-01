import numpy as np

ffp10_plancklens = {
    'label': 'ffp10_plancklens',
#     'Alens':1,
    'As': 2.119631e-09,
    'logA': np.log(2.119631e-09 *1e10),
    'ns': 0.9636852,
    # 'H0': 67.01904,
    # 'theta_MC_100':0.010407224216796775 *100,
    'cosmomc_theta':0.010407224216796775,
    'ombh2': 0.02216571,
    'omch2': 0.1202944,
    'tau':0.06018107,
    'mnu':0.06,
    'omk':0,
    'w':-1,
#     'wa':0
}

# fiducial_ffp10_plancklens = {
#     'logA':np.log(2.119631e-09 *1e10),
#     'ns': 0.9636852,
#     'H0': 67.01904,
#     'theta_MC_100':0.010407224216796775 *100,
#     'ombh2': 0.02216571,
#     'omch2': 0.1202944,
#     'tau':0.06018107,
#     'mnu':0.06
# }