"""
Calculate energy balance component as a function of met. forcing
"""

import collections

import numpy as np

# PARAMETERS (provided)
g = 9.81
rho_i = 910
rho_w = 1000
rho_0 = 1.29
P_0 = 101.325e3
c_p = 1.005e3
c_w = 4.180e3
L_f = 3.34e5
L_v = 2.514e6
L_s = 2.848e6
k = 0.4

z_0 = 2.4e-3
z_0T = 0.24e-3
z_0E = 0.24e-3
T_s = 0
z = 2

_hour = 3600
_mm_2_m = 1e-3

#
# def objective(T, u, Q_H, L):
#     u_fric = k*u/(np.log(z/z_0) - stab_M(z/L))
#     L_star = mon_length(T, u, Q_H)
#     Q_H_star = Q_H(P
#
# def mon_length(T, u, Q_H):
#     u_star =
#     return rho_0*c_p*u_star**3*T/(k*g*Q_H)

EnergyBalance = collections.namedtuple('EnergyBalance', ('QH', 'QE',
    'Qrad', 'Qground', 'Qrain', 'Qmelt', 'QSW', 'QLW'))

def stab_M(zeta):
    a = 1
    b = 2/3
    c = 5
    d = 0.35
    return -(a*zeta + b*(zeta - c/d)*np.exp(-d*zeta) + b*c/d)

def stab_H(zeta):
    # a = 1
    # b = 2/3
    # c = 5
    # d = 0.35
    # return -((1+2*zeta/3)**(1.5) + b*(zeta - c/d)*np.exp(-d*zeta) + b*c/d - 1)
    return 0



def Q_H(P, u, T, L):
    pref = c_p*rho_0*P*k**2/P_0
    denom = ( np.log(z/z_0) - stab_M(z/L) ) * ( np.log(z/z_0T) - stab_H(z/L) )
    return pref*u*(T - T_s)/denom

def Q_H_neutral(P, u, T):
    pref = c_p*rho_0*P*k**2/P_0
    denom = ( np.log(z/z_0)) * ( np.log(z/z_0T))
    return pref*u*(T - T_s)/denom

def Q_E(u, e, e_s, L):
    pref = L_v*0.623*rho_0*k**2/P_0
    denom = ( np.log(z/z_0) - stab_M(z/L) ) * ( np.log(z/z_0T) - stab_H(z/L) )
    return pref*u*(e - e_s)/denom

def Q_E_neutral(u, e, e_s):
    pref = L_v*0.623*rho_0*k**2/P_0
    denom = ( np.log(z/z_0) ) * ( np.log(z/z_0E) )
    return pref*u*(e - e_s)/denom


def sat_pressure(T):
    return 610.78*np.exp(17.08085*T/(234.15+T))

def vap_pressure(T, RH):
    return sat_pressure(T)*RH/100

def energy_balance(T, rh, u, P, SWin, SWout, LWnet, rain):
    # RADIATION
    SWnet = SWin - SWout
    Q_rad = SWnet + LWnet

    # RAIN
    Q_rain = rho_w*c_w*(rain/_hour*_mm_2_m)*(T - T_s)
    Q_rain[T<0] = 0
    # Q_rain[T>=0] = 0

    # ASSUMPTION
    Q_ground = np.zeros(T.shape)

    # Convert RH
    # Saturation vapour pressure
    e_2m = vap_pressure(T, rh)
    e_s = vap_pressure(T_s, rh)

    QH = Q_H_neutral(P, u, T)
    QE = Q_E_neutral(u, e_2m, e_s)

    # qh_guess = Q_H_neutral(P, u, T)
    # L_guess = z
    # max_iter = 1000
    # itnum = 0
    # while itnum < max_iter:
    #     u_fric = k*u/(np.log(z/z_0) - stab_M(z/L_guess))
    #     L_guess = rho_0*c_p*u_fric**3*T/(k*g*qh_guess)
    #     qh_new = Q_H(P, u, T, L_guess)
    #
    #     qh_delta = np.abs(qh_net - qh_guess)
    #     qh_guess = qh_new
    #     print(qh_delta)
    #     if qh_delta < _q_tol:
    #         break
    # L = L_guess
    # if L>z:
    #     QH = qh_guess
    #     QE = Q_E(u, e_2m, e_s, L)
    # else:
    #     QH = Q_H_neutral(P, u, T)
    #     QE = Q_E(u, e_2m, e_s)
    Q_melt = QH + QE + Q_rad + Q_ground + Q_rain
    Q_melt[Q_melt<0] = 0
    energy_balance = EnergyBalance(QH=QH, QE=QE, Qrad=Q_rad,
        Qground=Q_ground, Qrain=Q_rain, Qmelt=Q_melt, QSW=SWnet,
        QLW=LWnet)

    return energy_balance
