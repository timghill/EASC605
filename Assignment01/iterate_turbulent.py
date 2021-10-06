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
z0 = 2.4e-3

z_0 = 2.4e-3
z_0T = 0.24e-3
z_0E = 0.24e-3
T_s = 0
z = 2

_hour = 3600
_mm_2_m = 1e-3

E_s = 613
E_2m = 872

RH = 0.75

T_2m = 5
e_s = E_s*RH
e_2m = E_2m*RH

u = 3
z = 2
P = 80e3


def stab_M(zeta):
    a = 1
    b = 2/3
    c = 5
    d = 0.35
    return -(a*zeta + b*(zeta - c/d)*np.exp(-d*zeta) + b*c/d)

def stab_H(zeta):
    a = 1
    b = 2/3
    c = 5
    d = 0.35
    return -((1+2*zeta/3)**(1.5) + b*(zeta - c/d)*np.exp(-d*zeta) + b*c/d - 1)
    # return 0

def u_fric(xi):
    return k*u/(np.log(z/z0) - stab_M(xi))

def stab_length(Qh, uf):
    return (rho_0*c_p*uf**3*(T_2m+273.15))/(k*g*Qh)

def Q_H(P, u, T, xi):
    pref = c_p*rho_0*P*k**2/P_0
    denom = ( np.log(z/z_0) - stab_M(xi) ) * ( np.log(z/z_0T) - stab_H(xi) )
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

# Iterative procedure
xi = 0
Q = Q_H(P, u, T_2m, xi)
u_f = u_fric(xi)
tol = 1e-6
err = 1
while err>tol:
    xi = z/stab_length(Q, u_f)
    Qnew = Q_H(P, u, T_2m, xi)
    u_f = u_fric(xi)
    err = np.abs(Qnew - Q)
    Q = Qnew
    print(Q)
