# set the bounds of the domain
lx = 0
ux = 1

# Set a whole bunch of parameters
spy = 60 * 60 * 24 * 365.25
rhoi = 910.
rhow = 1028.
delta = 1. - rhoi / rhow
g = 9.81
a = 0.3 / spy
Q0 = 4.0e5 / spy
H0 = 1.0e3
B0 = 1.4688e8
n = 3

# Set a whole bunch of scalings
Z0 = a ** (1/(n+1)) * (4 * B0) ** (n / (n + 1)) / (rhoi * g * delta) ** (n/(n + 1))
U0 = 400 / spy
Lx = U0 * Z0 / a
h0 = H0 / Z0; q0 = Q0 / (U0 * Z0)
nu_star = (2 * B0) / ( rhoi * g * delta * Z0) * (U0 / Lx) ** (1 / n)
A0 = (a * Lx) / (U0 * Z0)
