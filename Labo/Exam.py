import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import helpers as hp

w = np.logspace(-1, 3, num=2000)
s = 1j * w
H1 = 100/(s+100)
H2 = s**2/(s**2+141*s+10000)
Hs = H1+H2
norme = np.abs(Hs)
phase = np.angle(Hs)
module = 20*np.log10(norme)
phase_deg = np.rad2deg(phase)

fig, ax = plt.subplots(2)
ax[0].plot(w, module)
ax[0].set_xscale('log')
ax[1].plot(w, phase_deg)
ax[1].set_xscale('log')

derive = -np.diff(phase)/np.diff(w)
plt.figure()
plt.plot(w[1:], derive*1000)

###########################################################################
b1 = [100]
a1 = [1, 100]
b2 = [1, 0, 0]
a2 = [1, 141, 10000]
z1, p1, k1 = signal.tf2zpk(b1, a1)
z2, p2, k2 = signal.tf2zpk(b2, a2)
zp, pp, kp = hp.paratf(z1, p1, k1, z2, p2, k2)
bp, ap = signal.zpk2tf(zp, pp, kp)
t = np.linspace(0, 0.2, 2000)
sinus = np.sin(100*t)
t1, y1 = signal.impulse((bp, ap))
t2, y2, _ = signal.lsim((bp, ap), sinus, t)
fig, ax = plt.subplots(2)
ax[0].plot(t1, y1)
ax[1].plot(t2, y2)

############################################################################
b1 = [1, 1, 1]
a1 = [1, -1, 4]
z1, p1, k1 = signal.tf2zpk(b1, a1)
hp.pzmap1(z1, p1, "Pôles et zéros")
a1[1] = np.abs(a1[1])
z2, p2, k2 = signal.tf2zpk(b1, a1)
hp.pzmap1(z2, p2, "Pôles et zéros corrigés")
w, mag, phdeg = signal.bode((b1, a1), w=np.logspace(1, 3, 1000))
phrad = np.deg2rad(phdeg)
hp.grpdel1(w, -np.diff(phrad) / np.diff(w), 'des filtres')

plt.show()
