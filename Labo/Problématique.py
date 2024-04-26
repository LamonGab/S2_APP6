
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp


def butterworth():
    b1 = [0, 0, -1.91e+07]
    a1 = [1.00000000e+00, 6.21e+03, 1.91e+07]
    z1, p1, k1 = signal.tf2zpk(b1, a1)
    print(f'Passe-bas 700 Hz Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    print(f'Zéros:{z1}, Pôles:{p1}, Gain:{k1}')
    hp.pzmap1(z1, p1, '700 Hz')

    tf1 = signal.TransferFunction(b1, a1)
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 10, 200))
    hp.bode1(w1, mag1, phlin1, 'Bode 1')

    b2 = [-1, 0, 0]
    a2 = [1.00000000e+00, 6.1e+04, 1.9e+09]
    z2, p2, k2 = signal.tf2zpk(b2, a2)
    print(f'Passe-haut 7000 Hz Numérateur {b2}, Dénominateur {a2}')  # affiche les coefficients correspondants au filtre
    print(f'Zéros:{z2}, Pôles:{p2}, Gain:{k2}')
    hp.pzmap1(z2, p2, '7000 Hz')

    tf2 = signal.TransferFunction(b2, a2)
    w2, mag2, phlin2 = signal.bode(tf2, np.logspace(-1.5, 10, 200))
    hp.bode1(w2, mag2, phlin2, 'Bode 2')

    b3 = [-1, 0, 0]
    a3 = [1.00000000e+00, 8.8e+04, 3.9e+09]
    z3, p3, k3 = signal.tf2zpk(b3, a3)
    print(f'Passe-bas 1000 Hz Numérateur {b3}, Dénominateur {a3}')  # affiche les coefficients correspondants au filtre
    print(f'Zéros:{z3}, Pôles:{p3}, Gain:{k3}')
    hp.pzmap1(z3, p3, '1000 Hz')

    tf3 = signal.TransferFunction(b3, a3)
    w3, mag3, phlin3 = signal.bode(tf3, np.logspace(-1.5, 10, 200))
    hp.bode1(w3, mag3, phlin3, 'Bode 1000 Hz')

    b4 = [0, 0, -9.9e+08]
    a4 = [1.00000000e+00, 4.5e+04, 9.9e+08]
    z4, p4, k4 = signal.tf2zpk(b4, a4)
    print(f'Passe-haut 5000 Hz Numérateur {b4}, Dénominateur {a4}')  # affiche les coefficients correspondants au filtre
    print(f'Zéros:{z4}, Pôles:{p4}, Gain:{k4}')
    hp.pzmap1(z4, p4, '5000 Hz')

    tf4 = signal.TransferFunction(b4, a4)
    w4, mag4, phlin4 = signal.bode(tf4, np.logspace(-1.5, 10, 200))
    hp.bode1(w4, mag4, phlin4, 'Bode 5000 Hz')

    zp1, pp1, kp1 = hp.paratf(z1, p1, k1, z2, p2, k2)
    zs, ps, ks = hp.seriestf(z3, p3, k3, z4, p4, k4)
    zp2, pp2, kp2 = hp.paratf(zp1, pp1, kp1, zs, ps, ks)
    bp2, ap2 = signal.zpk2tf(zp2, pp2, kp2)
    magp, php, wp, fig, ax = hp.bodeplot(bp2, ap2, 'Somme des filtres')
    hp.grpdel1(wp, -np.diff(php) / np.diff(wp), 'des filtres')

    bs, a_s = signal.zpk2tf(zs, ps, ks)
    tfs = signal.TransferFunction(bs, a_s)
    ws, mags, phlins = signal.bode(tfs, np.logspace(-1.5, 10, 200))
    hp.bode1(ws, mags, phlins, 'Passe-bande')


def main():
    butterworth()
    plt.show()


if __name__ == '__main__':
    main()
