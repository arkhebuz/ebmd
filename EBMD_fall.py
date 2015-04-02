# -*- coding: utf-8 -*-
from math import pi, exp, atan, cos, sin, sqrt
from scipy.integrate import odeint
from scipy.optimize import fmin as simplex
import matplotlib.pyplot as plt
from MSL_EV import *


def read_MOLA_csv(offset = 4966+2300):
    rel_x = []
    rel_y = []
    I = 0
    import csv
    with open('mola.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if I:
                rel_x.append(float(row[0])*1000 + offset)
                rel_y.append(float(row[3]))
            I = 1
    return rel_x, rel_y


def EBMD_diffeq2d(w, t, p):
    """
    Defines the movement equations.
    Arguments:
        w :  vector of the state variables:
                  w = [x, y, vx, vy]
        t :  time
        p :  vector of the parameters:
                  p = [Cd, A, EBMD_mass, GMm, Mr]
    """
    x,y, vx,vy = w
    Cd, A, mass, GMm, temp_radius = p

    # Create f = (x', y', vx', vy'):
    r = sqrt(x**2 + y**2)
    if r > temp_radius+3:
        FG = GMm/(x*x + y*y)**1.5

        ro = MCD_acc(r)
        D = 0.5*Cd*A*(vx**2 + vy**2)*ro
        FT = D/(mass*sqrt(vx*vx + vy*vy))

        f = [vx,
             vy,
             -FT*vx - FG*x,
             -FT*vy - FG*y]
    else:
        f = [0,0,0,0]
    return f


def EBMD_diffeq2d_mod(w, t, p):
    """
    Defines the movement equations.
    Arguments:
        w :  vector of the state variables:
                  w = [x, y, vx, vy]
        t :  time
        p :  vector of the parameters:
                  p = [Cd, A, EBMD_mass, GMm, Mr]
    """
    x,y, vx,vy = w
    Cd, A, mass, GMm, temp_radius = p

    # Create f = (x', y', vx', vy'):
    r = sqrt(x**2 + y**2)

    if r > temp_radius+1:
        ro = MCD_acc(r)
    else:
        ro = 0.15

    FG = GMm/(x*x + y*y)**1.5
    D = 0.5*Cd*A*(vx**2 + vy**2)*ro
    FT = D/(mass*sqrt(vx*vx + vy*vy))

    f = [-vx,
         -vy,
         -FT*vx + FG*x,
         -FT*vy + FG*y]
    return f


def EBMD_solvec_info(solvec):
    x, y, vx, vy =  solvec[-1]
    hend = MCD_acc(sqrt(x**2 + y**2), zkey = 1, conv_hgs = True)[1]
    vend = sqrt(vx**2 + vy**2)
    #print 'INFO:'
    print '   H term sim: ', hend,
    print '\tHit Velocity: ', vend,
    print '\tHit Angle: ', 180*atan(-vy/vx)/pi,
    temp_radius = MCD_acc(0, zkey = 2, conv_hgs = True)[0]
    x = [ temp_radius*atan2(i[0], i[1]) for i in solvec]
    y = [ sqrt(i[0]**2 +i[1]**2) -temp_radius for i in solvec]
    print '\tDownrange: ', x[-1],
    print '\tDwonrange from MSL: ', x[-1] -4966 -2329

    return x, y


class EBMD(object):
    def area(self):
        #20090025343.pdf
        return 0.0192

    def mass(self):
        return 25.

    def Cd(self):
        return 1.0


class mod_EBMD(object):
    def area(self):
        return pi*0.101**2

    def mass(self):
        return 1.3

    def Cd(self):
        return 1.0


def EBMD_fall(state_init, EBMD_class):
    stoptime = 150
    numpoints = int(abs(stoptime)/0.05 + 1)
    timetuple = [stoptime*float(i)/(numpoints-1) for i in range(numpoints)]

    GMm = 0.42828371300166*10**14  # m^3/s^2
    temp_radius = MCD_acc(-4060, zkey = 2, conv_hgs = True)[0]

    # Pack up the parameters and initial conditions:
    p = [EBMD_class.Cd(), EBMD_class.area(), EBMD_class.mass(), GMm, temp_radius]
    # Call the ODE solver
    solvec = odeint(EBMD_diffeq2d, state_init, timetuple, args=(p,), atol=1.0e-8, rtol=1.0e-8, full_output = 0)
    return solvec


def eject_EBMDs(s, EV):
    time_poss = [-1, -401, -801, -1201, -1601, -2001]
    times = [-19.25, -17.25, -15.25, -13.25, -11.25, -9.25]
    dir_coeffs = [cos(pi*i/180.) for i in range(0,180,35)]

    jett_coords = []
    for i, t in enumerate(time_poss):
        st_vec = s[t]
        Ex, Ey = EV.lift_drag(st_vec, times[i], vec = [0, 1, 1])
        st_vec[2] = Ex*dir_coeffs[i]*1.0 -st_vec[2]
        st_vec[3] = Ey*dir_coeffs[i]*1.0 -st_vec[3]
        jett_coords.append(st_vec)
    return jett_coords


if __name__ == '__main__':
    s = EV_SUFR()
    xm, ym = EV_solvec_info(s)
    plt.plot(xm, ym, label='EV')

    print "\nSimulating EBMD fall"
    ebmd = EBMD()
    ev = EntryVehicle()
    jett_coords = eject_EBMDs(s, ev)
    for i, vec in enumerate(jett_coords):
        print "", i+1,
        solvec = EBMD_fall(vec, ebmd)
        x, y = EBMD_solvec_info(solvec)
        cap = 'EBMD'+str(i+1)
        plt.plot(x, y, label=cap)

    print " MOD",
    ebmd = mod_EBMD()
    s = EBMD_fall(s[-1], ebmd)
    xce, yce = EBMD_solvec_info(s)
    plt.plot(xce, yce, label='MOD_EBMD')

    plt.plot([4966-758,], [5359.8,], 'o', label='HSS')
    plt.plot([4966+2089,], [-2832.8,], 'o', label='BSS')
    plt.plot([2329 +4966,], [-4500,], 'o', label='TD')

    rel_x, rel_y = read_MOLA_csv()
    plt.plot(rel_x, rel_y, label='MOLA')

    plt.legend(loc='upper right', prop={'size':10})

    print "\nDeploying plot"
    plt.tight_layout()
    plt.show()

