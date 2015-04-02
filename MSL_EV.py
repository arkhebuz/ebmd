# -*- coding: utf-8 -*-
from math import pi, exp, atan, atan2, cos, sin, sqrt,tan
from scipy.integrate import odeint
from scipy.optimize import fmin as simplex
from fmcd import call_mcd, julian
import matplotlib.pyplot as plt
import numpy as np

MCD_Data_Catalog = ''


def MCD_acc(h, zkey = 1, lat = -4.6, lon = 137.4, conv_hgs = False):
    """
    Mars Climate Database acces routine.

    h           - Vertical height

    zkey = 1    - Verical coordinate type
    Keys:
    1: distance to center of planet
    2: height above areoid,
    3: height above surface,
    4: Pressure

    lat = -4.6  - Latitude
    lon = 137.4 - Longnitude

    conv_hgs    - If True, returns converted heights, if False returns pressure
    """

    dset = MCD_Data_Catalog     # default to 'MCD_DATA'
    perturkey = 1               # default to no perturbation
    seedin = 0                  # perturbation seed (unused if perturkey=1)
    gwlength = 0                # Gravity Wave length for perturbations (unused if perturkey=1)

    extvarkeys = np.zeros(100)  # Output the extra variables: No, but:
    if conv_hgs:
        extvarkeys[0] = 1           # Radial distance to planet center (m).
        extvarkeys[1] = 1           # Altitude above areoid (Mars geoid) (m).
        extvarkeys[2] = 1           # Altitude above local surface (m).
        extvarkeys[3] = 1           # Orographic height (m) (altitude of the surface with respect to the areoid).

    datekey = 0                 # Earth date
    loct = 0                    # Local time must then also be set to zero
    #                 MM, DD, YY, hh, mm, ss
    __, xdate = julian(8, 6, 2012, 5, 17, 57)       # Date of MSL landing

    # Dust scenario
    dust = 1                    # typical Mars year dust scenario, average solar EUV conditions

    hrkey = 1                   # High resolution mode: yes
    xz = h                      # Vertical height

    # Call MCD
    p = call_mcd(zkey, xz, lon, lat, hrkey,
                 datekey, xdate, loct, dset, dust,
                 perturkey, seedin, gwlength, extvarkeys)

    # Unpack output
    pres, dens, temp, zonwind, merwind, meanvar, extvar, seedout, ierr = p
    if not conv_hgs:
        return dens
    else:
        return [extvar[i] for i in range(4)]


class EntryVehicle(object):
    def __init__(self):
        # sufr total time [6], [4]
        self.SUFR_duration_total = 19.25                # s

        # mass at mortar fire, estimated [8]
        self.mass_at_mortar_fire = 2925                 # kg

        # single ebmd mass, various sources
        self.mass_EBMD = 25                             # kg

        # ebmd count, various sources[citation needed]
        self.EBMD_count = 6

        # MSL entry vehicle diameter, per [10]
        self.EV_diameter = 4.519                        # m

        # 0.005s is the timestamp interval of IMU data (aquired at 200 Hz)[citation needed]
        self.IMU_timestamp = 0.005                      # s

        # Cd of Entry Vehicle, [citation needed]
        self.EV_Cd = 1.4

        # Angle of Attack at SUFR start, [21] s. 8, [22] s. 15
        self.AoA_at_SUFR_start = -23.2*pi/180           # rad

        # Flight Path Angle at PD [6]
        # (in MSL coordinate system the value is negative)
        angle = 22.4                                    # deg
        self.FPA_at_PD = pi*angle/180                   # convert to radians

        # Speed at PD [6]
        self.PD_speed = 406.349                         # m/s

        # Time at PD [6]
        self.PD_time = 799.125                          # s

        # Downrange at PD [6]
        self.PD_downrange = 4966                        # m

        # Altitude at PD [6]
        self.PD_altitude_mola = 7542.9                  # m

        # AoA history during SUFR [21] s8
        self.AoA_history = [[779.875, -23.3],
                            [780.875, -21],
                            [781.875, -18],
                            [782.875, -14.5],
                            [783.875, -11],
                            [784.875, -9],
                            [785.875, -7],
                            [786.875, -5],
                            [787.875, -3.3],
                            [788.875, -1.5],
                            [789.875, -0.3],
                            [790.875, 1.5],
                            [791.875, 4],
                            [792.875, 4],
                            [793.875, 2.2],
                            [794.875, 0.4],
                            [795.875, -0.2],
                            [796.875, 2],
                            [797.875, 3],
                            [798.875, 2.8],
                            [799.125, 2.5] ]

        # Bank angle history [citation needed]
        self.bank_angle_history = [ [240, -2],
                                    [242, -14],
                                    [245, -75],
                                    [250, -160],
                                    [252, -173],
                                    [255, -175],
                                    [260, -180] ]
        self.fit_AoA()
        self.fit_bank()

    def fit_AoA(self):
        t = [ i[0] - 779.875 for i in self.AoA_history]
        AoA = [ -pi*i[1]/180. for i in self.AoA_history]
        z = np.polyfit(t, AoA, 11) #fitnij polynomial
        f = np.poly1d(z) #zbuduj polynimoala
        self.AoA_function = f

    def fit_bank(self):
        t = [ i[0] - 240 for i in self.bank_angle_history]
        bank = [ -pi*i[1]/180. for i in self.bank_angle_history]
        z = np.polyfit(t, bank, 5) #fitnij polynomial
        f = np.poly1d(z) #zbuduj polynimoala
        self.bank_function = f

    def mass(self, t, dt0 = 0):
        '''
        Function at default takes negative time values,
        with t0=0 at parachute deploy and t=-19.25 at SUFR start
        (time is converted internally)

        From [6] about times:
        "Mode 14, ”SLEW AND SUFR SLEW TO RADAR ATTITUDE”; and Mode 15, ”SLEW AND SUFR WAIT
        FOR CHUTE DEPLOY”. The observed time in these two GN&C modes was 14.0 s and 5.25 s, respec-
        tively, with a combined time between SUFR and PD of 19.25 s."
        '''

        # convert time to positive numbers
        t = t + self.SUFR_duration_total

        # First EBMD jettision marks the beggining of the SUFR [21][27]
        n = self.EBMD_count - 1

        # ebmd jettison interval, various sources
        jett_interval = 2       # s

        # mass change model
        if dt0 <= t <= n*jett_interval + dt0:
            remaining_EBMDs = n - int((t-dt0)*1.0/jett_interval)
            mass = self.mass_at_mortar_fire + self.mass_EBMD*remaining_EBMDs
        elif t < dt0:
            mass = self.mass_at_mortar_fire + self.mass_EBMD*n
        elif t > n*jett_interval + dt0:
            mass = self.mass_at_mortar_fire

        return 1.0*mass

    def area(self):
        '''Returns frontal area of the MSL aeroshell'''
        return pi*(self.EV_diameter/2)**2      # return frontal area A

    def state_at_PD(self):
        Y, __, __, __ = MCD_acc(self.PD_altitude_mola, zkey = 2, conv_hgs = True)

        # Build state vector at parachute deploy (assuming time flowing backwards ;)
        #               x         y               vx             vy
        #state_init = [0.001, Mr_eq+H_PDmola, -self.PD_speed*cos(self.FPA_at_PD), self.PD_speed*sin(self.FPA_at_PD)]
        state_init = [0.001, Y, -self.PD_speed*cos(self.FPA_at_PD), self.PD_speed*sin(self.FPA_at_PD)]
        return state_init

    def simple_LD(self, t):
        # convert time to positive numbers
        t = t + self.SUFR_duration_total

        # [9]
        LD0 = 0.32

        # Bank change time (from charts, approx)
        roll_time = 12          # s

        # Cd change model
        if t <= roll_time:
            LD = LD0*(1-t*(1./roll_time))**2 # 0.66 coeff to better match FPA@SURF
        elif t > roll_time:
            LD = 0
        return abs(LD)

    def lift_drag(self, state_vector, t, vec = [-1, 0, 1]):
        """
        vec = [-1, 0, 1]
        Drag versor in capsule coordinate system
        """
        x, y, vx, vy = state_vector

        # kąt promienia wodzącego
        tau_r = atan(-x/y)

        # Flight Patch Angle
        FPA = atan(vy/vx)

        # convert time to positive numbers
        t = t + self.SUFR_duration_total

        # Bank change time (from charts, approx)
        roll_time = 10          # s

        #~ if t <= roll_time:
            #~ AoA = self.AoA_at_SUFR_start*(1-t*(1./roll_time))
        #~ elif t > roll_time:
            #~ AoA = 0.
        AoA = self.AoA_function(t)*cos(self.bank_function(t))
        # tau_r > 0; FPA < 0, AoA > 0
        # everything should be < 0
        Ex, Ey, __ = np.matrix(vec).dot(self.obrot_o_kat(-tau_r +FPA -AoA)).tolist()[0]
        return Ex, Ey

    def obrot_o_kat(self, kat):
        """Helper method for self.lift_drag"""
        m = np.matrix([ [ cos(kat), sin(kat), 0],
                        [-sin(kat), cos(kat), 0],
                        [        0,        0, 1] ])
        return m



# =============================================================================

def EV_diffeq2d(w, t, p):
    """
    Defines the movement equations.
    Arguments:
        w :  vector of the state variables:
                  w = [x, y, vx, vy]
        t :  time
        p :  vector of the parameters:
                  p = [Cd, A, EBMD_mass, LiftDrag, GMm]
    """
    x,y, vx,vy = w
    Cd, A, EV_mass, LiftDrag, GMm = p
    mass = EV_mass(t)

    # Create f = (x', y', vx', vy'):
    FG = GMm/(x*x + y*y)**1.5
    ro = MCD_acc(sqrt(x**2 + y**2))
    D = 0.5*Cd*A*(vx**2 + vy**2)*ro
    Ex, Ey = LiftDrag(w, t)

    f = [-vx,
         -vy,
         -D*Ex/mass + FG*x,
         -D*Ey/mass + FG*y]
    return f


def EV_solvec_info(solvec):
    x, y, vx, vy =  solvec[-1]
    hend = MCD_acc(sqrt(x**2 + y**2), zkey = 1, conv_hgs = True)[1]
    vend = sqrt(vx**2 + vy**2)
    print ' SUFR start MOLA Height:   ', hend
    print ' SUFR start velocity:      ', vend
    print ' SUFR start angle:        ', -180*atan(-vy/vx)/pi
    temp_radius = MCD_acc(0, zkey = 2, conv_hgs = True)[0]
    x = [ temp_radius*atan2(i[0], i[1]) for i in solvec]
    y = [ sqrt(i[0]**2 +i[1]**2) -temp_radius for i in solvec]
    print ' SUFR start Downrange:     ', -x[-1] + 4966
    print ' X Distance at PD-2.25s:   ', -x[450]
    print ' Y Distance at PD-2.25s:   ', y[450] - 7542.9
    print ' Velocity at PD-2.25s:     ', sqrt(solvec[450][2]**2+solvec[450][3]**2)

    return x, y


def EV_SUFR():
    print "Initialization"
    EV = EntryVehicle()
    state_init = EV.state_at_PD()
    print "\nSimulating SUFR"

    stoptime = -1*EV.SUFR_duration_total
    numpoints = int(abs(stoptime)/EV.IMU_timestamp + 1)    # there should be 3851 points
    timetuple = [stoptime*float(i)/(numpoints-1) for i in range(numpoints)]

    Cd = EV.EV_Cd
    GMm = 0.42828371300166*10**14  # m^3/s^2

    # Pack up the parameters and initial conditions:
    p = [Cd, EV.area(), EV.mass, EV.lift_drag, GMm]
    # Call the ODE solver
    solvec = odeint(EV_diffeq2d, state_init, timetuple, args=(p,), atol=1.0e-8, rtol=1.0e-8, full_output = 0)
    return solvec


if __name__ == '__main__':
    s = EV_SUFR()
    x, y = EV_solvec_info(s)
    #~ print s[-1]
    #~ plt.plot(x,y)
    #~ plt.show()

    #s = [ -7.89877517e+03,   3.40572309e+06,  -4.45226107e+02,   1.11121532e+02]
    s = s[-1]
    ev = EntryVehicle()
    print ev.lift_drag(s, -19.25, vec = [-1, 0,1])
    print ev.lift_drag(s, -19.25, vec = [0, -1,1])
    #~ x = [a/100. for a in range(0,1925,5)]
    #~ ya = map(ev.AoA_function, x)
    #~ plt.plot(x, ya)
    #~ plt.plot([i[0]- 779.875 for i in ev.AoA_history], [-pi*i[1]/180. for i in ev.AoA_history])
    #~ plt.show()
#~
    #~ yb = map(ev.bank_function, x)
    #~ plt.plot(x, yb)
    #~ plt.plot([i[0]- 240.0 for i in ev.bank_angle_history], [-pi*i[1]/180. for i in ev.bank_angle_history])
    #~ plt.show()

    #~ l = [a*cos(b) for a, b in zip(ya, yb)]
    #~ plt.plot(x, l)
    #~ plt.show()
