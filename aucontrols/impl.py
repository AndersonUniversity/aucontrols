from functools import partial
from math import acos, cos

import numpy as np
import pylab  # plotting routines
import scipy.signal  # signal processing toolbox
from matplotlib.pyplot import grid, plot, show, xlabel, xlim, ylabel
from numpy import (angle, convolve, flipud, linspace, loadtxt, ndarray, ones,
                   roots, shape, sqrt, squeeze, tan, where, zeros)
from scipy import array, imag, poly1d, real, row_stack, zeros_like

from control.exception import ControlMIMONotImplemented
from control.matlab import dcgain, feedback, pole, rlocus, step, tf
from control.xferfcn import _convertToTransferFunction

from .rlocus import *


def stepinfo(y_input, t_input, round_in=4):
    '''
    Get the second order time response characteristics
    Given the time domain response of a system
    r, ts, Mp, tp, yss = stepinfo(y_input,t_input,round_in)
    Parameters:
    :param y_input: [array-like] response of a step function
    :param t_input:  [array-like] time array from a step function
    :param round_in: [int] digits after decimal, default = 4
    Returns:
    :return: tr: [float] rise time (time from 0.1yss to 0,9yss) [s]
    :return: ts: [float] settling time (2% settling time) [s]
    :return: Mp: [float] percent overshoot [%]
    :return: tp: [float] time of peak [s]
    :return: yss: [float] steady state y value
    '''

    y = array(y_input)  # get the input y values
    t = array(t_input)  # get the input x values

    # yss
    yss = round(y[len(y) - 1], round_in)

    # find Mp and tp
    Mp = max(y)
    tp_index = where(y == Mp)
    Mp = round(100 * (Mp - yss) / yss, round_in)
    tp = squeeze(t[tp_index])
    tp = round(float(tp), round_in)

    # find tr
    t0 = -1
    t1 = -1
    for index in range(len(y)):
        if (t0 < 0 and y[index] >= 0.1 * yss):
            t0 = t[index]
        if (t1 < 0 and y[index] >= 0.9 * yss):
            t1 = t[index]
        if (t0 > 0 and t1 > 0):
            break
    tr = round(t1 - t0, round_in)

    # find ts by flipping t and y
    yud = flipud(y)
    tud = flipud(t)

    # Loop over indices
    for index in range(len(y)):
        if yud[index] >= 1.01 * yss or yud[index] <= 0.99 * yss:
            ts = tud[index]
            ts = round(ts, round_in)
            break

    return tr, ts, Mp, tp, yss


def zetaline(zeta_in, minx=-10.):
    '''
    Get a line plotted in the s-plane corresponding to a specific zeta.
    Creates a plot of the positive and negative zeta line.  You must use
    plt.show() to display these on an existing root locus plot.  You should use
    the following syntax:
        rlocus(G)
        zetaline(zeta_in)
        plt.show()
    zetaline(zeta_in) or zetaline(zeta_in, minx, num_pts)
    Parameters:
    :param zeta_in: [float] desired zeta (damping) value for the line
    :param minx: [float] minimum x value for drawn line, default = -10.
    Returns:
    :return: [none]
    '''
    num_pts = 1000.
    zeta = acos(zeta_in)
    x = linspace(minx, 0, num_pts)
    zline = x * -tan(zeta)
    neg_zline = x * tan(zeta)
    plot(x, zline, 'k--')
    plot(x, neg_zline, 'k--')
    return


def getstepfromtxt(filename, skip_rows=0):
    '''
    Get step response data from a .txt file.
    y, t = getstepfromtxt(filename,skip_rows=0)
    :param filename: [string] filename containing step response data, .txt with comma delimiters
                    first column is time data, second column is response data
    :param skip_rows: [int] rows to skip if file has headers
    :return: y: [array-like] step response output data
    :return: t: [array-like] step response time data
    '''

    data = loadtxt(filename, delimiter=",", skiprows=skip_rows)
    # organize data into two separate arrays
    y = array(data[:, 1])
    t = array(data[:, 0])
    return y, t


def pidplot(num, den, Kp, Ki, Kd, desired_settle=1.):
    '''
    Plot system step response when open loop system is subjected to feedback PID compensation.
    Also plots 2% settling lines, and a vertical line at a desired settling time.
    y, t =pidplot(num,den,Kp,Ki,Kd,desired_settle=1.)
    Parameters:
    :param num: [array-like], coefficients of numerator of open loop transfer function
    :param den: [array-like], coefficients of denominator of open loop transfer function
    :param Kp: [float] Proportional gain
    :param Ki: [float] Integral gain
    :param Kd: [float] Derivative gain
    :param desired_settle: [float] Desired settling time, for tuning PID controllers, default = 1.0
    Returns:
    :return: y: [array-like] time step response
    :return: t: [array-like] time vector
    '''
    numc = [Kd, Kp, Ki]
    denc = [0, 1, 0]

    numcg = convolve(numc, num)
    dencg = convolve(denc, den)

    Gfb = feedback(tf(numcg, dencg), 1)
    y, t = step(Gfb)
    yss = dcgain(Gfb)
    plot(t, y, 'r')
    plot(t, 1.02 * yss * ones(len(t)), 'k--')
    plot(t, 0.98 * yss * ones(len(t)), 'k--')
    plot(desired_settle * ones(15), linspace(0, yss + 0.25, 15), 'b-.')
    xlim(0)
    xlabel('Time [s]')
    ylabel('Magnitude')
    grid()
    show()
    return y, t


def pidtune(num, den, Kp, Ki, Kd, desired_settle=1.):
    '''
    Person in loop tuning algorithm for PID compensators
    Kp, Ki, Kd = pidtune(num, den, Kp, Ki, Kd)
    Parameters:
    :param num: [array-like], coefficients of numerator of open loop transfer function
    :param den: [array-like], coefficients of denominator of open loop transfer function
    :param Kp: [float] Proportional gain
    :param Ki: [float] Integral gain
    :param Kd: [float] Derivative gain
    :param desired_settle: [float] Desired settling time, for tuning PID controllers, default = 1.0
    Returns:
    :return: Kp: [float] Tuned proportional gain
    :return: Ki: [float] Tuned integral gain
    :return: Kd: [float] Tuned derivative gain
    '''
    flag = 1

    while flag:
        y1, t1 = pidplot(num, den, Kp, Ki, Kd, desired_settle)
        tr, ts, Mp, tp, yss = stepinfo(y1, t1)
        Kp_old = Kp
        Ki_old = Ki
        Kd_old = Kd
        print(["tr = ", tr])
        print(["ts = ", ts])
        print(["Mp(%) = ", Mp])
        print(["tp = ", tp])
        print(["yss = ", yss])

        userin = input("Is the response satisfactory (0=No, 1=Yes) ? ")
        if float(userin) == 1.0:
            flag = 0

        if flag:
            Kp = input(["Enter new Kp (<return> yields old Kp=", Kp, ": "])
            try:
                Kp = float(Kp)
            except:
                Kp = Kp_old
            # see if i can do try except here
            Ki = input(["Enter new Ki (<return> yields old Ki=", Ki, ": "])
            try:
                Ki = float(Ki)
            except:
                Ki = Ki_old
            Kd = input(["Enter new Kd (<return> yields old Kd=", Kd, ": "])
            try:
                Kd = float(Kd)
            except:
                Kd = Kd_old

    return Kp, Ki, Kd


def rlocfind2(num, den, desired_zeta):
    '''
    Find the locations on the root locus with the closest damping values
    Computes numerically, a bit hacky
    Parameters:
    :param num: [array-like] coefficients of numerator of open loop transfer function
    :param den: [array-like] coefficients of denominator of open loop transfer function
    :param desired_zeta: [float] desired damping coefficient value
    Returns:
    :return: polelocs: [array-like] complex valued pole locations that meet requested zeta values
    :return: ks: [array-like] gain at selected pole locations on root locus
    :return: wnvals: [array-like] natural frequency at selected pole locations
    :return: zvals: [array-like] actual damping value and selected pole locations
    '''
    rlist, klist = rlocus(tf(num, den))
    anglelist = angle(rlist)
    tem = shape(anglelist)
    tem = tem[1]
    zlist = ones(shape(rlist))
    for k in range(tem):
        for j in range(len(klist)):
            zlist[j, k] = abs(cos(anglelist[j, k]))
    locclosest = ones(tem)
    eps = ones(tem)
    for k in range(tem):
        difflist = ones(len(klist))
        for j in range(len(klist)):
            difflist[j] = abs(desired_zeta - zlist[j, k])
        # minv = min(difflist[0:len(difflist)])
        for j in range(len(klist)):
            if difflist[j + 1] <= difflist[j]:
                locclosest[k] = j + 1
                eps[k] = difflist[j + 1]
            elif difflist[j + 1] > difflist[j]:
                break
    locclosest = ndarray.tolist(locclosest)
    for k in range(len(locclosest)):
        locclosest[k] = int(locclosest[k])
    locs = ones((tem, 3))
    for k in range(tem):
        locs[k, :] = [real(rlist[locclosest[k], k]), imag(
            rlist[locclosest[k], k]), klist[locclosest[k]]]
    polelocs = locs[:, 0] + locs[:, 1] * 1j
    ks = locs[:, 2]
    validvals = zeros((tem, 1))
    for k in range(len(eps)):
        if eps[k] < 0.1:
            validvals[k] = 1

    inc = 0
    finallocs = ndarray.tolist(zeros(int(sum(validvals))))
    finalks = zeros(int(sum(validvals)))
    for k in range(len(eps)):
        if validvals[k] == 1.:
            finallocs[inc] = polelocs[k]
            finalks[inc] = ks[k]
            inc = inc + 1

    ks = finalks
    polelocs = finallocs
    wnvals = sqrt(real(polelocs)**2 + imag(polelocs)**2)
    zvals = angle(polelocs)
    for k in range(len(zvals)):
        zvals[k] = abs(cos(zvals[k]))

    return polelocs, ks, wnvals, zvals


def rlocfind(sys, desired_zeta, kvectin=None):
    '''
    Interactive gain selection from the root locus plot
    of the SISO system SYS.
    rlocfind lets you select a pole location
    in the graphics window on the root locus
    computed from SYS. The root locus gain associated
    with this point is returned as K
    and the system poles for this gain are returned as POLES.
    :param sys: [transfer function object] transfer function of open loop system
    :param desired_zeta: [float] desired damping coefficient value
    :param kvectin: [array-like] k vector of values for root locus determination, default = None
    Returns:
    :return: K: [float] gain at point clicked on root locus
    :return: POLES: [array-like] all (complex) pole locations for the gain chosen
    '''
    rlist, klist, Gdict = root_locus(sys, kvect=kvectin, PrintGain=True)
    zetaline(desired_zeta)
    show()
    K = squeeze(Gdict["k"].real)
    POLES = pole(feedback(K * sys, 1))
    return K, POLES
