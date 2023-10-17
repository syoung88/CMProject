import warnings
import numpy as np
import math
import sklearn
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from sklearn.linear_model import BayesianRidge

from plotting import *
from gradient_descent_complete import *
from gradient_descent_complete import gaussian3D as obj

def ode_model(t, p, q, a, b, c, p0, p1):
    """ODE model for aquifer pressure.
    
    Parameters
    ----------
    t : float
        Time.
    p : float
        Aquifer pressure.
    q : float
        Extraction rate.
    a : float
        Extraction parameter.
    b : float
        Recharge parameter.
    c : float
        Saltwater intrusion parameter.
    p0 : float
        Pressure of freshwater spring.
    p1 : float
        Pressure of ocean.
    
    Returns
    -------
    dpdt : float
        Derivative of aquifer pressure with respect to time.
    """

    dpdt = -a*q - b*(p - p0) - c*(p - p1)
    return dpdt

def load_data():
    """ Load data throughout the time period.
    Parameters:
    -----------
    Returns:
    ----------
    t_q : array-like
        Vector of times at which measurements of q were taken.
    q : array-like
        Vector of q (kg/year)
    p_x : array-like
        Vector of times at which measurements of p were taken.
    p : array-like
        Vector of p (MPa)
    """
    # Load kettle data
    t_step_p, p = np.genfromtxt('P_acquifer.csv', delimiter=',', skip_header=1).T
    t_step_q, q = np.genfromtxt('q_acquifer.csv', delimiter=',', skip_header=1).T

    # # calibration step of 70%
    # lengthp = len(p)
    # calbp = round(0.7 * lengthp - 1)
    # t_step_p = t_step_p[0:calbp]
    # p = p[0:calbp]

    # # calibration step of 70% from 1990
    # lengthq = len(q) - 30
    # calbq = round(0.7 * lengthq - 1)
    # t_step_q = t_step_q[30:calbq + 30]
    # q = q[30:calbq + 30]

    # Pressure of the aquifer
    p += 0.101

    # Convert q to kg/s
    q /= 31536000

    # Convert to Pa?
    # p *= 1000000

    # Convert to SI units
    # t_step_p *= 31536000
    # t_step_q *= 31536000
    # p *= 1000000

    return t_step_q, q, t_step_p, p

# This function solves your ODE using Improved Euler
def solve_ode(f, t0, t1, dt, xi, pars):
    """ Solve an ODE using the Improved Euler Method.
    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    xi : float
        Initial value of solution.
    pars : array-like
        List of parameters passed to ODE function f.
    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.
    Notes:
    ------
    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """

    # set an arbitrary initial value of q for benchmark solution
    q = -1.0

    if pars is None:
        pars = []

    # calculate the time span
    tspan = t1 - t0
    # use floor rounding to calculate the number of variables
    n = int(tspan // dt)

    # initialise the independent and dependent variable solution vectors
    x = [xi]
    t = [t0]

    # perform Improved Euler to calculate the independent and dependent variable solutions
    for i in range(n):
        f0 = f(t[i], x[i], q, *pars)
        f1 = f(t[i] + dt, x[i] + dt * f0, q, *pars)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return t, x



# This function defines your ODE as a numerical function suitable for calling 'curve_fit' in scipy.
def x_curve_fitting(t, a, b, c):
    """ Function designed to be used with scipy.optimize.curve_fit which solves the ODE using the Improved Euler Method.
        Parameters:
        -----------
        t : array-like
            Independent time variable vector
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        c : float
            saltwater intrusion strength parameter.
        Returns:
        --------
        x : array-like
            Dependent variable solution vector.
        """
    # model parameters
    pars = [a, b, c]

    # ambient value of dependent variable
    x0 = 0.304
    x1 = 0.101

    # time vector information
    n = len(t)
    dt = t[1] - t[0]

    # read in time and dependent variable information
    [t, x_exact] = [load_data()[2], load_data()[3]]

    # initialise x
    x = [x_exact[0]]

    # read in q data
    [t_q, q] = [load_data()[0], load_data()[1]]

    # using interpolation to find the injection rate at each point in time
    # MIGHT NOT BE NEEDED, as time step of q data is same as P data
    q = np.interp(t, t_q, q)

    # using the improved euler method to solve the ODE
    for i in range(n - 1):
        f0 = ode_model(t[i], x[i], q[i], *pars, x0, x1)
        f1 = ode_model(t[i] + dt, x[i] + dt * f0, q[i], *pars, x0, x1)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))

    return x


# This function calls 'curve_fit' to improve your parameter guess.
def x_pars(pars_guess):
    """ Uses curve fitting to calculate required parameters to fit ODE equation
    Parameters
    ----------
    pars_guess : array-like
        Initial parameters guess
    Returns
    -------
    pars : array-like
           Array consisting of a: mass injection strength parameter, b: recharge strength parameter, c: saltwater intrusion strength parameter
    """
    # read in time and dependent variable data
    [t_exact, x_exact] = [load_data()[2], load_data()[3]]

    # finding model constants in the formulation of the ODE using curve fitting
    # optimised parameters (pars) and covariance (pars_cov) between parameters
    try:
        pars, pars_cov = curve_fit(x_curve_fitting, t_exact, x_exact, pars_guess, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
    except:
        pars = [0, 0, 0]
        pars_cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    return pars, pars_cov


def solve_ode_prediction(f, t0, t1, dt, pi, q, a, b, c, p0, p1, p_want=0.0, t_want=0.0, wait=0.0, mp=0.0):
    """ Solve the pressure prediction ODE model using the Improved Euler Method.
    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    pi : float
        Initial value of solution.
    q : float
        Extraction rate
    a : float
        mass injection strength parameter.
    b : float
        recharge strength parameter.
    c : float
        saltwater intrusion strength parameter.
    p0 : float
        Freshwater spring pressure.
    p1 : float
        Ocean pressure.
    p_want : float (optional)
        The desired pressure of the aquifer
    t_want : float (optional)
        The desired length of time to reach the pressure of the aquifer (p_want)
    wait : float (optional)
        How long until you begin the process of changing the pressure toward p_want
    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    p : array-like
        Dependent variable solution vector.
    Notes:
    ------
    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """

    # finding the number of time steps
    tspan = t1 - t0
    n = int(tspan // dt)

    # initialising the time and solution vectors
    p = [pi]
    t = [t0]
    incr = [0]*n

    if mp != 0.0:
        for i in range(n):
            incr[i] = n * mp

    if p_want != 0.0 and t_want != 0.0 and mp == 0.0:
        # from the model analytic solution, estimate the extraction rate that will resolve to a desired pressure over a
        # desired time (I suggest you highly ignore my cheeky use of the analytical solution)
        pars = [3.944e-08,  5.418, -4.417]
        q_est = (1/(pars[0]*(1 - np.exp(-(t_want+t1)))))*((1 - np.exp(-(t_want + t1)))*(pars[1]*p0 + pars[2]*p1) -
                                                          p_want*(pars[1] + pars[2]))
        # find the average increase per year that will reach the desired pressure
        d = int((q_est - q)/t_want)

        # calculate array of extraction changes
        for i in range(n):
            if i <= wait:
                incr[i] = 0
            elif i - wait <= t_want:
                incr[i] = d
            else:
                incr[i] = 0

    # using the improved euler method to solve the pressure ODE
    for i in range(n):
        q = q + incr[i]
        f0 = f(t[i], p[i], q, a, b, c, p0, p1)
        f1 = f(t[i] + dt, p[i] + dt * f0, q, a, b, c, p0, p1)
        p.append(p[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return t, p


# This function plots your model over the data using your estimate for a and b
def plot_suitable():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and pressure data
    [t, p_exact] = [load_data()[2], load_data()[3]]

    # TYPE IN YOUR PARAMETER ESTIMATE FOR a, b and c HERE
    pars = [3.271e-9, 0.421, 0.0421]
  
    # solve ODE with estimated parameters and plot 
    p = x_curve_fitting(t, *pars)
    ax1.plot(t, p_exact, 'k.', label='Observation')
    ax1.plot(t, p, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure of aquifer (MPa)')
    ax1.set_xlabel('Time (years)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = p
    for i in range(len(p)):
        misfit[i] = p_exact[i] - p[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Pressure misfit (MPa)')
    ax2.set_xlabel('Time (years)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# This function plots your model over the data using your improved model after curve fitting.
def plot_improve(a, b, c):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and pressure data
    [t, p_exact] = [load_data()[2], load_data()[3]]

    # TYPE IN YOUR PARAMETER GUESS FOR a, b and c HERE AS A START FOR OPTIMISATION
    pars_guess = [a, b, c]
    # a = 0.00327
    # b = 0.147
    # c = 0.0147
    
    # call to find out optimal parameters using guess as start
    pars, pars_cov = x_pars(pars_guess)

    # check new optimised parameters
    print ("Improved a, b and c: ", pars)

    # solve ODE with new parameters and plot 
    p = x_curve_fitting(t, *pars)

    ax1.plot(t, p_exact, 'k.', label='Observation')
    ax1.plot(t, p, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (years)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = p
    total_misfit = 0
    for i in range(len(p)):
        misfit[i] = p_exact[i] - p[i]
        total_misfit += misfit[i] ** 2
    print("Total misfit: ", total_misfit)

    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Pressure misfit (MPa)')
    ax2.set_xlabel('Time (years)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    return total_misfit


# GRADIENT DESCENT
def grad_descent():
    # parameter vector - initial guess of minimum location
    theta0 = np.array([0.00327, 2.496, 0.499])
    # compute steepest descent direction
    s0 = obj_dir(obj, theta0)
    # plot 1: compare against lab2_instructions.pdf, Figure 1 
    plot_s0(obj, theta0, s0)
    return
    
    # choose step size 
    alpha = 0.5
    # update parameter estimate
    theta1 = step(theta0, s0, alpha)
    # plot 2: compare against lab2_instructions.pdf, Figure 2 
    #plot_step(obj, theta0, s0, theta1)
    #return
    
    # Get the new Jacobian for the last parameters estimation
    s1 = obj_dir(obj, theta1)
    # plot 3: compare against lab2_instructions.pdf, Figure 3 
    #plot_s1(obj, theta0, s0, theta1, s1)
    #return
    
    # Plot iterations
    # The following script repeats the process until an optimum is reached, or until the maximum number of iterations allowed is reached
    # Try with different gammas to see how it impacts the optimization process 
    # Uncomment line 47 to use a line search algorithm
    theta_all = [theta0]
    s_all = [s0]
    # iteration control
    N_max = 30
    N_it = 0
    # begin steepest descent iterations
        # exit when max iterations exceeded
    while N_it < N_max:
        # uncomment line below to implement line search
        alpha = line_search(obj, theta_all[-1], s_all[-1])
        # update parameter vector 
        theta_next = step(theta0, s0, alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting
        # compute new direction for line search (thetas[-1]
        s_next = obj_dir(obj, theta_next)
        s_all.append(s_next) 			# save search direction for plotting
        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # restart next iteration with values at end of previous iteration
        theta0 = 1.*theta_next
        s0 = 1.*s_next
    
    print('Optimum a, b, c: ', round(theta_all[-1][0], 2), round(theta_all[-1][1], 2), round(theta_all[-1][2], 3))
    print('Number of iterations needed: ', N_it)

    # plot 4: compare against lab2_instructions.pdf, Figure 4 
    #plot_steps(obj, theta_all, s_all)
    #return theta_all[-1]


# This function plots your model against a benchmark analytic solution.
def plot_benchmark():
    """ Compare analytical and numerical solutions via plotting.

    Parameters:
    -----------
    none

    Returns:
    --------
    none

    """
    # values for benchmark solution
    t0 = 0
    t1 = 10
    dt = 0.1

    # model values for benchmark analytic solution
    a = 1
    b = 1
    c = 1

    # set ambient values to zero for benchmark analytic solution
    p0 = 0
    p1 = 0
    # set inital value to zero for benchmark analytic solution
    pi = 0

    # set extraction rate to a constant
    q0 = 1

    # setup parameters array with constants
    pars = [a, b, c, p0, p1]

    fig, plot = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

    # Solve ODE and plot
    t, p = solve_ode(ode_model, t0, t1, dt, pi, pars)
    plot[0].plot(t, p, "bx", label="Numerical Solution")
    plot[0].set_ylabel("Pressure [MPa]")
    plot[0].set_xlabel("t")
    plot[0].set_title("Benchmark")

    # Analytical Solution
    t = np.array(t)

#   TYPE IN YOUR ANALYTIC SOLUTION HERE
#   if P = 0 at t = 0:
    p_analytical = -((b * p0 + c * p1 - a * q0) / (b + c)) * (1 - np.exp(-(b + c) * t))

    plot[0].plot(t, p_analytical, "r-", label="Analytical Solution")
    plot[0].legend(loc=1)

    # Plot error
    p_error = []
    for i in range(1, len(p)):
        if (p[i] - p_analytical[i]) == 0:
            p_error.append(0)
            print("check line Error Analysis Plot section")
        else:
            p_error.append((np.abs(p[i] - p_analytical[i]) / np.abs(p_analytical[i])))
    plot[1].plot(t[1:], p_error, "k*")
    plot[1].set_ylabel("Relative Error Against Benchmark")
    plot[1].set_xlabel("t")
    plot[1].set_title("Error Analysis")
    plot[1].set_yscale("log")

    # Timestep convergence plot
    time_step = np.flip(np.linspace(1/5, 1, 13))
    for i in time_step:
        t, p = solve_ode(ode_model, t0, t1, i, p0, pars)
        plot[2].plot(1 / i, p[-1], "kx")

    plot[2].set_ylabel(f"Pressure(t = {10})")
    plot[2].set_xlabel("1/\u0394t")
    plot[2].set_title("Timestep Convergence")

    # plot spacings
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def plot_x_forecast():
    ''' Plot the ODE LPM model over the given data plot with different q-value scenario for predictions.
    Use a curve fitting function to accurately define the optimum parameter values.
    Parameters:
    -----------
    none
    Returns:
    --------
    none
    '''

    # Read in time and dependent variable data
    [t, p_exact] = [load_data()[2], load_data()[3]]

    # GUESS PARAMETERS HERE
    pars_guess = [3.944e-08,  5.418, -4.417]

    # Optimise parameters for model fit
    pars, pars_cov = x_pars(pars_guess)

    # Store optimal values for later use
    [a, b, c] = pars

    # Solve ODE and plot model
    p = x_curve_fitting(t, *pars)
    f, ax1 = plt.subplots()
    ax1.plot(t, p_exact, 'r.', label='data')
    ax1.plot(t, p, 'black', label='Model')

    # Remember the last time
    t_end = t[-1]

    # Create forecast time with 20 new time steps
    t1 = []
    for i in range(30):
        t1.append(i + t_end)

    # Set initial and ambient values for forecast
    pi = p[-1]  # Initial value of x is final value of model fit
    p0 = 0.3  # Ambient value of spring pressure
    p_ocean = 0.1  # Ambient value of ocean pressure

    # Solve ODE prediction for scenario 1: 'Continue extracting at an increasing rate as seen from the past five years.'
    # multiplier (5.71e4) is the average increase in q from 2014 - 2019

    # Businesses:
    # 'Extract at an increasing rate as seen for the past 5 years'
    q1 = 2.65e7  # average q from 2014 - 2019
    p1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q1, a, b, c, p0, p_ocean, mp=5.71e4)[1]
    ax1.plot(t1, p1, 'purple', label='Businesses: Increase the extraction rate as seen for the past 5 years.')

    # q1 = 5.71e7  # 2x average q from 2014 - 2019
    # p1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q1, a, b, c, p0, p_ocean)[1]
    # ax1.plot(t1, p1, 'purple', label='extract @ incr rate')

    # Farmers:
    # Solve ODE prediction for scenario 2: 'Extract at the current rate and no higher.'
    q2 = 2.65e7  # average q from 2014 - 2019
    p2 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q2, a, b, c, p0, p_ocean)[1]
    ax1.plot(t1, p2, 'green', label='Farmers: Keep same extraction')

    # # Solve ODE prediction for scenario 3: 'Extract at a reduced rate and continue no higher.'
    # q3 = 1.33e7 # ~half average q from 2014 - 2019
    # p3 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q3, a, b, c, p0, p_ocean)[1]
    # ax1.plot(t1, p3, 'blue', label='extract @ decr rate')

    # Solve ODE prediction for scenario 4: 'Extract at a decreasing rate until pressures stabilise to the level seen
    # between 2000 and 2010.'
    # q4 = 2.5e7 # average q from 2014 - 2019
    # p4 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q4, a, b, c, p0, p_ocean, p_want=0.281, t_want=5, wait=5)[1]
    # ax1.plot(t1, p4, 'red', label='extract @ decr rate until P stable')

    # Iwi:
    # Solve ODE prediction for scenario 5: 'Halt the extraction until pressures stabilise to the level seen between
    # 2000 and 2010.'
    q5 = 0
    p5 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q5, a, b, c, p0, p_ocean, p_want=0.281, t_want=1, wait=0)[1]
    ax1.plot(t1, p5, 'pink', label='Iwi: Halt extraction until stabilised.')

    # Axis information
    ax1.set_title('Pressure Forecast')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (years)')
    ax1.legend()
    plt.show()

# This function computes uncertainty in your model
def plot_x_uncertainty():
    """
    This function plots the uncertainty of the ODE model.
    """

    # read in time and dependent variable data
    [t, p_exact] = [load_data()[2], load_data()[3]]

    # GUESS PARAMETERS HERE
    pars_guess = [3.944e-08,  5.418, -4.417]

    # Optimise parameters for model fit
    pars, pars_cov = x_pars(pars_guess)

    # Store optimal values for later use
    [a,b,c] = pars

    # Solve ODE and plot model
    p = x_curve_fitting(t, *pars)
    figa, ax1 = plt.subplots()
    ax1.plot(t, p_exact, 'r.', label='data')
    ax1.plot(t, p, 'black', label='Model')

    # Remember the last time
    t_end = t[-1]

    # Create forecast time with 400 new time steps
    t1 = []
    for i in range(40):
        t1.append(i+t_end)

    # Set initial and ambient values for forecast
    pi = p[-1] # Initial value of x is final value of model fit
    p0 = 0.3 # Ambient value of
    p_ocean = 0.1

    # q1 = 5.71e7 # 2x average q from 2014 - 2019
    # p1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q1, a, b, c, p0, p_ocean)[1]
    # ax1.plot(t1, p1, 'purple', label='extract @ incr rate')
    #
    # # Solve ODE prediction for scenario 2: 'Extract at the current rate and no higher.'
    q1 = 2.65e7  # average q from 2014 - 2019
    p1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q1, a, b, c, p0, p_ocean, mp=5.71e4)[1]
    ax1.plot(t1, p1, 'purple', label='Businesses')

    # Solve ODE prediction for scenario 3: 'Extract at a reduced rate and continue no higher.'
    q2 = 2.65e7  # average q from 2014 - 2019
    p2 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q2, a, b, c, p0, p_ocean)[1]
    ax1.plot(t1, p2, 'blue', label='Farmers')

    # Solve ODE prediction for scenario 4: 'Extract at a decreasing rate until pressures stabilise to the level seen
    # between 2000 and 2010.'
    # q4 = 2.5e7  # average q from 2014 - 2019
    # p4 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q4, a, b, c, p0, p_ocean, p_want=0.281, t_want=5, wait=5)[1]
    # ax1.plot(t1, p4, 'red', label='extract @ decr rate until P stable')

    # Solve ODE prediction for scenario 5: 'Halt the extraction until pressures stabilise to the level seen between
    # 2000 and 2010.'
    q5 = 0
    p5 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q5, a, b, c, p0, p_ocean, p_want=0.281, t_want=1, wait=0)[1]
    ax1.plot(t1, p5, 'pink', label='no extraction')

    # Estimate the variability of parameter b
    # We are assuming that parameter b has the biggest source of error in the system
    # (you could choose another parameter if you like)

    density = np.random.normal(1000, 14.3736, 500)

    # var = 0.1

    # using Normal function to generate 500 random samples from a Gaussian distribution
    # b_samples = np.random.normal(b, var, 500)
    c_values = -4.417*density/1000
    var = (np.std(c_values))
    print(var)
    c_samples = np.random.normal(c, var, 500)

    # initialise list to count parameters for histograms
    # b_list = []
    c_list = []

    # loop to plot the different predictions with uncertainty
    for i in range(0, 499):  # 500 samples are 0 to 499

        # frequency distribution for histograms for parameters
        # b_list.append(b_samples[i])
        c_list.append(c_samples[i])

        # Solve model fit with uncertainty
        spars = [a, b, c_samples[i]]
        p = x_curve_fitting(t, *spars)
        ax1.plot(t, p, 'black', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 1 with uncertainty
        q1 = 2.65e7  # average q from 2014 - 2019
        p1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q1, a, b, c_samples[i], p0, p_ocean, mp=5.71e4)[1]
        ax1.plot(t1, p1, 'purple', alpha=0.1, lw=0.5)
        #
        # # Solve ODE prediction for scenario 2 with uncertainty
        q2 = 2.65e7  # average q from 2014 - 2019
        p2 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q2, a, b, c_samples[i], p0, p_ocean)[1]
        ax1.plot(t1, p2, 'blue', alpha=0.1, lw=0.5)


        # Solve ODE prediction for scenario 4 with uncertainty
        # q4 = 2.5e7  # average q from 2014 - 2019
        # p4 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q4, a, b, c_samples[i], p0, p_ocean, p_want=0.281,
        #                           t_want=5, wait=5)[1]
        # ax1.plot(t1, p4, 'red', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 5 with uncertainty
        q5 = 0
        p5 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], pi, q5, a, b, c_samples[i], p0, p_ocean, p_want=0.281,
                                  t_want=1, wait=0)[1]
        ax1.plot(t1, p5, 'pink', alpha=0.1, lw=0.5)

    ax1.set_title('Pressure Uncertainty Forecast')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (years)')
    ax1.legend()

    # plotting the histograms
    # figb, (ax2) = plt.subplots(1, 1)
    # num_bins = 30
    # ax2.hist(c_list, num_bins)
    # ax2.set_title("Frequency Density plot for Parameter b", fontsize=9)
    # ax2.set_xlabel('Parameter c', fontsize=9)
    # ax2.set_ylabel('Frequency density', fontsize=9)
    # a_yf5, a_yf95 = np.percentile(c_list, [5, 95])
    # ax2.axvline(a_yf5, label='95% interval', color='r', linestyle='--')
    # ax2.axvline(a_yf95, color='r', linestyle='--')
    # ax2.legend(loc=0, fontsize=9)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
