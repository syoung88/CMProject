import warnings
import numpy as np
import math
import sklearn
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from sklearn.linear_model import BayesianRidge


# This function defines your ODE.
def ode_model(t, p, q, a, b, c, p0, p1):
    """ Return the derivative dx/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable time.
        x : float
            Dependent variable (pressure or temperature)
        q : float
            mass injection/ejection rate.
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.
        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable time.
        Notes:
        ------
        None
    """
    # equation to return the derivative of dependent variable with respect to time

    dpdt = a*q - b*(p - p0) - c*(p - p1)

    return dpdt


# This function loads in your data.
def load_data():
    """ Load data throughout the time period.
    Parameters:
    -----------
    Returns:
    ----------
    t_q : array-like
        Vector of times at which measurements of q were taken.
    q : array-like
        Vector of q (units)
    t_x : array-like
        Vector of times at which measurements of x were taken.
    x : array-like
        Vector of x (units)
    """
    # Load kettle data
    t_step, p = np.genfromtxt('P_acquifer.csv', delimiter=',', skip_header=7).T
    t_step_1, q = np.genfromtxt('q_acquifer.csv', delimiter=',', skip_header=7).T

    # Pressure of the aquifer
    for pi in p:
        pi += 0.101

    return t_step, p, t_step_1, q


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
def x_curve_fitting(t, a, b, c, const):
    """ Function designed to be used with scipy.optimize.curve_fit which solves the ODE using the Improved Euler Method.
        Parameters:
        -----------
        t : array-like
            Independent time variable vector
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        Returns:
        --------
        x : array-like
            Dependent variable solution vector.
        """
    # model parameters
    pars = [a, b, c, const]

    # ambient value of dependent variable
    p0 = 22

    # time vector information
    n = len(t)
    dt = t[1] - t[0]

    # read in time and dependent variable information
    [t, p_exact] = [load_data()[0], load_data()[1]]

    # initialise p
    p = [p_exact[0]]

    # read in q data
    [t_q, q] = [load_data()[2], load_data()[3]]

    # using interpolation to find the injection rate at each point in time
    q = np.interp(t, t_q, q)

    # using the improved euler method to solve the ODE
    for i in range(n - 1):
        f0 = ode_model(t[i], p[i], q[i], *pars, p0)
        f1 = ode_model(t[i] + dt, p[i] + dt * f0, q[i], *pars, p0)
        p.append(p[i] + dt * (f0 / 2 + f1 / 2))

    return p


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
           Array consisting of a: mass injection strength parameter, b: recharge strength parameter
    """
    # read in time and dependent variable data
    [t_exact, p_exact] = [load_data()[0], load_data()[1]]

    # finding model constants in the formulation of the ODE using curve fitting
    # optimised parameters (pars) and covariance (pars_cov) between parameters
    pars, pars_cov = curve_fit(x_curve_fitting, t_exact, p_exact, pars_guess)
 
    return pars, pars_cov


# This function solves your ODE using Improved Euler for a future prediction with new q
def solve_ode_prediction(f, t0, t1, dt, pi, q, a, b, c, x0, const):
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
    xi : float
        Initial value of solution.
    q : array
        Values of extraction over time.
    a : float
        mass injection strength parameter.
    b : float
        recharge strength parameter.
    x0 : float
        Ambient value of solution.
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
    # finding the number of time steps
    tspan = t1 - t0
    n = int(tspan // dt)

    # initialising the time and solution vectors
    p = [pi]
    t = [t0]

    # using the improved euler method to solve the pressure ODE
    for i in range(n):
        f0 = f(t[i], p[i], q, a, b, c, x0, const)
        f1 = f(t[i] + dt, p[i] + dt * f0, q, a, b, x0)
        p.append(p[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return t, p


# This function plots your model over the data using your estimate for a and b
def plot_suitable():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and temperature data
    [t, p_exact] = [load_data()[0], load_data()[1]]

    # TYPE IN YOUR PARAMETER ESTIMATE FOR a AND b HERE
    pars = [1/(4.186*997*0.5), 0.0005, 0.0005, 0]
  
    # solve ODE with estimated parameters and plot 
    p = x_curve_fitting(t, *pars)
    ax1.plot(t, p_exact, 'k.', label='Observation')
    ax1.plot(t, p, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (sec)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = p
    for i in range(len(p)):
        misfit[i] = p_exact[i] - p[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Temp misfit (C)')
    ax2.set_xlabel('Time (sec)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# This function plots your model over the data using your improved model after curve fitting.
def plot_improve():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and temperature data
    [t, x_exact] = [load_data()[0], load_data()[1]]

    # TYPE IN YOUR PARAMETER GUESS FOR a AND b HERE AS A START FOR OPTIMISATION
    pars_guess = [1/(4.186*997*0.5), 0.0005, 0.0005, 0]
    
    # call to find out optimal parameters using guess as start
    pars, pars_cov = x_pars(pars_guess)

    # check new optimised parameters
    print("Improved a and b")
    print(pars[0], pars[1])

    # solve ODE with new parameters and plot 
    x = x_curve_fitting(t, *pars)
    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Temp (C)')
    ax1.set_xlabel('Time (sec)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Temp misfit (C)')
    ax2.set_xlabel('Time (sec)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


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

    # set ambient value to zero for benchmark analytic solution
    p0 = 0
    p1 = 0
    # set initial value to zero for benchmark analytic solution
    pi = 0

    # set the constant of integration
    const = 1

    # read in q data
    [t_q, q] = [load_data()[2], load_data()[3]]

    # using interpolation to find the injection rate at each point in time
    q = np.interp(t, t_q, q)

    # setup parameters array with constants
    pars = [a, b, c, p0, p1, const]
    fig, plot = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

    # Solve ODE and plot
    t, p = solve_ode(ode_model, t0, t1, dt, pi, pars)
    plot[0].plot(t, p, "bx", label="Numerical Solution")
    plot[0].set_ylabel("Pressure")
    plot[0].set_xlabel("t")
    plot[0].set_title("Benchmark")

    # Analytical Solution
    t = np.array(t)

#   TYPE IN YOUR ANALYTIC SOLUTION HERE (removed negative bc solution is absolute value)
    p_analytical = (b*p0 + c*p1 - a*q - const*np.exp(-t))/(b + c)

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
        t, x = solve_ode(ode_model, t0, t1, i, p0, pars)
        plot[2].plot(1 / i, p[-1], "kx")

    plot[2].set_ylabel(f"Temp(t = {10})")
    plot[2].set_xlabel("1/\u0394t")
    plot[2].set_title("Timestep Convergence")

    # plot spacings
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()


