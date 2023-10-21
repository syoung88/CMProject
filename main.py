from ode import *

if __name__ == "__main__":
    # benchmarking for ODE
    # plot_benchmark()

    #grad_descent()

    # ODE model with initial parameter values
    # plot_suitable()

    # ODE model with improved parameter values from curve_fit
    # plot_improve(0.00327, 0.147, 0.0147)

    # Try and find best b and c values
    # min_misfit = np.inf
    # best_b = 0
    # best_c = 0
    # for b in np.arange(0, 1, 0.1):
    #     for c in np.arange(0, 1, 0.1):
    #         try:
    #             total_misfit = plot_improve(0.00327, b, c)
    #             if total_misfit < min_misfit:
    #                 min_misfit = total_misfit
    #                 best_b = b
    #                 best_c = c
    #         except:
    #             pass
    # print("Best b: ", best_b)
    # print("Best c: ", best_c)

    # plot_x_forecast()
    plot_x_uncertainty()
