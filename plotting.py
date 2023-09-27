
#####################################################################################
#
# Function library for plotting various steps in steepest descent lab
#
# 	Functions:
#		plot_s0: plot initial parameter vector and steepest descent direction
#		plot_step0: plot first step in descent
#		plot_s1: plot first step and next descent direction
#		plot_steps: plot all steps in descent
#
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

J_scale = 0.2
hw = 0.02
hl = 0.02
	
def plot_s0(obj, theta0, s0):
	"""Plot initial parameter vector and steepest descent direction, overlaid on contours of objective function.

    Args:
		obj (callable): objective function
		theta0 (list): initial parameter vector
		s0 (list): initial steepest descent direction
    """
	# create plotting grid
	N = 501								# number grid points
	x = np.linspace(-1, 1., N)			# vector in x direction
	y = np.linspace(-1., 1., N)			# vector in y direction
	xv, yv = np.meshgrid(x, y)			# generate meshgrid
	
	# compute objective function on grid
	Z = np.zeros(xv.shape) 				# empty matrix
	for i in range(len(y)):					# for each x value
		for j in range(len(x)):					# for each y value
			Z[i][j] = obj([xv[i][j], yv[i][j]])		# compute objective function
	
	# plotting
	plt.clf()
	ax1 = plt.axes()			# create axes
	ax1.contourf(xv, yv, Z, 21, alpha = .8, cmap=cm.jet)		# contours of objective function
	ax1.scatter(theta0[0], theta0[1], color='k', s = 20.)		# show parameter steps
	ax1.arrow(theta0[0], theta0[1], -J_scale*s0[0], -J_scale*s0[1], head_width=hw, head_length=hl)	# show descent direction
	
	# plotting upkeep
	ax1.set_xlim(-1.,1.)
	ax1.set_ylim(-1.,1.)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_aspect('equal')
	
	# save and show
	plt.savefig('lab2_plot1.png', bbox_inches = 'tight')
	plt.show()
	
def plot_step(obj, theta0, s, theta1):
	"""Plot first step in descent, overlaid on contours of objective function.

    Args:
		obj (callable): objective function
		theta0 (list): initial parameter vector
		s (list): initial steepest descent direction
		theta1 (list): parameter vector at end of first step
    """
	# create plotting grid
	N = 501								# number grid points
	x = np.linspace(-1, 1., N)			# vector in x direction
	y = np.linspace(-1., 1., N)			# vector in y direction
	xv, yv = np.meshgrid(x, y)			# generate meshgrid
	
	# compute objective function on grid
	Z = np.zeros(xv.shape) 				# empty matrix
	for i in range(len(y)):					# for each x value
		for j in range(len(x)):					# for each y value
			Z[i][j] = obj([xv[i][j], yv[i][j]])		# compute objective function

	# plotting
	plt.clf()
	ax1 = plt.axes()		# create axes
	ax1.contourf(xv, yv, Z, 21, alpha = .8, cmap=cm.jet)		# contours of objective function
	# show parameter steps
	ax1.scatter([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', s = 20.)
	ax1.plot([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', linestyle = '--')
	# show descent direction
	ax1.arrow(theta0[0], theta0[1], -J_scale*s[0], -J_scale*s[1], head_width=hw, head_length=hl)
	
	# plotting upkeep
	ax1.set_xlim(-1.,1.)
	ax1.set_ylim(-1.,1.)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_aspect('equal')
	
	# save and show
	plt.savefig('lab2_plot2.png', bbox_inches = 'tight')
	plt.show()
	
def plot_s1(obj, theta0, s0, theta1, s1):
	"""Plot first step and next descent direction, overlaid on contours of objective function.

    Args:
		obj (callable): objective function
		theta0 (list): initial parameter vector
		s0 (list): initial steepest descent direction
		theta1 (list): parameter vector at end of first step
		s1 (list): descent direction at end of first step
    """
	# create plotting grid
	N = 501								# number grid points
	x = np.linspace(-1, 1., N)			# vector in x direction
	y = np.linspace(-1., 1., N)			# vector in y direction
	xv, yv = np.meshgrid(x, y)			# generate meshgrid
	
	# compute objective function on grid
	Z = np.zeros(xv.shape) 				# empty matrix
	for i in range(len(y)):					# for each x value
		for j in range(len(x)):					# for each y value
			Z[i][j] = obj([xv[i][j], yv[i][j]])		# compute objective function

	# plotting
	plt.clf()
	ax1 = plt.axes()		# create axes
	ax1.contourf(xv, yv, Z, 21, alpha = .8, cmap=cm.jet)		# contours of objective function
	# show parameter steps
	ax1.scatter([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', s = 20.)
	ax1.plot([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', linestyle = '--')
	# show descent directions
	ax1.arrow(theta0[0], theta0[1], -J_scale*s0[0], -J_scale*s0[1], head_width=hw, head_length=hl)
	ax1.arrow(theta1[0], theta1[1], -J_scale*s1[0], -J_scale*s1[1], head_width=hw, head_length=hl)
	
	# plotting upkeep
	ax1.set_xlim(-1.,1.)
	ax1.set_ylim(-1.,1.)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_aspect('equal')
	
	# save and show
	plt.savefig('lab2_plot3.png', bbox_inches = 'tight')
	plt.show()
	
def plot_steps(obj, theta_all, s_all):
	"""Plot all steps in descent, overlaid on contours of objective function.

    Args:
		obj (callable): objective function
		theta_all (list): list of parameter vector updates during descent
		s_all (list): list of descent directions
    """
	# create plotting grid
	N = 501								# number grid points
	x = np.linspace(-1, 1., N)			# vector in x direction
	y = np.linspace(-1., 1., N)			# vector in y direction
	xv, yv = np.meshgrid(x, y)			# generate meshgrid
	
	# compute objective function on grid
	Z = np.zeros(xv.shape) 				# empty matrix
	for i in range(len(y)):					# for each x value
		for j in range(len(x)):					# for each y value
			Z[i][j] = obj([xv[i][j], yv[i][j]])		# compute objective function

	# plotting
	plt.clf()
	ax1 = plt.axes()		# create axes
	ax1.contourf(xv, yv, Z, 21, alpha = .8, cmap=cm.jet)		# contours of objective function
	# show parameter steps
	ax1.scatter([theta[0] for theta in theta_all[1:-1]], [theta[1] for theta in theta_all[1:-1]], color='k', linestyle = '--')
	ax1.scatter(theta_all[0][0], theta_all[0][1], color='b', linestyle = '--', label = 'Initial values')
	ax1.scatter(theta_all[-1][0], theta_all[-1][1], color='g', linestyle = '--', label = 'Final values')
	ax1.plot([theta[0] for theta in theta_all], [theta[1] for theta in theta_all], color='k', linestyle = '--')
	# show decent directions
	for i in range(len(theta_all)-1):
		ax1.arrow(theta_all[i][0], theta_all[i][1], -J_scale*s_all[i][0], -J_scale*s_all[i][1], head_width=hw, head_length=hl)
	
	# plotting upkeep
	ax1.set_xlim(-1.,1.)
	ax1.set_ylim(-1.,1.)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_aspect('equal')
	
	# save and show
	plt.savefig('lab2_plot4.png', bbox_inches = 'tight')
	plt.show()



	
	

	
	