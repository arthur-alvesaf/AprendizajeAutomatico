# program to solve for constant or line that approximates data in a graph
# import libs
import matplotlib.pyplot as plt

xdata = []
ydata = []

# read data from a file in the same directory named 'data.txt'
for line in open('data.txt'):
	temp = line.rstrip()
	temp = temp.split(',')
	xdata.append(float(temp[0]))
	ydata.append(float(temp[1]))

# global variables
size = len(xdata)

# hypothesis function
def hyp(x, theta):
	return theta[0] + theta[1] * x

# funccion que hace el metodo del gradiente descendiente
def gradienteDescendiente(X, Y, theta, alpha_max, iteraciones):
	for itx in range(0,iteraciones):
		# alpha = alpha_max
		alpha = (1 - (itx/iteraciones)) * alpha_max

		# calculating theta0
		tempsum = 0
		for i in range(0,size):
			tempsum += hyp(X[i], theta) - Y[i]
			pass
		temp0 = theta[0] - alpha/size * tempsum

		# calculating theta1
		tempsum = 0
		for i in range(0,size):
			tempsum += (hyp(X[i], theta) - Y[i]) * X[i]
			pass
		temp1 = theta[1] - alpha/size * tempsum

		# setting global values
		theta[0] = temp0
		theta[1] = temp1
		pass
	return theta

# graph data
def graficaDatos(X, Y, theta):
	# plot points
	plt.plot(X, Y, 'ro')
	# plot line
	plt.plot([min(X),max(X)], [hyp(min(X), theta), hyp(max(X), theta)])
	plt.show()
	return

# calculate error
def calculaCosto(X,Y,theta):
	tempsum = 0
	for i in range(0,size):
		tempsum += pow((hyp(X[i], theta) - Y[i]), 2)
		pass
	error = tempsum/(2*size)
	return error

# run the method to approximate the data
theta = [0.001, 0.001]
alpha = float(input("What initial value of alpha do you want to use? (0.05 is recommended, it will decrease in implementation) "))
n_iterations = int(input("How many iterations do you want to use for the linear regression? (higher is better) "))
theta = gradienteDescendiente(xdata, ydata, theta, alpha, n_iterations)

# print theta values for the line
print("Theta0: ", theta[0], "Theta1: ", theta[1])

# print error
print ("Cost: ", calculaCosto(xdata, ydata, theta))

# graph the data
graficaDatos(xdata, ydata, theta)
