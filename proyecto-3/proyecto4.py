# program to solve for constant or line that approximates data in a graph
# program by Arthur Alves A01022593

# import libs
import numpy as np
from numpy.linalg import inv
from math import exp
import matplotlib.pyplot as plt

X = []
y = []

# funcion que lee un archivo y guarda la informacion en las variables X y y
def leerArchivo(filename):
    global X, y
    xparams = 0
    linen = 0
    for line in open(filename):
        temp = line.rstrip()
        temp = temp.split(',')

        if xparams == 0:
            xparams = len(temp) - 1

        xdata = []
        for i in range(xparams):
            xdata.append(float(temp[i]))
        ydata = float(temp[xparams])

        X = np.append(X, xdata)
        y = np.append(y, ydata)
        linen = linen + 1
        pass

    X = X.reshape((linen, xparams))
    y = y.reshape((linen, 1))

# funcion sigmoidal
def sigmoidal(z):
    return 1 / (1 + np.e**(-z))

# hypothesis function
def hip(X, theta):
    return sigmoidal(theta.transpose().dot(X))

def normalizacionSimple(Xinput):
    Xinput = Xinput.copy()
    Xinput = Xinput/100
    return Xinput

# funcion que normaliza los valores en X usando la normalizacion media
def normalizacionMedia(Xinput):
    m = Xinput.ndim
    mu = []
    sigma = []
    Xinput = Xinput.copy().transpose()
    # para cada parametro conseguir la media y la variacion
    for i in range(m):
        mu.append(np.average(Xinput[i]))
        sigma.append(np.amax(Xinput[i]) - np.amin(Xinput[i]))
        for j in range(Xinput.shape[1]):
            Xinput[i][j] = (Xinput[i][j] - mu[i]) / sigma[i]
        pass
    Xinput = Xinput.transpose()
    return (Xinput,mu,sigma)

# funcion que calcula el costo
def funcionCosto(theta, X, y):
    m = X.shape[0]

    h = np.array(list(sigmoidal(z) for z in X.dot(theta)))
    cost = (1/m) * (- y.transpose().dot(np.log(h)) - (1-y).transpose().dot(np.log(1-h)))

    grad = (1/m) * X.transpose().dot(h - y)

    return (cost, grad)

# funcion que aplica el metodo de gradiente descendiente al conjunto de datos
def aprende(theta, X, y, iteraciones):
    alpha = 0.001
    m = X.shape[0]
    J_Hist = []
    # agrega la columna de 1's a la izq y hace la transpuesta para tener los datos horizontalmente
    Xones = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    bestJ = float('Inf')

    # for que cambia alpha y prueba el error
    for run in range(3):
        # for que recorre el numero de iteraciones
        temp = theta.copy()
        for itx in range(iteraciones):
            (J, grad) = funcionCosto(temp, Xones, y)

            temp = temp - (alpha * grad)

            J_Hist = np.append(J_Hist, J)
            pass

        (J, grad) = funcionCosto(temp, Xones, y)
        J_Hist = np.append(J_Hist, J)

        # aceptar variable temporal como theta si su error es menor que los anteriores
        if (J < bestJ):
            theta = temp
            bestJ = J
        alpha = alpha * 10
        pass

    # print(J_Hist.reshape(int(J_Hist.shape[0]/3), 3))
    # print(bestJ)
    return theta

def graficaDatos(X, y, theta):
    # scatter points = circles for accepted x for not accepted
    for x,y in zip(X,y):
        plt.scatter(x[0],x[1], marker = "o" if y else "x", color = 'red')

    # plot line
    Xrange = [np.amin(X), np.amax(X)]
    lineY = list((0.5 - theta[0] - theta[1]*xi)/theta[2] for xi in Xrange)
    plt.plot(Xrange, lineY, color = 'blue')
    plt.show()
    return

# funcion que predice la probabilidad de 1
def predice(theta, X):
    m = X.ndim
    Xones = np.insert(X, 0, np.ones(m), axis=1)
    p = np.zeros(m)
    y = hip(Xones.transpose(), theta).transpose()
    for i in range(m):
        if (y[i] >= 0.5):
            p[i] = 1
        else:
            p[i] = 0
        pass
    return p

# popular variables
leerArchivo("ex2data1.txt")

# encontrar theta con gradiente descendente normalizando valores
Xn = normalizacionSimple(X)
theta = aprende(np.zeros(3).reshape(3, 1), Xn, y, 1000)

# graficar datos
graficaDatos(Xn, y, theta)

# predice
print(predice(theta, np.array([[45, 85], [0,0]])))
