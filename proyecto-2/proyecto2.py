# program to solve for constant or line that approximates data in a graph
# program by Arthur Alves A01022593

# import libs
import numpy as np
from numpy.linalg import inv
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

# hypothesis function
def hip(X, theta):
    # Xones = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    return theta.transpose().dot(X)

# funcion que normaliza los valores en X usando la normalizacion media
def normalizacionDeCaracteristicas(X):
    mu = []
    sigma = []
    X = np.transpose(X)
    # para cada parametro conseguir la media y la variacion
    for i in range(X.shape[0]):
        mu.append(np.average(X[i]))
        sigma.append(np.amax(X[i]) - np.amin(X[i]))
        for j in range(X.shape[1]):
            X[i][j] = (X[i][j] - mu[i]) / sigma[i]
        pass
    X = np.transpose(X)
    return (X,mu,sigma)

# funcion que aplica el metodo de gradiente descendiente al conjunto de datos
def gradienteDescendenteMultivariable(X,y,theta,alpha,iteraciones):
    J_Historial = []

    # agrega la columna de 1's a la izq y hace la transpuesta para tener los datos horizontalmente
    Xones = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

    for itx in range(iteraciones):
        temp = np.zeros(Xones.shape[1])
        for j in range(Xones.shape[1]):
            temp[j] = theta[j] - (alpha/X.shape[0]) * sum((hip(xi, theta) - yi) * xi[j] for (xi, yi) in zip(Xones, y))
            pass
        theta = temp
        J_Historial = np.append(J_Historial, calculaCosto(Xones, y, theta))
        pass

    return (theta,J_Historial)

# funcion que encuentra el conjunto de theta con vectorizacion
def ecuacionNormal(X, y):
    # inserta la fila de 1's para la vectorizacion
    Xones = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    Xt = np.transpose(Xones)
    return ( inv( Xt.dot(Xones) ) ).dot( Xt.dot(y) )

# funcion que grafica el error del metodo del gradiente descendiente
def graficaError(J_Historial):
    itx = np.arange(J_Historial.size)
    plt.plot(itx, J_Historial)
    plt.show()
    return

# funcion que calcula el costo
def calculaCosto(X, y, theta):
    # X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    return 1/(2*X.shape[0]) * sum([pow(hip(xi, theta) - yi, 2) for (xi, yi) in zip(X, y)])

# funcion que predice el precio
def predicePrecio(X, theta):
    return hip(np.append([1], X), theta)

# popular variables
leerArchivo("ex1data2.txt")

# encontrar theta con gradiente descendente normalizando valores
(Xn, mu, sigma) = normalizacionDeCaracteristicas(X)
(theta, J_Historial) = gradienteDescendenteMultivariable(Xn, y, np.array([0,0,0]), 0.1, 100)

# encontrar theta con la ecuacion normal
theta = ecuacionNormal(X, y)

# graficar el error
graficaError(J_Historial)
