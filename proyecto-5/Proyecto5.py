# Program by Arthur Alves
# ITESM ID A01022593

# importacion de dependencias
import numpy as np
import random as rand

# variables globales
b = 1
default_iteraciones = 5000
emax = 0.08

developer_mode = 0

# funcion que lee de un archivo y separa las entradas y salidas de los ejemplos en arreglos
def leerArchivo(filename):
    X = []
    y = []
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

    X = X.reshape((linen, xparams)).transpose()
    y = y.reshape((linen, 1)).transpose()
    return (X, y)

# funcion que normaliza las entradas
def normalizacionDeCaracteristicas(X):
    mu = []
    sigma = []
    # para cada parametro conseguir la media y la variacion
    for i in range(X.shape[0]):
        mu.append(np.average(X[i]))
        sigma.append(np.amax(X[i]) - np.amin(X[i]))
        for j in range(X.shape[1]):
            X[i][j] = (X[i][j] - mu[i]) / sigma[i]
        pass
    return (X,mu,sigma)

# funciones de activacion
def sigmoidal(x):
    return 1/(1 + np.exp(-x))

def lineal(x):
    return x

# funcion que regresa el costo promedio sigmoidal
def funcionCostoSigmoidal(nn_params, b, X, y):
    J = 0
    X = X.transpose()
    y = y.transpose()
    # print(y, y.shape, y[0])
    m = len(X)
    for i in range(m):
        # -yi*log(sig(xi))-(1-yi)*log(1-sig(xi))
        evalsigmoidal = sigmoidal(np.dot(nn_params.T, X[i]) + b)
        J += -y[i]*np.log(evalsigmoidal)-(1-y[i])*np.log(1-evalsigmoidal)
    J /= m
    return J

# funcion que regresa el promedio del costo lineal
def funcionCostoLineal(nn_params, b, X, y):
    J = 0
    X = X.transpose()
    y = y.transpose()
    # print(y, y.shape, y[0])
    m = len(X)
    for i in range(m):
        J += (y[i] - np.dot(nn_params.T, X[i]) + b)**2
    J /= m
    return J

# funcion que regresa el costo máximo lineal
def funcionCostoMaxLineal(nn_params, b, X, y):
    X = X.transpose()
    y = y.transpose()
    # print(y, y.shape, y[0])
    m = len(X)
    J = abs((y[0] - np.dot(nn_params.T, X[0]) + b)**2)
    for i in range(m):
        temp = abs((y[i] - np.dot(nn_params.T, X[i]) + b)**2)
        if (temp > J):
            J = temp
    return J

#
def bpnUnaNeurona(nn_params, n, X, y, alpha, activacion):
    global b, default_iteraciones, emax

    m = X.shape[1]

    iteraciones = default_iteraciones
    convergido = False
    while not convergido and iteraciones != 0:
        # FW prop
        Z = np.dot(nn_params.T, X) + b
        if (activacion == "sigmoidal"):
            A = sigmoidal(Z)
            J = funcionCostoSigmoidal(nn_params, b, X, y)
        elif (activacion == "lineal"):
            A = lineal(Z)
            J = funcionCostoMaxLineal(nn_params, b, X, y)
        # BW prop
        dZ = A - y
        dW = (1/m)*X.dot(dZ.transpose())
        db = (1/m)*np.sum(dZ)
        nn_params -= alpha * dW
        b -= alpha * db
        iteraciones -= 1
        if J < emax:
            convergido = True
        # print(J)

    if (developer_mode):
        print("Alpha: ", alpha)
        print("Iterations to convergence: ", default_iteraciones-iteraciones)
        print("Final error: ", J)
        print("Weights: ", nn_params.transpose(), " + bias: ", b)
    return nn_params

# funciones de los gradientes
def sigmoidGradiente(z):
    return sigmoidal(z)*(1-sigmoidal(z))

def linealGradiente(z):
    return 1

# funcion que inicializa un arreglo con valores aleatorios entre -0.12 y 0.12
def randInicializaPesos(L_in):
    res = np.empty(L_in)
    for i in range(L_in):
        res[i] = rand.uniform(-0.12, 0.12)
        pass
    res = res.reshape(L_in, 1)
    return res

# funcion que encuentra las salidas de la red neuronal dado una matriz de entradas
def prediceRNYaEntrenada(X, nn_params, activacion):
    Z = np.dot(nn_params.T, X) + b
    if (activacion == "sigmoidal"):
        A = sigmoidal(Z)

        # Round numbers
        A = A.T
        for i in range(len(A)):
            if A[i] < 0.5:
                A[i] = 0
            else:
                A[i] = 1
        A = A.T
    elif (activacion == "lineal"):
        A = lineal(Z)
    return A

# funcion principal
def main():
    # ejemplos de la tabla de verdad AND
    Xand = np.array([[0,0], [0,1], [1,0], [1,1]]).transpose()
    Yand = np.array([[0], [1], [1], [1]]).transpose()

    # inicializacion de los pesos aleatorios
    weights_sig = randInicializaPesos(Xand.shape[0])

    # entrenamiento de la red neuronal para clasificación
    nn_params_sig = bpnUnaNeurona(weights_sig, Xand.shape[0], Xand, Yand, 10, 'sigmoidal')

    # impresion de resultados de la red neuronal para clasificacion de AND
    print("Predicted ŷ from NN for AND: ", prediceRNYaEntrenada(Xand, nn_params_sig, 'sigmoidal'))

    # Get data from file
    (Xhouses, Yhouses) = leerArchivo("data.txt")

    # normalizacion de la matriz X y el vector y
    (Xhouses,mux,sigmax) = normalizacionDeCaracteristicas(Xhouses)
    (Yhouses,muy,sigmay) = normalizacionDeCaracteristicas(Yhouses)

    # inicializacion de los pesos aleatorios
    weights_lin = randInicializaPesos(Xhouses.shape[0])

    # entrenamiento de la red neuronal para prediccion
    nn_params_lin = bpnUnaNeurona(weights_lin, Xand.shape[0], Xhouses, Yhouses, 0.05, 'lineal')

    # prediccion de resultados de la red neuronal para el costo de casas
    ypred = prediceRNYaEntrenada(Xhouses, nn_params_lin, 'lineal')
    # inversa de la normalizacion para conseguir salidas con números veridicos
    ypred = ypred * sigmay + muy
    # impression de resultados de la red neuronal para la prediccion de los precios de casas
    print("Predicted ŷ from NN for housing prices: ", ypred)

    return

if __name__ == "__main__":
    main()
