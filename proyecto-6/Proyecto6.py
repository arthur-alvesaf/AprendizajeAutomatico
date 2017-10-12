# Proyecto desarrollado por Arthur Alves
# Matricula ITESM A01022593

import numpy as np
import matplotlib.pyplot as plt
import random as rand

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Yand = np.array([[0], [1], [1], [1]])
Yor = np.array([[0], [0], [0], [1]])

# hiperparametros y variables globales
alpha = 0.01
nfeat = X.shape[1]
maxerror = 0.1
J_Hist = np.empty(0)


# Funciones de activacion
def escalon(x):
    if (x < 0):
        return 0
    elif (x == 0):
        return 0.5
    else:
        return 1

def sigmoidal(x):
    return 1/(1 + np.exp(-x))

def lineal(x):
    return x

# Entrada: pesos theta, matriz de entradas X, vector de salidas y
# Salida: funcion de costo J y el gradiente
def funcionCostoPerceptron(theta, X, y):
    J = 0
    error = 0
    grad = 0
    idxgrad = 0
    for ejidx in range(0, X.shape[0]):
        xi = X[ejidx]
        di = y[ejidx]
        net = theta.dot(xi)
        yt = escalon(net)
        for i in range(0, nfeat+1):
            error += (di - yt) * xi[i]
            idxgrad += (di - yt) * xi[i]
        error /= nfeat+1
        idxgrad /= nfeat+1
        J += error
        grad += idxgrad
    J /= X.shape[0]
    grad /= X.shape[0]
    return [J, grad]

# Entrada: vector X con los ejemplos, y con las salidas deseadas y theta que es el vector de pesos
# Salida: los pesos del perceptron
def entrenaPerceptron(X, y, theta):
    global alpha, nfeat, maxerror, J_Hist
    examples = X.shape[0]
    error = 100
    while (error >= maxerror):
        error = 0
        for ejidx in range(0, examples):
            xi = X[ejidx]
            di = y[ejidx]
            net = theta.dot(xi)
            yt = escalon(net)

            for i in range(0, nfeat+1):
                iterr = (di - yt) * xi[i]
                theta[i] = theta[i] + alpha * iterr
                if (abs(iterr) > error):
                    error = abs(iterr)
        J_Hist = np.append(J_Hist, error)

    # print(J_Hist)
    return theta

# Entradas: theta (pesos) y vector X con muchos ejemplos
# Salida: vector p de predicciones de los ejemplos en x
def predicePerceptron(theta, X):
    p = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
        xi = X[i]
        net = theta.dot(xi)
        yt = escalon(net)
        p[i] = yt
    return p

# Entrada: pesos theta, matriz de entradas X, vector de salidas y
# Salida: funcion de costo J y el gradiente
def funcionCostoAdaline(theta, X, y):
    J = 0
    error = 0
    grad = 0
    idxgrad = 0
    for ejidx in range(0, X.shape[0]):
        xi = X[ejidx]
        di = y[ejidx]
        net = theta.dot(xi)
        yt = escalon(net)
        for i in range(0, nfeat+1):
            error += (di - net)
            idxgrad += (di - yt) * xi[i]
        error = (1/X.shape[0]) * (error**2)
        idxgrad /= nfeat+1
        J += error
        grad += idxgrad
    J /= X.shape[0]
    grad /= X.shape[0]
    return [J, grad]

# Entrada: vector X con los ejemplos, y con las salidas deseadas y theta que es el vector de pesos
# Salida: los pesos del adaline
def entrenaAdaline(X, y, theta):
    global alpha, nfeat, maxerror, J_Hist
    examples = X.shape[0]
    error = 100
    while (error >= maxerror):
        error = 0

        for ejidx in range(0, examples):
            xi = X[ejidx]
            di = y[ejidx]
            net = xi.dot(theta.transpose())
            # yt = escalon(net)

            # actualizar pesos
            for i in range(0, nfeat+1):
                theta[i] = theta[i] + alpha * (di - net) * xi[i]

            error += ((di - net) ** 2)
        error = error * (1/examples)
        # print (theta, error)

        J_Hist = np.append(J_Hist, error)

    # print(J_Hist)
    return theta

# Entradas: theta (pesos) y vector X con muchos ejemplos
# Salida: vector p de predicciones de los ejemplos en x
def prediceAdaline(theta, X):
    p = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
        xi = X[i]
        net = theta.dot(xi)
        yt = escalon(net)
        p[i] = yt
    return p

def main():
    global nfeat, X
    # inicializar theta de forma aleatoria
    theta = np.zeros(nfeat+1)
    theta[0:theta.size] = rand.random()

    # agregar umbral a vector de x
    X = np.append(np.ones(X.shape[0]).reshape(X.shape[0], 1), X, axis=1)

    nntype = int(input("What type of neural network would you like to implement? Write 1 for perceptron and 2 for adaline. _"))
    nninput = int(input("With which values would you like to train the neural network? Write 1 to train it for AND, and 2 to train it for OR. _"))
    wronginputs = 0
    if (nntype == 1):
        if (nninput == 1):
            entrenaPerceptron(X, Yand, theta)
        elif (nninput == 2):
            entrenaPerceptron(X, Yor, theta)
        else:
            print(str(nninput) + " is not a valid option for training.")
            wronginputs = 1
    elif (nntype == 2):
        if (nninput == 1):
            entrenaAdaline(X, Yand, theta)
        elif (nninput == 2):
            entrenaAdaline(X, Yor, theta)
        else:
            print(str(nninput) + " is not a valid option for training.")
            wronginputs = 1
    else:
        print(str(nntype) + " is not a valid option of neural network implementation.")
        wronginputs = 1
    if (wronginputs == 0):
        if (nntype == 1):
            print("Outputs from Perceptron: ", predicePerceptron(theta, X))
            print("Perceptron weights: ", theta)
        elif (nntype == 2):
            print("Outputs from Adaline: ", prediceAdaline(theta, X))
            print("Adaline weights: ", theta)

    plt.plot(J_Hist)
    plt.show()
    return

if __name__ == "__main__":
    main()
