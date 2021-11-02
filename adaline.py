# Autor: Emmanuel Bandres, 14-10071

import numpy, random

numpy.seterr(all='raise')
class Adaline:
    # Clase de Perceptron
    # Se inicializa con la tasa de aprendizaje lr y el numero maximo de epocas
    def __init__(self, lr, n_epochs):
        self.lr = lr
        self.n_epochs = n_epochs
        self.b = 0.0
        self.w = []
        self.misclassified = []

    def net_input(self, X):
        return numpy.dot(X, self.w) + self.b

    def predict(self, X):
        return numpy.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w = [random.uniform(-0.05, 0.05) for i in range(len(X[0]))]

        for epoch in range(self.n_epochs):
            cnt = 0
            for xi, target in zip(X, y):
                cnt += 1
                net_inputs = self.net_input(xi)
                update = target - net_inputs
                try:
                    self.w += self.lr * xi * update
                except FloatingPointError:
                    print(f"target: {target}")
                    print(f"ni: {net_inputs}")
                    print(f"update: {update}")
                    exit()
                self.b += self.lr * update.sum()
                
