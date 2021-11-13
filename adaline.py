# Autor: Emmanuel Bandres, 14-10071

import numpy, random

numpy.seterr(all='raise')
class Adaline:
    # Clase de Adaline
    # Se inicializa con la tasa de aprendizaje lr y el numero maximo de epocas
    def __init__(self, lr, n_epochs, n_neurons=1, print_progress=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.b = 0.0
        self.w = []
        self.misclassified = []
        self.n_neurons = n_neurons
        self.print_progress = print_progress

    def net_input(self, X, n=0, multi=False):
        if multi:
            return numpy.dot(X, self.w[n,:]) + self.b
        else:
            return numpy.dot(X, self.w) + self.b

    def predict(self, X):
        return numpy.where(self.net_input(X) >= 0.0, 1, -1)

    def fit_multi(self, X, y):
        # Para esta tarea esta funcion sirve especificamente para los datos del mnist...
        self.w = numpy.random.uniform(-0.05, 0.05,(self.n_neurons, len(X[0])))
        
        for epoch in range(self.n_epochs):
            if self.print_progress and epoch % 10 == 0: print(f"{epoch*100 / self.n_epochs}%")
            
            for xi, target in zip(X, y):
                vector = [1 if i == target else -1 for i in range(self.n_neurons)]
                sums = []

                for n in range(self.n_neurons):
                    net_inputs = self.net_input(xi, n, True)
                    sums.append(net_inputs)

                for n in range(self.n_neurons):
                    update = vector[n] - sums[n]
                    self.w[n,:] += self.lr * xi * update
                    self.b += self.lr * update.sum()

    def fit(self, X, y):
        self.w = [random.uniform(-0.05, 0.05) for i in range(len(X[0]))]

        for epoch in range(self.n_epochs):
            cnt = 0
            for xi, target in zip(X, y):
                cnt += 1
                net_inputs = self.net_input(xi)
                update = target - net_inputs
                self.w += self.lr * xi * update
                self.b += self.lr * update.sum()

                
