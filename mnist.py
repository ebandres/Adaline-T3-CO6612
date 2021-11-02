# Autor: Emmanuel Bandres, 14-10071

import utils, adaline, random, numpy

def split_col(lst, index):
    x = [i[:index] + i[index+1:] for i in lst]
    y = [i[index:index+1] for i in lst]
    return numpy.array(x), numpy.array(y)

if __name__ == '__main__':
    # Variables para el perceptron
    learning_rate = 0.001
    max_epochs = 50

    # Imprimimos el progreso para que se sepa que no quedo atascado...

    print("Preprocessing...")
    print("Reading training data...")
    train_data = utils.readfile('mnist_train.csv', header=False, skip_last=False, cast_type=int)
    print("Done. Reading test data...")
    test_data = utils.readfile('mnist_test.csv', header=False, skip_last=False, cast_type=int)
    print("Done.")
    random.shuffle(train_data)

    print("Splitting...")
    # Tomamos la primera columna que contiene los digitos correspondientes a cada caso
    xtrain, ytr = split_col(train_data, 0)
    xtest, yts = split_col(test_data, 0)

    xtrain = xtrain/255
    xtest = xtest/255

    # Creamos una lista de 10 arreglos con los digitos correspondientes a cada perceptron
    ytrain = []
    for i in range(10):
        ytrain += [numpy.where(ytr == i, 1, -1)]

    print("Preprocessing done. Training...")
    # Inicializamos y entrenamos los perceptrones
    classifiers = []
    for i in range(10):
        print(f"Training Perceptron {i}")
        classifiers += [adaline.Adaline(learning_rate, max_epochs)]
        classifiers[i].fit(xtrain, ytrain[i])

    print("Done. Predicting...")
    # Pasamos los datos de prueba para obtener las predicciones y las comparamos
    predictions = []
    for i in range(10):
        print(f"Predicting with Perceptron {i}")
        predictions += [classifiers[i].predict(xtest)]
    
    # Calculamos la precision
    # inconclusive indica que un digito clasifico como dos o mas a la vez
    accuracy, inconclusive = 0, 0
    results = []
    for i in range(len(yts)):
        # Creamos una tupla como pide el enunciado, pero esto no es necesario
        results += [tuple([1 if l[i] == 1 else 0 for l in predictions])]
        if sum(results[i]) == 1:
            idx = results[i].index(1)
            if idx == yts[i][0]:
                accuracy += 1
        else:
            inconclusive += 1
    
    print(f"Inconclusive: {inconclusive}\nAccuracy: {accuracy}/{len(yts)} = {accuracy/len(yts)}")

