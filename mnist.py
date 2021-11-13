# Autor: Emmanuel Bandres, 14-10071

import utils, adaline, random, numpy

def split_col(lst, index):
    x = [i[:index] + i[index+1:] for i in lst]
    y = [i[index:index+1] for i in lst]
    return numpy.array(x), numpy.array(y)

if __name__ == '__main__':
    # Variables para el perceptron
    learning_rate = 0.1
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
    # Inicializamos y entrenamos los adaline
    classifier = adaline.Adaline(learning_rate, max_epochs, 10, print_progress=True)
    classifier.fit_multi(xtrain, ytr)

    print("Done. Predicting...")
    # Pasamos los datos de prueba para obtener las predicciones y las comparamos
    scores = []
    for xi in xtest:
        scores += [[classifier.net_input(xi, n) for n in range(10)]]
    
    predictions = [score.index(max(score)) for score in scores]
    
    # Calculamos la precision
    accuracy = 0
    for prediction, target in zip(predictions, yts):
        if prediction == target: accuracy += 1
    
    print(f"Accuracy: {accuracy}/{len(yts)} = {accuracy/len(yts)}")

