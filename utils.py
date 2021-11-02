# Autor: Emmanuel Bandres, 14-10071

import csv, numpy, copy

def readfile(filename, header=True, skip_last=True, cast_type=float):
    # Lee un archivo .csv
    # header indica si la primera linea del archivo es un header y lo salta
    # skip_last se utiliza para no castear el ultimo elemento de c/ fila
    # cast_type es el tipo de variable al que se quiere hacer el cast

    with open(filename) as file:
        csvreader = csv.reader(file, delimiter=',')
        if header: csvreader.__next__()

        data = []

        for row in csvreader:
            for i in range(len(row) - int(skip_last)):
                row[i] = cast_type(row[i])
            data.append(row)
    
    return data

def train_test_split(data, test_ratio, target):
    # Recibe una lista con la informacion para dividirla en 4 conjuntos diferentes X, y
    # Tanto de entrenamiento para el perceptron como de prueba

    tmp = copy.deepcopy(data)
    for i in tmp:
        i[-1] = 1 if i[-1] == target else -1

    n = round(len(tmp) * test_ratio)

    test = tmp[:n]
    train = tmp[n:]
    
    xtrain = [i[:-1] for i in train]
    ytrain = [i[-1:] for i in train]
    xtest = [i[:-1] for i in test]
    ytest = [i[-1:] for i in test]

    return numpy.array(xtrain), numpy.array(xtest), numpy.array(ytrain), numpy.array(ytest)

if __name__ == "__main__":
    # testing
    l = readfile('iris.csv')
    
    #train_test_split(l, 0.3)
    xtrain, xtest, ytrain, ytest = train_test_split(l, 0.3)
    print('xtrain', xtrain)
    print('xtest', xtest)
    print('ytrain', ytrain)
    print('ytest', ytest)
    print(len(xtrain), len(ytrain), len(xtest), len(ytest))