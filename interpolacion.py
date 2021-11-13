# Autor: Emmanuel Bandres, 14-10071

import utils, adaline, random, numpy
import matplotlib.pyplot as plt

def split_col(lst, index):
    x = [i[:index] + i[index+1:] for i in lst]
    y = [i[index:index+1] for i in lst]
    return numpy.array(x), numpy.array(y)

if __name__ == '__main__':
    # Variables para el perceptron
    learning_rate = 0.001
    max_epochs = 100

    # Imprimimos el progreso para que se sepa que no quedo atascado...

    print("Preprocessing...")
    print("Reading training data...")
    train_data = utils.readfile('interpolacionpolinomial_train.csv', header=False, skip_last=False, cast_type=float)
    print("Done. Reading test data...")
    test_data = utils.readfile('interpolacionpolinomial_test.csv', header=False, skip_last=False, cast_type=float)
    real_data = utils.readfile('interpolacionpolinomial_real.csv', header=False, skip_last=False, cast_type=float)
    print("Done.")
    random.shuffle(train_data)

    print("Splitting...")
    # Tomamos la primera columna que contiene los digitos correspondientes a cada caso
    xtrain, ytr = split_col(train_data, 1)
    xtest, yts = split_col(test_data, 1)
    xreal, yreal = split_col(real_data, 1)
    classifier_3 = adaline.Adaline(learning_rate, max_epochs)
    classifier_5 = adaline.Adaline(learning_rate, max_epochs)
    classifier_7 = adaline.Adaline(learning_rate, max_epochs)

    xtrain_3, xtrain_5, xtrain_7 = [], [], []
    xtest_3, xtest_5, xtest_7 = [], [], []
    
    for row in xtrain:
        xtrain_3 += [[row[0], row[0]**2, row[0]**3]]
        xtrain_5 += [[row[0], row[0]**2, row[0]**3, row[0]**4, row[0]**5]]
        xtrain_7 += [[row[0], row[0]**2, row[0]**3, row[0]**4, row[0]**5, row[0]**6, row[0]**7]]

    for row in xtest:
        xtest_3 += [[row[0], row[0]**2, row[0]**3]]
        xtest_5 += [[row[0], row[0]**2, row[0]**3, row[0]**4, row[0]**5]]
        xtest_7 += [[row[0], row[0]**2, row[0]**3, row[0]**4, row[0]**5, row[0]**6, row[0]**7]]

    xtrain_3 = numpy.array(xtrain_3)
    xtrain_5 = numpy.array(xtrain_5)
    xtrain_7 = numpy.array(xtrain_7)

    xtest_3 = numpy.array(xtest_3)
    xtest_5 = numpy.array(xtest_5)
    xtest_7 = numpy.array(xtest_7)

    print("Preprocessing done. Training...")
    # Inicializamos y entrenamos los perceptrones
    
    classifier_3.fit(xtrain_3, ytr)
    classifier_5.fit(xtrain_5, ytr)
    classifier_7.fit(xtrain_7, ytr)

    print("Done. Testing...")
    # Pasamos los datos de prueba para obtener las predicciones y las comparamos

    results_3_y, results_5_y, results_7_y = [], [], []
    for i in range(len(yts)):
        results_3_y += [classifier_3.net_input(xtest_3[i])]
        results_5_y += [classifier_5.net_input(xtest_5[i])]
        results_7_y += [classifier_7.net_input(xtest_7[i])]

    results_3_train_y, results_5_train_y, results_7_train_y = [], [], []
    for i in range(len(ytr)):
        results_3_train_y += [classifier_3.net_input(xtrain_3[i])]
        results_5_train_y += [classifier_5.net_input(xtrain_5[i])]
        results_7_train_y += [classifier_7.net_input(xtrain_7[i])]

    sorted_x, sorted_y = [], []
    for x, y in sorted(zip(xtest, yts)):
        sorted_x += [x]
        sorted_y += [y]

    sorted_3_y, sorted_5_y, sorted_7_y = [], [], []
    for x, y3, y5, y7 in sorted(zip(xtest, results_3_y, results_5_y, results_7_y)):
        sorted_3_y += [y3]
        sorted_5_y += [y5]
        sorted_7_y += [y7]

    sorted_3_train_y, sorted_5_train_y, sorted_7_train_y = [], [], []
    sorted_x_train, sorted_y_train = [], []
    for x, y, y3, y5, y7 in sorted(zip(xtrain, ytr, results_3_train_y, results_5_train_y, results_7_train_y)):
        sorted_x_train += [x]
        sorted_y_train += [y]
        sorted_3_train_y += [y3]
        sorted_5_train_y += [y5]
        sorted_7_train_y += [y7]

    sorted_x_real, sorted_y_real = [], []
    for x, y in sorted(zip(xreal, yreal)):
        sorted_x_real += [x]
        sorted_y_real += [y]

    #plt.plot(sorted_x, sorted_y, 'ro', label='Muestras')
    #plt.plot(sorted_x, sorted_3_y, 'b-', label='Grado 3')
    #plt.plot(sorted_x, sorted_5_y, 'g-', label='Grado 5')
    #plt.plot(sorted_x, sorted_7_y, 'y-', label='Grado 7')
    #plt.plot(sorted_x_train, sorted_y_train, 'ro', label='Train', markersize=3)
    plt.plot(sorted_x_real, sorted_y_real, 'm-', label='Real')
    #plt.plot(sorted_x_train, sorted_3_train_y, 'b-', label='Grado 3')
    #plt.plot(sorted_x_train, sorted_5_train_y, 'g-', label='Grado 5')
    plt.plot(sorted_x_train, sorted_7_train_y, 'y-', label='Grado 7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f"{max_epochs} epocas, η = {learning_rate}")

    print(f"MSE Train 3: {utils.mse(sorted_y_real, sorted_3_train_y)}")
    print(f"MSE Train 5: {utils.mse(sorted_y_real, sorted_5_train_y)}")
    print(f"MSE Train 7: {utils.mse(sorted_y_real, sorted_7_train_y)}")
    print(f"MSE Test 3: {utils.mse(sorted_y_real, sorted_3_y)}")
    print(f"MSE Test 5: {utils.mse(sorted_y_real, sorted_5_y)}")
    print(f"MSE Test 7: {utils.mse(sorted_y_real, sorted_7_y)}")

    print(f"7 Weights: {classifier_7.w}")
    plt.show()

