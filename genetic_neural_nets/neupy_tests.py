#neupy test
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from neupy import environment
from neupy import algorithms, layers
from neupy import plots
from neupy.estimators import mae

import time

def run_neural_net(connection, data):

    #import_modules()

    dataset = data

    data, target = dataset.data, dataset.target

    data_scalar = preprocessing.MinMaxScaler()
    target_scalar = preprocessing.MinMaxScaler()

    data = data_scalar.fit_transform(data)
    target = target_scalar.fit_transform(target.reshape(-1,1))


    environment.reproducible()

    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.85)

    cgnet = algorithms.ConjugateGradient(
        connection,
        search_method = 'golden',
        show_epoch=1,
        verbose=True,
        addons=[algorithms.LinearSearch],

    )

    time_start = time.time()
    cgnet.train(x_train, y_train, x_test, y_test, epochs=5)
    time_end = time.time()

    #plots.error_plot(cgnet)

    y_predict = cgnet.predict(x_test).round(1)
    error = mae(target_scalar.inverse_transform(y_test), \
                  target_scalar.inverse_transform(y_predict))

    print(time_end - time_start)

    #print(target_scalar.inverse_transform(y_test), \
    #              target_scalar.inverse_transform(y_predict))

    print(error)

    return([time_end - time_start, error])
#--------------------------------------------
