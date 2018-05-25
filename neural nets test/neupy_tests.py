#neupy test
import os

def import_modules():

    try:
        from sklearn import datasets
    except:
        os.sys('pip install sklearn')
        from sklearn import datasets
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    try:
        from neupy import environment
    except:
        os.sys('pip install neupy')
        from neupy import environment

    from neupy import algorithms, layers
    from neupy import plots
    from neupy.estimators import rmsle

def run_neural_net():

    import_modules()

    dataset = datasets.load_boston()
    data, target = dataset.data, dataset.target

    data_scalar = preprocessing.MinMaxScaler()
    target_scalar = preprocessing.MinMaxScaler()

    data = data_scalar.fit_transform(data)
    target = target_scalar.fit_transform(target.reshape(-1,1))


    environment.reproducible()

    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.85)

    cgnet = algorithms.ConjugateGradient(
        connection = [
            layers.Input(13),
            layers.Sigmoid(75),
            layers.Sigmoid(25),
            layers.Sigmoid(1),
        ],
        search_method = 'golden',
        show_epoch=1,
        verbose=True,
        addons=[algorithms.LinearSearch],

    )

    cgnet.train(x_train, y_train, x_test, y_test, epochs=30)

    plots.error_plot(cgnet)

    y_predict = cgnet.predict(x_test).round(1)
    error = rmsle(target_scalar.invers_transform(y_test), \
                  target_scalar.invers_transform(y_predict))

    return(error)
