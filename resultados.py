#encoding: utf-8
import traceback
from classificadores import Metodo_acuracia, Metodo_similaridade
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
#from sklearn import cross_validation
import sklearn
import util
import matplotlib.pyplot as plt
import seaborn as sns
from csv import writer
from itertools import product
from datasets import load_electrical_grid_stability, load_htru_pulsar, load_image_segmentation, load_rice, load_statlog_vehicle, load_gamma_telecope, load_pen_digit_recognition, load_leaf, load_accent_recognition, load_statlog_sattelite_image, loader_to_dataset_name
import time
import numpy as np
np.random.seed(0)
import pandas as pd

csv_file_name_similaridade = "resultados_similaridade.csv"
csv_file_name_acuracia = "resultados_acuracia.csv"
resultados_folder = "./resultados/"

#csv com testes já realizados para evitar repetição
df_testes_realizados_similaridade = pd.read_csv(csv_file_name_similaridade)
df_testes_realizados_similaridade = df_testes_realizados_similaridade.iloc[:, 0:5]

df_testes_realizados_acuracia = pd.read_csv(csv_file_name_acuracia)
df_testes_realizados_acuracia = df_testes_realizados_acuracia.iloc[:, 0:5]

N_BASE_CLASSIFIERS = 100

def isOnDataframe(df, row):
    return (df == row).all(1).any()

def run_similarity(numObjetosGerados, classifierType, load_dataset_function, qtdeClassificadores):
    startTime = time.time()

    X, y = load_dataset_function()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clfs = getEnsemble(classifierType, N_BASE_CLASSIFIERS)

    msim = Metodo_similaridade(clfs, numObjetosGerados, qtdeClassificadores)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    
    csvRow = [load_dataset_function.__name__, N_BASE_CLASSIFIERS, numObjetosGerados, qtdeClassificadores, classifierType]
    accuracies = []
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        msim_clone = sklearn.base.clone(msim)

        msim_clone.fit(X_train, y_train)
        y_pred = msim_clone.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        accuracies.append(accuracy)
        csvRow.append(accuracy)
    
    csvRow.append(np.mean(accuracies))
    endTime = time.time()
    csvRow.append(endTime - startTime)
    with open(csv_file_name_similaridade, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(csvRow)

    print(csvRow)
    print(f"Time to run: {endTime - startTime}")

def run_acuracia(numPontosProximos, accMin, classifierType, load_dataset_function):
    startTime = time.time()

    X, y = load_dataset_function()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clfs = getEnsemble(classifierType, N_BASE_CLASSIFIERS)

    macc = Metodo_acuracia(clfs, numPontosProximos, accMin)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    
    csvRow = [load_dataset_function.__name__, N_BASE_CLASSIFIERS, numPontosProximos, accMin, classifierType]
    accuracies = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        macc_clone = sklearn.base.clone(macc)

        macc_clone.fit(X_train, y_train)
        y_pred = macc_clone.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        accuracies.append(accuracy)
        csvRow.append(accuracy)
    
    csvRow.append(np.mean(accuracies))
    endTime = time.time()
    csvRow.append(endTime - startTime)
    with open(csv_file_name_acuracia, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(csvRow)

    print(csvRow)
    print(f"Time to run: {endTime - startTime}")

def all_runs_acuracia():
    startTime = time.time()

    numPontosProximos_list = [10]
    classifierTypes = ['tree', 'perceptron', 'naive-bayes', 'knn', 'mix']
    loaders = [load_electrical_grid_stability, load_htru_pulsar, load_image_segmentation, load_rice, load_statlog_vehicle, load_gamma_telecope, load_pen_digit_recognition, load_leaf, load_accent_recognition, load_statlog_sattelite_image]
    accMin_list = [.5, .75, .9]

    all_combinations = product(
        numPontosProximos_list,
        classifierTypes,
        loaders,
        accMin_list
    )
    for numPontosProximos, clfType, loader, accMin in all_combinations:
        row = [loader.__name__, N_BASE_CLASSIFIERS, numPontosProximos, accMin, clfType]
        if (isOnDataframe(df_testes_realizados_acuracia, row)):
            print(f"Already tested: {numPontosProximos}, {N_BASE_CLASSIFIERS}, {clfType}, {loader.__name__}, {accMin}")
            continue
        else:
            print(f"Testing: {numPontosProximos}, {N_BASE_CLASSIFIERS}, {clfType}, {loader.__name__}, {accMin}")
        try:
            run_acuracia(numPontosProximos, accMin, clfType, loader)
        except KeyboardInterrupt:
            print("Keyboard Interruption")
            quit()
        except:
            print(f"Failed to run {loader.__name__} with {clfType}")
            continue

    endTime = time.time()
    print(f"Total time to run: {endTime - startTime}")

def all_runs_similaridade():
    startTime = time.time()

    numObjetosGerados_list = [10]
    classifierTypes = ['tree', 'perceptron', 'naive-bayes', 'knn', 'mix']
    loaders = [load_electrical_grid_stability, load_htru_pulsar, load_image_segmentation, load_rice, load_statlog_vehicle, load_gamma_telecope, load_pen_digit_recognition, load_leaf, load_accent_recognition, load_statlog_sattelite_image]
    qtdeClassificadores_list = [5, 10, 50, 100]

    all_combinations = product(
        numObjetosGerados_list,
        classifierTypes,
        loaders,
        qtdeClassificadores_list
    )
    for numObjGerados, clfType, loader, qtdClfs in all_combinations:
        row = [loader.__name__, N_BASE_CLASSIFIERS, numObjGerados, qtdClfs, clfType]
        if (isOnDataframe(df_testes_realizados_similaridade, row)):
            print(f"Already tested: {numObjGerados}, {N_BASE_CLASSIFIERS}, {clfType}, {loader.__name__}, {qtdClfs}")
            continue
        else:
            print(f"Testing: {numObjGerados}, {N_BASE_CLASSIFIERS}, {clfType}, {loader.__name__}, {qtdClfs}")
        try:
            run_similarity(numObjGerados, clfType, loader, qtdClfs)
        except KeyboardInterrupt:
            print("Keyboard Interruption")
            quit()
        except:
            print(f"Failed to run {loader.__name__} with {clfType}")
            continue

    endTime = time.time()
    print(f"Total time to run: {endTime - startTime}")

def getEnsemble(classifier, n_clfs):
    if classifier == "tree":
        clfs = getDecisionTreeEnsemble(n_clfs)
    if classifier == "perceptron":
        clfs = getPerceptronEnsemble(n_clfs)
    if classifier == "naive-bayes":
        clfs = getNBEnsemble(n_clfs)
    if classifier == "knn":
        clfs = getKnnEnsemble(n_clfs)
    if classifier == "mix":
        clfs = getMixedEnsemble(n_clfs)
    return clfs

def getNBEnsemble(n_clfs):
    return [GaussianNB(var_smoothing = .00000001 * np.sqrt(i + 1)) for i in range(n_clfs)]

def getPerceptronEnsemble(n_clfs):
    return [Perceptron(penalty = 'l2', alpha = 0.0001 * np.sqrt(i + 1) ) for i in range(n_clfs)]

def getDecisionTreeEnsemble(n_clfs):
    return [DecisionTreeClassifier() for i in range(n_clfs)]

def getMixedEnsemble(n_clfs):
    clfs = []
    clfs.extend(getDecisionTreeEnsemble(n_clfs // 3))
    clfs.extend(getPerceptronEnsemble(n_clfs // 3))
    clfs.extend(getKnnEnsemble(n_clfs + (1 - 3) * (n_clfs // 3)))
    return clfs
    
def getKnnEnsemble(n_clfs):
    # 1 to n_clfs//2 clfs, uniform or distance
    clfs = []
    for i in range(2, n_clfs + 2):
        if i%2 == 0:
            w = 'uniform'
        else:
            w = 'distance'
        clf = KNeighborsClassifier(weights=w, n_neighbors=i//2)
        clfs.append(clf)
    return clfs


def analysis_similaridade():
    df_full = pd.read_csv(csv_file_name_similaridade)
    datasets = df_full['dataset'].unique()

    #cria figura com 10 subplots
    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=False, figsize=(10,10))

    #cria uma ax que encompassa as outras menores, para colocar título e labels comuns
    allax = fig.add_subplot(111, frameon=False)
    plt.suptitle('Método da similaridade')
    plt.xlabel('Quantidade de classificadores')
    plt.ylabel('Acurácia média')
    #deixa ticks transparentes (retirar faz com que labels fiquem acima do tick dos subplots)
    plt.tick_params(axis='both', which='both',colors = '#0f0f0f00')

    for dataset, ax in zip(datasets, axs.flatten()):
        plt.sca(ax)

        df = df_full[df_full['dataset'] == dataset]
        ds_name = loader_to_dataset_name(dataset)

        sns.lineplot(data=df, x = 'qtde_classifiers', y='acc_mean', hue='classifier_type')

        #remove legenda e labels de cada subplot
        ax.get_legend().remove()
        ax.set_ylabel('')
        ax.set_xlabel('')
        #Título com nome do dataset
        plt.title(ds_name)
        
    #Copia legenda do último ax iterado e coloca abaixo do gráfico
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5)
    #conserta layout
    plt.tight_layout()
    
    plt.savefig(resultados_folder + 'similaridade.png')

def analysis_acuracia():
    df_full = pd.read_csv(csv_file_name_acuracia)
    datasets = df_full['dataset'].unique()

    #cria figura com 10 subplots
    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=False, figsize=(10,10))

    #cria uma ax que encompassa as outras menores, para colocar título e labels comuns
    allax = fig.add_subplot(111, frameon=False)
    plt.suptitle('Método da acurácia')
    plt.xlabel('Acurácia mínima')
    plt.ylabel('Acurácia média')
    #deixa ticks transparentes (retirar faz com que labels fiquem acima do tick dos subplots)
    plt.tick_params(axis='both', which='both',colors = '#0f0f0f00')

    for dataset, ax in zip(datasets, axs.flatten()):
        plt.sca(ax)

        df = df_full[df_full['dataset'] == dataset]
        ds_name = loader_to_dataset_name(dataset)

        sns.lineplot(data=df, x = 'acc_min', y='acc_mean', hue='classifier_type')

        #remove legenda e labels de cada subplot
        ax.get_legend().remove()
        ax.set_ylabel('')
        ax.set_xlabel('')
        #Título com nome do dataset
        plt.title(ds_name)
        
    #Copia legenda do último ax iterado e coloca abaixo do gráfico
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5)
    #conserta layout
    plt.tight_layout()
    
    plt.savefig(resultados_folder + 'acuracia.png')

if __name__ == "__main__":
    # all_runs_similaridade()
    # all_runs_acuracia()
    analysis_similaridade()
    analysis_acuracia()
