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

from scipy.stats import friedmanchisquare, rankdata
from scikit_posthocs import posthoc_nemenyi_friedman

csv_file_name_similaridade = "resultados_similaridade.csv"
csv_file_name_acuracia = "resultados_acuracia.csv"
resultados_folder = "./resultados/"

#csv com testes já realizados para evitar repetição
df_testes_realizados_similaridade = pd.read_csv(csv_file_name_similaridade)
df_testes_realizados_similaridade = df_testes_realizados_similaridade.iloc[:, 0:5]

df_testes_realizados_acuracia = pd.read_csv(csv_file_name_acuracia)
df_testes_realizados_acuracia = df_testes_realizados_acuracia.iloc[:, 0:5]

N_BASE_CLASSIFIERS = 20

def isOnDataframe(df, row):
    return (df == row).all(1).any()

def run_similarity(numObjetosGerados, classifierType, load_dataset_function, qtdeClassificadores, save_to_csv=True):
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
    if save_to_csv:
        with open(csv_file_name_similaridade, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(csvRow)

    print(csvRow)
    print(f"Time to run: {endTime - startTime}")

def run_acuracia(numPontosProximos, accMin, classifierType, load_dataset_function, save_to_csv=True):
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

    if save_to_csv:
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
    accMin_list = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 ,
       0.55, 0.6 ,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

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
    qtdeClassificadoresPct_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 ,
       0.55, 0.6 ,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
    all_combinations = product(
        numObjetosGerados_list,
        classifierTypes,
        loaders,
        qtdeClassificadoresPct_list
    )
    for numObjGerados, clfType, loader, qtdClfsPct in all_combinations:
        qtdClfs = int(qtdClfsPct * N_BASE_CLASSIFIERS)
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
            traceback.print_exc()
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
    return [GaussianNB(var_smoothing = np.random.uniform(1e-10, 1e-8)) for i in range(n_clfs)]

def getPerceptronEnsemble(n_clfs):
    return [Perceptron(penalty = 'l2', alpha = np.random.uniform(0., .1) ) for i in range(n_clfs)]

def getDecisionTreeEnsemble(n_clfs):
    return [DecisionTreeClassifier(splitter='random') for i in range(n_clfs)]

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
    classifiers = df_full['classifier_type'].unique()

    df_full = df_full.drop(['acc_mean', 'time_to_run'], axis = 1)

    df_full = df_full.melt(id_vars=['dataset', 'n_base_classifiers', 'n_objetos_gerados', 'qtde_classifiers', 'classifier_type'], var_name='kfold', value_name='acc')

    for classifier in classifiers:
        #cria figura com 10 subplots
        fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=False, figsize=(10,10))

        #cria uma ax que encompassa as outras menores, para colocar título e labels comuns
        allax = fig.add_subplot(111, frameon=False)
        plt.ylabel('')
        plt.xlabel('Quantidade de classificadores')
        #deixa ticks transparentes (retirar faz com que labels fiquem acima do tick dos subplots)
        plt.tick_params(axis='both', which='both',colors = '#0f0f0f00')

        df_per_clf = df_full[df_full['classifier_type'] == classifier]
        plt.suptitle(f'Método da similaridade - {classifier}')

        for dataset, ax in zip(datasets, axs.flatten()):
            plt.sca(ax)

            df = df_per_clf[df_per_clf['dataset'] == dataset]
            ds_name = loader_to_dataset_name(dataset)

            sns.boxplot(data=df, x='qtde_classifiers', y='acc', palette='colorblind')

            #remove legenda e labels de cada subplot
            # ax.get_legend().remove()
            ax.set_ylabel('')
            ax.set_xlabel('')
            #Título com nome do dataset
            plt.title(ds_name)
            
        #Copia legenda do último ax iterado e coloca abaixo do gráfico
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5)
        #conserta layout
        plt.tight_layout()
        
        plt.savefig(resultados_folder + f'similaridade-{classifier}.png')

def analysis_acuracia():
    df_full = pd.read_csv(csv_file_name_acuracia)
    datasets = df_full['dataset'].unique()
    classifiers = df_full['classifier_type'].unique()

    df_full = df_full.drop(['acc_mean', 'time_to_run'], axis = 1)

    df_full = df_full.melt(id_vars=['dataset','n_base_classifiers','n_pontos_proximos','acc_min','classifier_type'], var_name='kfold', value_name='acc')

    for classifier in classifiers:    
        #cria figura com 10 subplots
        fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=False, figsize=(10,10))

        #cria uma ax que encompassa as outras menores, para colocar título e labels comuns
        allax = fig.add_subplot(111, frameon=False)
        plt.xlabel('Acurácia mínima')
        plt.ylabel('')
        #deixa ticks transparentes (retirar faz com que labels fiquem acima do tick dos subplots)
        plt.tick_params(axis='both', which='both',colors = '#0f0f0f00')
        
        df_per_clf = df_full[df_full['classifier_type'] == classifier]
        plt.suptitle(f'Método da acurácia - {classifier}')

        for dataset, ax in zip(datasets, axs.flatten()):
            plt.sca(ax)

            df = df_per_clf[df_per_clf['dataset'] == dataset]
            ds_name = loader_to_dataset_name(dataset)

            sns.boxplot(data=df,  y='acc', x='acc_min', palette='colorblind')

            #remove legenda e labels de cada subplot
            # ax.get_legend().remove()
            ax.set_ylabel('')
            ax.set_xlabel('')
            #Título com nome do dataset
            plt.title(ds_name)
            
        #Copia legenda do último ax iterado e coloca abaixo do gráfico
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5)
        #conserta layout
        plt.tight_layout()

        plt.savefig(resultados_folder + f'acuracia-{classifier}.png')

def friedman_test():
    df_acc = pd.read_csv(csv_file_name_acuracia)
    df_sim = pd.read_csv(csv_file_name_similaridade)

    #renomeia dataframes para que possam ser concatenados
    df_acc = df_acc[['acc_min', 'classifier_type', 'dataset', 'acc_mean']].rename(columns={'acc_min': 'parametro', 'acc_mean': 'measurement'})
    df_sim = df_sim[['qtde_classifiers', 'classifier_type', 'dataset', 'acc_mean']].rename(columns={'qtde_classifiers': 'parametro', 'acc_mean': 'measurement'})
    df_acc['metodo'] = 'acuracia'
    df_sim['metodo'] = 'similaridade'

    #concatena os dataframes
    df_all = pd.concat([df_acc, df_sim])

    

    #transforma metodo, classifier_type e parametro em uma coluna só
    df_all = df_all.astype({'parametro': str})
    df_all = df_all.set_index(keys=['metodo', 'classifier_type', 'parametro'])
    df_all.index = df_all.index.map('-'.join)
    df_all.reset_index(inplace=True)


    #cada algoritmo diferente vira uma coluna, cada linha corresponde a um dataset, e os valores são a acurácia
    df_all_pivoted = df_all.pivot(index='dataset', columns='index', values='measurement')

    #teste de friedman    
    statistic, pvalue = friedmanchisquare(*df_all_pivoted.values.tolist())
    print(f'p-value={pvalue}')

    #teste de nemenyi
    nemenyi = posthoc_nemenyi_friedman(df_all, melted=True, group_col='index', block_col='dataset', y_col='measurement')

    nemenyi.to_csv('resultado_nemenyi.csv')

def rank_acuracia():
    df = pd.read_csv(csv_file_name_acuracia)
    df = df.rename(columns={'acc_min': 'parametro'})
    rank(df, 'acuracia')

def rank_similaridade():
    df = pd.read_csv(csv_file_name_similaridade)
    df = df.rename(columns={'qtde_classifiers': 'parametro'})
    rank(df, 'similaridade')

def rank(df_original, metodo):
    df_original = df_original[['parametro', 'classifier_type', 'dataset', 'acc_mean']]

    #transforma metodo, classifier_type e parametro em uma coluna só
    df = df_original.astype({'parametro': str})
    df = df.set_index(keys=['classifier_type', 'parametro'])
    df.index = df.index.map('-'.join)
    df.reset_index(inplace=True)

    df_pivoted = df.pivot(index='dataset', columns='index', values='acc_mean')

    #ranks (tipo de classificador, quantidade de classificadores)
    ranks = pd.DataFrame(rankdata(df_pivoted, axis=1), index=df_pivoted.index, columns=df_pivoted.columns)
    ranks = ranks.mean(axis=0).sort_values()
    ranks.to_csv(f'ranks_{metodo}.csv')

    #ranks (quantidade de classificadores)
    df = df_original.groupby(by = ['parametro', 'dataset']).mean()
    df.reset_index(inplace=True)
    df_pivoted = df.pivot(index='dataset', columns='parametro', values='acc_mean')
    ranks = pd.DataFrame(rankdata(df_pivoted, axis=1), index=df_pivoted.index, columns=df_pivoted.columns)
    ranks = ranks.mean(axis=0).sort_values()
    ranks.to_csv(f'ranks_{metodo}_por_parametro.csv')


def friedman_test_similaridade():
    df = pd.read_csv(csv_file_name_similaridade)
    df = df[['qtde_classifiers', 'classifier_type', 'dataset', 'acc_mean']]

    #transforma metodo, classifier_type e parametro em uma coluna só
    df = df.astype({'qtde_classifiers': str})
    df = df.set_index(keys=['classifier_type', 'qtde_classifiers'])
    df.index = df.index.map('-'.join)
    df.reset_index(inplace=True)


    #teste de friedman    
    df_pivoted = df.pivot(index='dataset', columns='index', values='acc_mean')
    statistic, pvalue = friedmanchisquare(*df_pivoted.values.tolist())
    print(f'p-value={pvalue}')

    #ranks
    ranks = pd.DataFrame(rankdata(df_pivoted, axis=1), index=df_pivoted.index, columns=df_pivoted.columns)
    ranks = ranks.mean(axis=0).sort_values()
    ranks.to_csv('ranks_similaridade')

    

    #teste de nemenyi
    nemenyi = posthoc_nemenyi_friedman(df, melted=True, group_col='index', block_col='dataset', y_col='acc_mean')

    nemenyi.to_csv('resultado_nemenyi_similaridade.csv')

def friedman_test_acuracia():
    df = pd.read_csv(csv_file_name_acuracia)
    df = df[['acc_min', 'classifier_type', 'dataset', 'acc_mean']]

    #transforma metodo, classifier_type e parametro em uma coluna só
    df = df.astype({'acc_min': str})
    df = df.set_index(keys=['classifier_type', 'acc_min'])
    df.index = df.index.map('-'.join)
    df.reset_index(inplace=True)

    #teste de friedman    
    df_pivoted = df.pivot(index='dataset', columns='index', values='acc_mean')
    statistic, pvalue = friedmanchisquare(*df_pivoted.values.tolist())
    print(f'p-value={pvalue}')

    #teste de nemenyi
    nemenyi = posthoc_nemenyi_friedman(df, melted=True, group_col='index', block_col='dataset', y_col='acc_mean')

    nemenyi.to_csv('resultado_nemenyi_acuracia.csv')

if __name__ == "__main__":
    # all_runs_acuracia()
    # all_runs_similaridade()
    # analysis_acuracia()
    # analysis_similaridade()
    # friedman_test()
    # friedman_test_similaridade()
    # friedman_test_acuracia()
    rank_similaridade()
    rank_acuracia()