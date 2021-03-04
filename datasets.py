from pandas import read_csv
import pandas as pd
import glob
import numpy as np

PATH_DATASETS               = "./datasets/"
FN_PULSAR                   = PATH_DATASETS + "HTRU2/HTRU_2.csv"
FN_IMG_SEG_DATA             = PATH_DATASETS + "Image segmentation/segmentation.data"
FN_IMG_SEG_TEST             = PATH_DATASETS + "Image segmentation/segmentation.test"
FN_STATLOG_SHUTTLE_VEHICLE  = PATH_DATASETS + "Statlog Vehicles/*.dat"
FN_RICE                     = PATH_DATASETS + "rice_gonen_and_jasmine/Rice-Gonen andJasmine.csv"
FN_ELECT_GRID_STABILITY     = PATH_DATASETS + "electrical_grid_stability/Data_for_UCI_named.csv"
FN_MAGIC_GAMMA_TELESCOPE    = PATH_DATASETS + "MAGIC_gamma_telescope/magic04.data"
FN_PEN_DIGITS_TS            = PATH_DATASETS + "pen_digits_recognition/pendigits.tes"
FN_PEN_DIGITS_TR            = PATH_DATASETS + "pen_digits_recognition/pendigits.tra"
FN_LEAF                     = PATH_DATASETS + "leaf/leaf.csv"
FN_ACCENT_RECOGNITION       = PATH_DATASETS + "accent_recognition/accent-mfcc-data-1.csv"
FN_STATLOG_SAT_IMAGE_TR     = PATH_DATASETS + "statlog_sat_image/sat.trn"
FN_STATLOG_SAT_IMAGE_TS     = PATH_DATASETS + "statlog_sat_image/sat.tst"



def load_statlog_sattelite_image(format='np'):
    data1 = read_csv(FN_STATLOG_SAT_IMAGE_TR, sep = " ", header = None)
    data2 = read_csv(FN_STATLOG_SAT_IMAGE_TS, sep = " ", header = None)

    data = pd.concat([data1, data2], axis = 0)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_accent_recognition(format='np'):
    data = pd.read_csv(FN_ACCENT_RECOGNITION, sep=',')

    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_leaf(format='np'):
    data = pd.read_csv(FN_LEAF, sep=',', header=None)

    X = data.iloc[:, 2:] # segunda coluna se refere ao espécime
    y = data.iloc[:, 0]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_pen_digit_recognition(format='np'):
    data1 = read_csv(FN_PEN_DIGITS_TS, sep = ",", header = None)
    data2 = read_csv(FN_PEN_DIGITS_TR, sep = ",", header = None)

    data = pd.concat([data1, data2], axis = 0)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_gamma_telecope(format='np'):

    data = pd.read_csv(FN_MAGIC_GAMMA_TELESCOPE, sep=',', header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_statlog_vehicle(format = 'np'):
    files = glob.glob(FN_STATLOG_SHUTTLE_VEHICLE)

    dfs = [pd.read_csv(fn, sep = "\s+", header = None) for fn in files]

    data = pd.concat(dfs, axis = 0)

    X = data.iloc[:, :18].astype(float)
    y = data.iloc[:, 18]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_image_segmentation(format = 'np'):
    data1 = read_csv(FN_IMG_SEG_DATA, sep = ",", skiprows = 5, header = None)
    data2 = read_csv(FN_IMG_SEG_TEST, sep = ",", skiprows = 5, header = None)

    data = pd.concat([data1, data2], axis = 0)

    X = data.iloc[:, 1:]
    X = X.drop(columns = [3]) #todos as instâncias tem valor igual nesse atributo
    y = data.iloc[:, 0]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_htru_pulsar(format = 'np'):
    data = read_csv(FN_PULSAR, sep = ",", header = None)
    X = data.iloc[:, 0:8]
    y = data.iloc[:, 8]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_rice(format = 'np'):
    data = pd.read_csv(FN_RICE)

    X = data.iloc[:, 1:-1].astype(float)#primeira coluna é id
    y = data.iloc[:, -1]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_electrical_grid_stability(format='np'):
    data = pd.read_csv(FN_ELECT_GRID_STABILITY)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def z_score_normalize(X):
    for i in range(X.shape[1]):
        X.iloc[:, i] = (X.iloc[:, i] -X.iloc[:, i].mean())/X.iloc[:, i].std()
    return X

loader_to_dataset_name_dict = {
    load_accent_recognition.__name__ : 'Accent Recognition',
    load_electrical_grid_stability.__name__ : 'Electrical Grid Stability',
    load_gamma_telecope.__name__ : 'GAMMA Telescope',
    load_htru_pulsar.__name__ : 'HTRU Pulsar',
    load_image_segmentation.__name__ : 'Image Segmentation',
    load_leaf.__name__ : 'Leaf',
    load_pen_digit_recognition.__name__ : 'Pen Digit Recognition',
    load_rice.__name__ : 'Rice',
    load_statlog_sattelite_image.__name__ : 'Statlog Sattelite Image',
    load_statlog_vehicle.__name__ : 'Statlog Vehicle',
}

def loader_to_dataset_name(loader):
    return loader_to_dataset_name_dict[loader]