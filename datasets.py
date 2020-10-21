from pandas import read_csv
import pandas as pd
import glob
import numpy as np

PATH_DATASETS               = "./datasets/"
FN_SEEDS                    = PATH_DATASETS + "Seeds/seeds_dataset.txt"
FN_ANURAN                   = PATH_DATASETS + "Anuran Calls (MFCCs)/Frogs_MFCCs.csv"
FN_ECOLI                    = PATH_DATASETS + "ecoli/ecoli.data"
FN_PULSAR                   = PATH_DATASETS + "HTRU2/HTRU_2.csv"
FN_IMG_SEG_DATA             = PATH_DATASETS + "Image segmentation/segmentation.data"
FN_IMG_SEG_TEST             = PATH_DATASETS + "Image segmentation/segmentation.test"
FN_STATLOG_SHUTTLE_DATA     = PATH_DATASETS + "Statlog Shuttle/shuttle.trn"
FN_STATLOG_SHUTTLE_TEST     = PATH_DATASETS + "Statlog Shuttle/shuttle.tst"
FN_STATLOG_SHUTTLE_VEHICLE  = PATH_DATASETS + "Statlog Vehicles/*.dat"
FN_YEAST                    = PATH_DATASETS + "yeast/yeast.data"
FN_AVILA_TS                 = PATH_DATASETS + "avila/avila-ts.txt"
FN_AVILA_TR                 = PATH_DATASETS + "avila/avila-tr.txt"
FN_CROWDSOURCEMAPPING_TEST  = PATH_DATASETS + "crowdsourcedMapping/testing.csv"
FN_CROWDSOURCEMAPPING_TRAIN = PATH_DATASETS + "crowdsourcedMapping/training.csv"
FN_PAGE_BLOCKS              = PATH_DATASETS + "page_blocks/page-blocks.data"
FN_RICE                     = PATH_DATASETS + "rice_gonen_and_jasmine/Rice-Gonen andJasmine.csv"
FN_LETTER_RECOGNITIION      = PATH_DATASETS + "letter_recognition/letter-recognition.data"
FN_ELECT_GRID_STABILITY     = PATH_DATASETS + "electrical_grid_stability/Data_for_UCI_named.csv"
FN_CAR_EVALUATION           = PATH_DATASETS + "car_evaluation/car.data"
FN_MAGIC_GAMMA_TELESCOPE    = PATH_DATASETS + "MAGIC_gamma_telescope/magic04.data"
FN_PEN_DIGITS_TS            = PATH_DATASETS + "pen_letter_recognition/pendigits.tes"
FN_PEN_DIGITS_TR            = PATH_DATASETS + "pen_letter_recognition/pendigits.tra"
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

def load_pen_letter_recognition(format='np'):
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

def load_yeast(format = 'np'):
    data = pd.read_csv(FN_YEAST, sep = "\s+", header = None)

    X = data.iloc[:, 1:9] # primeira coluna é id
    y = data.iloc[:, 9]


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

def load_statlog_shuttle(format = 'np'):
    data1 = read_csv(FN_STATLOG_SHUTTLE_DATA, sep = " ", header = None)
    data2 = read_csv(FN_STATLOG_SHUTTLE_TEST, sep = " ", header = None)

    data = pd.concat([data1, data2], axis = 0)

    X = data.iloc[:, :9]
    y = data.iloc[:, 9]

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

def load_ecoli(format = 'np'):
    data = read_csv(FN_ECOLI, sep = "\s+", header = None)
    X = data.iloc[:, 1:8]
    y = data.iloc[:, 8]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_seeds(format = 'np'):
    data = read_csv(FN_SEEDS, sep = "\s+", header = None)
    X = data.iloc[:, 0:7]
    y = data.iloc[:, 7]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X


def _load_anuran(classification = "species", format = 'np'):
    data = read_csv(FN_ANURAN, sep = ",")

    X = data.iloc[:, 0:22]
    if classification == "species":
        y = data.iloc[:, 24]
    if classification == "genus":
        y = data.iloc[:, 23]
    if classification == "family":
        y = data.iloc[:, 22]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X
        

def load_anuran_species(format = 'np'):
    return _load_anuran("species", format)

def load_anuran_genus(format = 'np'):
    return _load_anuran("genus", format)

def load_anuran_family(format = 'np'):
    return _load_anuran("family", format)

def load_avila(format = 'np'):

    data1 = pd.read_csv(FN_AVILA_TS, sep=',', header=None)
    data2 = pd.read_csv(FN_AVILA_TR, sep=',', header=None)

    data = pd.concat([data1, data2], axis = 0)

    X = data.iloc[:, 0:10]
    y = data.iloc[:, 10]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X

def load_crowdsource_mapping(format = 'np'):

    data1 = pd.read_csv(FN_CROWDSOURCEMAPPING_TEST, sep=',')
    data2 = pd.read_csv(FN_CROWDSOURCEMAPPING_TRAIN, sep=',')

    data = pd.concat([data1, data2], axis = 0)

    X = data.iloc[:, 1:29]
    y = data.iloc[:, 0]

    if format == 'np':
        return X.to_numpy(), y.to_numpy()
    if format == 'pd':
        X['class'] = y
        return X


def load_page_blocks(format = 'np'):
    data = pd.read_csv(FN_PAGE_BLOCKS, sep="\s+", header=None)

    X = data.iloc[:, :-1].astype(float)
    y = data.iloc[:, -1]

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

def load_letter_recognition(format = 'np'):
    data = pd.read_csv(FN_LETTER_RECOGNITIION, header=None)

    X = data.iloc[:, 1:].astype(float)#primeira coluna é id
    y = data.iloc[:, 0]

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

def _test_dataframe():

    df = pd.DataFrame(np.array([
        [1,2,3,0,5],
        [0,0,7,8,9],
        [0,12,13,14,15]
    ]))
    return df

def main():
    df = load_page_blocks(format='pd')
    print(df.head())
    print(df.dtypes)
    print(df['class'].value_counts())

if __name__ == '__main__':
    main()