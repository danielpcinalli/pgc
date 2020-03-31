from pandas import read_csv
import pandas as pd
import glob

PATH_DATASETS              = "./datasets/"
FN_SEEDS                   = PATH_DATASETS + "Seeds/seeds_dataset.txt"
FN_ANURAN                  = PATH_DATASETS + "Anuran Calls (MFCCs)/Frogs_MFCCs.csv"
FN_ECOLI                   = PATH_DATASETS + "ecoli/ecoli.data"
FN_PULSAR                  = PATH_DATASETS + "HTRU2/HTRU_2.csv"
FN_IMG_SEG_DATA            = PATH_DATASETS + "Image segmentation/segmentation.data"
FN_IMG_SEG_TEST            = PATH_DATASETS + "Image segmentation/segmentation.test"
FN_STATLOG_SHUTTLE_DATA    = PATH_DATASETS + "Statlog Shuttle/shuttle.trn"
FN_STATLOG_SHUTTLE_TEST    = PATH_DATASETS + "Statlog Shuttle/shuttle.tst"
FN_STATLOG_SHUTTLE_VEHICLE = PATH_DATASETS + "Statlog Vehicles/*.dat"
FN_YEAST                   = PATH_DATASETS + "yeast/yeast.data"

def load_yeast():
    data = pd.read_csv(FN_YEAST, sep = "\s+", header = None)
    
    X = data.iloc[:, 1:9].to_numpy()
    y = data.iloc[:, 9].to_numpy()
    return X, y

def load_statlog_vehicle():
    files = glob.glob(FN_STATLOG_SHUTTLE_VEHICLE)
    
    dfs = [pd.read_csv(fn, sep = "\s+", header = None) for fn in files]
    
    data = pd.concat(dfs, axis = 0)
    
    X = data.iloc[:, :18].to_numpy()
    y = data.iloc[:, 18].to_numpy()
    return X, y

def load_statlog_shuttle():
    data1 = read_csv(FN_STATLOG_SHUTTLE_DATA, sep = " ", header = None)
    data2 = read_csv(FN_STATLOG_SHUTTLE_TEST, sep = " ", header = None)

    data = pd.concat([data1, data2], axis = 0)
    
    X = data.iloc[:, :9].to_numpy()
    y = data.iloc[:, 9].to_numpy()
    return X, y    

def load_img_seg():
    data1 = read_csv(FN_IMG_SEG_DATA, sep = ",", skiprows = 5, header = None)
    data2 = read_csv(FN_IMG_SEG_TEST, sep = ",", skiprows = 5, header = None)
    
    data = pd.concat([data1, data2], axis = 0)
    
    X = data.iloc[:, 1:].to_numpy()
    y = data.iloc[:, 0].to_numpy()
    return X, y
    
def load_pulsar():
    data = read_csv(FN_PULSAR, sep = ",", header = None)
    X = data.iloc[:, 0:8].to_numpy()
    y = data.iloc[:, 8].to_numpy()
    return X, y    
    
def load_ecoli():
    data = read_csv(FN_ECOLI, sep = "\s+", header = None)
    X = data.iloc[:, 1:8].to_numpy()
    y = data.iloc[:, 8].to_numpy()
    return X, y
    
def load_seeds():
    data = read_csv(FN_SEEDS, sep = "\s+", header = None)
    X = data.iloc[:, 0:7].to_numpy()
    y = data.iloc[:, 7].to_numpy()
    return X, y

def load_anuran(classification = "species"):
    data = read_csv(FN_ANURAN, sep = ",")
    
    X = data.iloc[:, 0:22].to_numpy()
    if classification == "species":
        y = data.iloc[:, 24].to_numpy()
    if classification == "genus":
        y = data.iloc[:, 23].to_numpy()
    if classification == "family":
        y = data.iloc[:, 22].to_numpy()
    
    return X, y

def load_anuran_species():
    return load_anuran("species")
    
def load_anuran_genus():
    return load_anuran("genus")

def load_anuran_family():
    return load_anuran("family")