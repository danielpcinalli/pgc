"""
>>> X, y = load_iris(return_X_y = True)
>>> type(X)
<class 'numpy.ndarray'>
>>> type(y)
<class 'numpy.ndarray'>
>>> type(X[0])
<class 'numpy.ndarray'>
>>> type(X[0][0])
<class 'numpy.float64'>
>>> type(y[0])
<class 'numpy.int64'>

"""


from pandas import read_csv

PATH_DATASETS         = "./datasets/"
FILENAME_SEEDS        = PATH_DATASETS + "seeds_dataset.txt"
FILENAME_ANURAN       = PATH_DATASETS + "Anuran Calls (MFCCs)/Frogs_MFCCs.csv"
FILENAME_ECOLI        = PATH_DATASETS + "ecoli/ecoli.data"
FILENAME_PULSAR       = PATH_DATASETS + "HTRU2/HTRU_2.csv"
FILENAME_IMG_SEG_DATA = PATH_DATASETS + "segmentation.data"
FILENAME_IMG_SEG_TEST = PATH_DATASETS + "segmentation.test"

def load_img_seg():
    data = read_csv(FILENAME_IMG_SEG_DATA, sep = ",", skiprows = 5)
    X = data.iloc[:, 1:].to_numpy()
    y = data.iloc[:, 0].to_numpy()
    return X, y
    
def load_pulsar():
    data = read_csv(FILENAME_PULSAR, sep = ",")
    X = data.iloc[:, 0:8].to_numpy()
    y = data.iloc[:, 8].to_numpy()
    return X, y    
    
def load_ecoli():
    data = read_csv(FILENAME_ECOLI, sep = "\s+")
    X = data.iloc[:, 1:8].to_numpy()
    y = data.iloc[:, 8].to_numpy()
    return X, y
    
def load_seeds():
    data = read_csv(FILENAME_SEEDS, sep = "\s+")
    X = data.iloc[:, 0:7].to_numpy()
    y = data.iloc[:, 7].to_numpy()
    return X, y
    
def load_anuran(classification = "species"):
    data = read_csv(FILENAME_ANURAN, sep = ",")
    
    X = data.iloc[:, 0:22].to_numpy()
    if classification == "species":
        y = data.iloc[:, 24].to_numpy()
    if classification == "genus":
        y = data.iloc[:, 23].to_numpy()
    if classification == "family":
        y = data.iloc[:, 22].to_numpy()
    
    return X, y
    

