import os,sys,pickle,gzip,math
import pandas as pd
import numpy as np
from os.path import join

# [(11000000 rows (5,170,877 rows bkg) x 28 columns] 
HIGGS_COL_NAMES=np.array(['lepton pT','lepton eta','lepton phi','missing energy','missing energy phi',
'jet_1 pt','jet_1 eta','jet_1 phi','jet_1 b-tag','jet_2 pt','jet_2 eta','jet_2 phi','jet_2 b-tag',
'jet_3 pt','jet_3 eta','jet_3 phi','jet_3 b-tag','jet_4 pt','jet_4 eta','jet_4 phi','jet_4 b-tag',
'm_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb'])
print(len(HIGGS_COL_NAMES), "HIGGS columns")
#continuous and non-flat variables
CONT_VARS=[0,1,3,5,6,9,10,13,14,17,18,21,22,23,24,25,26,27]

#BPK local data management in pickled files
datasets_root = 'data/'
INPUT_HIGGS = '/Users/borut/CERNBox/ML-sampling/SIM/EXAMPLE_code/maf/data/'
DEF_DATADIR=os.path.join(os.getcwd(),datasets_root)
INDATASET='higgs-parsed'

def save_data(trn, val, fnames,dataset, datadir):
    path = os.path.join(datadir, dataset+'/')
    # Create the model directory if not present.
    if not os.path.exists(path):
        os.makedirs(path)        
    outfile = os.path.join(path, dataset+'.h5')
    print("Storing variables to hdf5:\n {}".format(fnames))
    store = pd.HDFStore(outfile)
    store['train']=trn
    store['valid']=val
    store['feature_names']=fnames
    store.close()

    return None


def download_and_make_data(dataset=None,datadir=None):
    """ Download  preprocessed datasets and convert to pickle files to use
    for training.Saves pickle files in specifed local directory.
    """
    #BPK modified
    if datadir is None:
        datadir = DEF_DATADIR
    #higgsbkg data file
    print('Making higgs pickle...')
    if dataset is None:
        dataset = INDATASET 
    data = HIGGS()
    save_data(data.trn,data.val,data.feature_names,dataset, datadir)
    return None

def load_data(datadir=None,dataset=None):
    """ Retrieve the already processed data from the specified location.
    """
    #BPK modified
    if datadir is None:
        datadir = DEF_DATADIR
    #HIGGS data file
    if dataset is None:
        dataset = INDATASET 
    path = os.path.join(datadir, dataset+'/')
    # Create the model directory if not present.
    infile = os.path.join(path, dataset+'.h5')
    print('Loading {}...'.format(infile))
    dataset = pd.HDFStore(infile,'r')

    #data_trn=dataset['train']
    #data_val=dataset['valid'] 
    #n_dims=data_trn.shape[0]

    print('Loaded.')


    return dataset


class HIGGS:
    """
    The HIGGS data set. Pick the BACKGROUND out of the sample!
    http://archive.ics.uci.edu/ml/datasets/HIGGS
    """
            

    def __init__(self,train_start=None,train_count=None,test_start=None,test_count=None):
        """
        arguments:
        train_start: Start index of train examples within the data. default=0
        train_count: Number of train examples within the data. default=400000
        test_start: Start index of test examples within the data. default=4000000
        test_count", help="Number of test examples within the data. default=100000
        
        Validation samples are 10% of the training sample.

        """
        #download locations
        self.URL_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280"
        self.INPUT_FILE = "HIGGS.csv.gz"
        self.feature_names = None #fill later

        self.train_start=0      if train_start is None else train_start 
        self.train_count=400000 if train_count is None else train_count
        self.test_start=4000000 if test_start is None else test_start 
        self.test_count=100000  if test_count is None else test_count

        path = "./"

        # get the trainind and validation data.
        self.trn, self.val  = self.process_data(path)




    def process_data(self,path):

        input_url = os.path.join(self.URL_ROOT, self.INPUT_FILE)
        temp_filename=os.path.join(path, self.INPUT_FILE)
        if not os.path.exists(path):
            os.makedirs(path)        

        if os.path.isfile(temp_filename):
            print("data_dir already has the downloaded data file: {}".format(temp_filename))
        else:
            print("need to wget!!")
            sys.exit(1)
            wget.download(input_url,temp_filename)

        print("Data input : {}".format(temp_filename))
        # Reading and parsing 11 million csv lines takes 2~3 minutes.
        print("Data processing... taking multiple minutes...")
        feature_names = [fname.replace(" ","-") for fname in np.concatenate((['hlabel'],HIGGS_COL_NAMES))]
        with gzip.open(temp_filename, "rb") as csv_file:
                data = pd.read_csv(
                csv_file,
                dtype=np.float32,
                header=None,
                index_col=False,
                names=feature_names, # label + 28 features.
                #nrows=20, # debug
            )

        # Gets rid of any background noise examples i.e. class label 0.
        data.head()
        print ("Read variables: {}".format(data.columns))
        print ("Dataset shape : {}".format(data.shape))

        #data_bkg=data[data_raw["hlabel"] == 0.0].drop(['hlabel'], axis=1) recipe for filtering sig/bkg!

        sel_vars=[ i+1 for i in CONT_VARS ] # have to shift for hlabel
        sel_names=[fname.replace(" ","-") for fname in HIGGS_COL_NAMES[CONT_VARS]]
        print ("Selected variables: {}".format(sel_names))

        self.feature_names = pd.Series(np.concatenate((['hlabel'],sel_names))) # add the label!
        z_sel_vars=[0]+sel_vars # add the label!
        data_train=data.iloc[:,z_sel_vars] 
        data_val =data.iloc[self.test_start:self.test_start+self.test_count,z_sel_vars] 

        #eventual to_array?   data_train, data_val = data_train.values, data_val.values or .to_numpy()

        return data_train,data_val


    # #BPK standard scaling = normalized
    # def process_data_no_discrete_no_flat_normalised(self,path):

    #     data_train, data_val = self.process_data(path)
    #     mu = data_train.mean()
    #     s = data_train.std()
    #     data_train = (data_train - mu)/s
    #     data_val = (data_val - mu)/s

    #     return data_train, data_val

    # #BPK scaling to [-1,1] range
    # def process_data_no_discrete_no_flat_symscaled(self,path):

    #     data_train, data_val = self.process_data(path)
    #     dmax = data_train.max()
    #     dmin = data_train.min()
    #     data_train = -1. + 2.*(data_train - dmin)/(dmax-dmin)
    #     data_val  = -1. + 2.*(data_val - dmin)/(dmax-dmin)

    #     return data_train, data_val


    # #BPK scaling to [0,1] range
    # def process_data_no_discrete_no_flat_scaled(self,path):

    #     data_train, data_val = self.process_data(path)
    #     dmax = data_train.max()
    #     dmin = data_train.min()
    #     data_train = (data_train - dmin)/(dmax-dmin)
    #     data_val = (data_val - dmin)/(dmax-dmin)

    #     return data_train, data_val


