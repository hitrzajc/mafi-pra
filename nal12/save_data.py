import data_higgs as dh
def save_higgs_data(dataset="data1", datadir="data", test_count=1*int(1e5), train_count=8*int(1e5)):
    """
    Save the HIGGS dataset to a specified directory.
    """
    HIGGS = dh.HIGGS(test_count=test_count, train_count=train_count)
    dh.save_data(HIGGS.trn, HIGGS.val, HIGGS.feature_names, dataset=dataset, datadir=datadir)


save_higgs_data()