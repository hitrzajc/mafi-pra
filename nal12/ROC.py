import data_higgs as dh
from time import time

def split_xy(rawdata):
    #split features and labels from data 
    #prepare the data => normalizations !   

    # split 
    data_y=rawdata['hlabel'] # labels only: 0.=bkg, 1.=sig
    data_x=rawdata.drop(['hlabel'], axis=1) # features only
    
    #now prepare the data
    mu = data_x.mean()
    s = data_x.std()
    dmax = data_x.max()
    dmin = data_x.min()

    # normal/standard rescaling 
    data_x = (data_x - mu)/s

    # scaling to [-1,1] range
    # data_x = -1. + 2.*(data_x - dmin)/(dmax-dmin)

    # scaling to [0,1] range
    # data_x = (data_x - dmin)/(dmax-dmin)


    return data_x,data_y


import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

path = "pdf/"
dataset = dh.load_data("data","data1")
data_trn=dataset['train']
data_val=dataset['valid'] 
data_fnames=dataset['feature_names'].to_numpy()[1:]

print("HIGGS dataset loaded successfully.")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier, Pool

# X = HIGGS.trn.iloc[:, 1:]  # parameters (columns 1 to end)
# y = HIGGS.trn.iloc[:, 0]   # labels (first column)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

size = int(4* 1e5)
print(size)
# X_train = data_trn.iloc[:size, 1:]  # parameters (columns 1 to end)
# y_train = data_trn.iloc[:size, 0]   # labels (first column)
# X_test = data_val.iloc[:len(data_val)//2, 1:]  # parameters (columns 1 to end)
# y_test = data_val.iloc[:len(data_val)//2, 0]   # labels (first column)

X_train, y_train = split_xy(data_trn.iloc[:size])
X_test, y_test = split_xy(data_val.iloc[:len(data_val)])

# Create and train model
time_start = time()

pool_train = Pool(data=X_train.to_numpy(),label=y_train.to_numpy(),feature_names=data_fnames.tolist())
pool_test = Pool(data=X_test.to_numpy(),label=y_test.to_numpy(),feature_names=data_fnames.tolist())
# CatBoost parameters 
eval_metric = 'AUC' # see https://catboost.ai/docs/concepts/loss-functions-classification.html
task_type = 'CPU'  # if GPU else 'CPU'
max_number_of_trees = 200


print("Training... dataset size:", len(X_train))

acc = []
x_os = []

model = CatBoostClassifier(
    verbose=False,
    task_type=task_type,
    loss_function="Logloss", # see values same as in eval_metric, e.g. CrossEntropy, Logloss is default
    iterations=max_number_of_trees,
    eval_metric=eval_metric,
    learning_rate=0.5,
    #max_depth=6
    use_best_model=True,
    random_seed=42,
    )  
model.fit(pool_train, eval_set=pool_test,)
print("Model {} trained successfully.".format(1))
print("Evaluating model...")
print("Training time:", time() - time_start)
# Predict and evaluate
y_pred = model.predict(X_test)
tmp = accuracy_score(y_test, y_pred)
print("Accuracy:", tmp)
acc.append(tmp)

import plotting
from sklearn.metrics import roc_auc_score

y_score=model.predict_proba(X_test.to_numpy())[:,1]
print("score shape {}",y_score.shape)


from sklearn.metrics import RocCurveDisplay
disp = RocCurveDisplay.from_predictions(y_test, y_score)
fpr, tpr = disp.fpr, disp.tpr
disp.line_.set_color("black")

legend = plt.legend()
for text in legend.get_texts():
    text.set_color("black")

plt.plot([0, 1], [0, 1], linestyle="--", color="red")
plt.fill_between(fpr, tpr, fpr, where=(tpr > fpr),
               facecolor="gray", alpha=0.2, hatch="\\\\", edgecolor="white",hatch_linewidth=2)

plt.title("ROC Curve dataset size: $4\\times 10^5$")
plt.savefig(path+"roc_boost.pdf", dpi=200)
# plotting.plot_roc(y_test,y_score)
# plotting.plot_score(y_test,y_score)

auc=roc_auc_score(y_test,y_score)
print("AUC score: {}".format(auc))
# plt.savefig(path+'model_accuracy_vs_learning_rate.pdf', dpi=200)