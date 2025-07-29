import data_higgs as dh
from time import time


import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

path = "pdf/"
dataset = dh.load_data("data","data1")
data_trn=dataset['train']
data_val=dataset['valid'] 

print("HIGGS dataset loaded successfully.")
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# X = HIGGS.trn.iloc[:, 1:]  # parameters (columns 1 to end)
# y = HIGGS.trn.iloc[:, 0]   # labels (first column)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

size = int(5* 1e5)
X_train = data_trn.iloc[:size, 1:]  # parameters (columns 1 to end)
y_train = data_trn.iloc[:size, 0]   # labels (first column)
X_test = data_val.iloc[:len(data_val)//2, 1:]  # parameters (columns 1 to end)
y_test = data_val.iloc[:len(data_val)//2, 0]   # labels (first column)

# Create and train model

# print("Training... dataset size:", len(X_train))

acc = []
x_os = []
times = []
max_iters = [10, 100, 200, 300]
fig, ax1 = plt.subplots()

for i in range(len(max_iters)):
    time_start = time()
    model = HistGradientBoostingClassifier(
           learning_rate=0.4,
        max_iter=1000,
        max_leaf_nodes=63,
        min_samples_leaf=5,
        # early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
        )
    model.fit(X_train, y_train)
    tme = time() - time_start
    print("Model {} trained successfully.".format(i))
    print("Evaluating model...")
    print("Training time:", tme)
    # Predict and evaluate
    y_pred = model.predict(X_test)
    tmp = accuracy_score(y_test, y_pred)
    print("Accuracy:", tmp)
    acc.append(tmp)
    times.append(tme)
    x_os.append(max_iters[i])
    exit()
ax1.scatter(x_os, acc, label='Accuracy', color='blue',alpha=0.7)
ax1.set_ylabel("Accuracy",color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.scatter(x_os, times, color='red', alpha=0.7)
ax2.set_ylabel('Training Time (s)', color='red')
ax2.tick_params(axis='y', labelcolor='red')




# plt.grid()
# plt.scatter(x_os, acc, label='Accuracy',color='black', zorder=2)
# plt.xlabel('Model learning rate')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Model Accuracy vs Learning Rate, training size: $5 \\times 10^5$')

plt.savefig(path+'itter_time.pdf', dpi=200)