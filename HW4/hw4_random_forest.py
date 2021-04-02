import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import tree

def get_data():
    ''' Function to get Mushroom data'''
    data = pd.read_csv("Mushroom.csv",names=["label","cap-shape","cap-surface","cap-color","bruises?","odor","gill-attachement"    
                        ,"gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring",
                        "stalk-surface-below-ring", "stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color",
                        "ring-number","ring-type","spore-print-color","population","habitat"])

    train_df = data.iloc[:6000]
    X_train = train_df[train_df.columns[1:]].values
    y_train = train_df['label'].values 

    test_df = data.iloc[6000:]
    X_test = test_df[test_df.columns[1:]].values 
    y_test = test_df['label'].values

    return (train_df,X_train,y_train,test_df,X_test,y_test)



class RandomForest():
    def __init__(self, X, y, n_trees, n_features,depth=2):
        self.n_features = n_features
        self.n_features_arr = np.array(range(X.shape[1]))
        self.X, self.y, self.depth = X, y, depth
        self.trees = [tree.DecisionTreeClassifier(max_depth=self.depth) for i in range(n_trees)]
        self.fidxs = []

    def predict(self, X):
        return np.sign(np.mean([t.predict(X[:,self.fidxs[i]]) for i,t in enumerate(self.trees)], axis=0))

    def fit(self, X, y):

        for t in self.trees:
            np.random.shuffle(self.n_features_arr)
            fidx = self.n_features_arr[:self.n_features]
            self.fidxs.append(fidx)

            feats = X[:,fidx]
            t.fit(feats,y)


train_df, X_train, y_train, test_df, X_test, y_test = get_data()

feature_sets = [5,10,15,20]
train_accuracies = []
test_accuracies = []

for f in feature_sets:
    rf = RandomForest(X_train, y_train, 100, f)
    rf.fit(X_train, y_train)
    train_acc = 1 -  ((rf.predict(X_train) == y_train)*1).sum() / len(y_train)
    test_acc = ((rf.predict(X_test) == y_test)*1).sum() / len(y_test)
    
    train_accuracies.append(round(train_acc*100,4))
    test_accuracies.append(round(test_acc*100,4))


plt.figure()
plt.xlabel("Features in Feature Set")
plt.ylabel("Accuracies (%)")
plt.title("Accuracies On Training and Testing Data # Feature Sets")
plt.xticks(range(len(feature_sets)),("5","10","15","20"))
plt.plot(range(len(feature_sets)),train_accuracies,label="train")
plt.plot(range(len(feature_sets)),test_accuracies,label="test")
plt.legend()
plt.show()


estimators = [10, 20, 40, 80, 100]
train_accuracies = []
test_accuracies = []

for e in estimators:
    rf = RandomForest(X_train, y_train, e, 20)
    rf.fit(X_train, y_train)
    train_acc = ((rf.predict(X_train) == y_train)*1).sum() / len(y_train)
    test_acc = ((rf.predict(X_test) == y_test)*1).sum() / len(y_test)
    
    train_accuracies.append(round(train_acc*100,4))
    test_accuracies.append(round(test_acc*100,4))


plt.figure()
plt.xlabel("# Of Decision Trees")
plt.ylabel("Accuracies (%)")
plt.title("Accuracies On Training and Testing Data # Decision Trees")
plt.xticks(range(len(estimators)),("10","20","40","80","100"))
plt.plot(range(len(estimators)),train_accuracies,label="train")
plt.plot(range(len(estimators)),test_accuracies,label="test")
plt.legend()
plt.show()


