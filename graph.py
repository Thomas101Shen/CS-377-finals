import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm

data = pd.read_csv("./cs377-Weekly.csv")
# for i in range(len(data["Direction"])):
# 	if data["Direction"][i]=="Down": data["Direction"][i]=0
# 	else: data["Direction"][i]=1


data.loc[data.Direction == "Down", 'Direction'] = 0
data.loc[data.Direction == "Up", 'Direction'] = 1
indicies = []
for i in range(len(data)):
	indicies.append(i+1)

data.insert(1, "Week", indicies)
data = data.drop("Year", axis=1)

print(data.columns)
print(data)


print(data.Lag1.mean(),
data.Lag1.std(ddof=0),
data.Lag1.var(ddof=0),
)

print(data.Lag1.mean(),
data.Lag2.std(ddof=0),
data.Lag2.var(ddof=0))

print(data.Lag1.mean(),
data.Lag3.std(ddof=0),
data.Lag3.var(ddof=0))

print(data.Lag1.mean(),
data.Lag4.std(ddof=0),
data.Lag4.var(ddof=0))

print(data.Lag1.mean(),
data.Lag5.std(ddof=0),
data.Lag5.var(ddof=0))

y = data.Week
x1 = data.Lag1
x2 = data.Lag2
x3 = data.Lag3
x4 = data.Lag4
x5 = data.Lag5

plt.ylabel("Week")

plt.plot(x1, y)
# plt.show()
plt.plot(x2, y)
# plt.show()
plt.plot(x3, y)
# plt.show()
plt.plot(x4, y)
# plt.show()
plt.plot(x5, y)
plt.show()

feature_cols = ["Lag1" ,"Lag2", "Lag3", "Lag4", "Lag5", "Volume"]

X = data[feature_cols]

# target = data.Direction.values.flatten()
# print(target)

target = data["Direction"].astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=16)

log_reg = LogisticRegression(random_state=16)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

correct = [i for i in range(len(y_pred)) if y_pred.ravel()[i] == y_test.ravel()[i]]
print("percentage correct: ", len(correct)/len(y_pred))

# obs = pd.crosstab()

# chi2, p, dof, expected = chi2_contingency(obs)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names = ["Down", "Up"]
print(cnf_matrix)

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual direction')
plt.xlabel('Predicted direction')

plt.show()