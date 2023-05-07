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

features = ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]

data.loc[data.Direction == "Down", 'Direction'] = 0
data.loc[data.Direction == "Up", 'Direction'] = 1

def log_regression(features, data, split=False):
	X = data[features]

	y = data["Direction"].astype('int')

	X = sm.add_constant(X)

	if not split: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
	else:
		X_train = data[data.Year < 2009][features]
		X_test = data[data.Year >=2009][features]
		y_train = data[data.Year < 2009]["Direction"].astype("int")
		y_test = data[data.Year >=2009]["Direction"].astype("int")

	logit_model = sm.Logit(y_train, X_train)

	result = logit_model.fit()

	print(result.summary())

	p_values = result.pvalues
	print(p_values)

	pred = result.predict(X_test)
	prediction = list(map(round, pred))

	cnf_matrix = metrics.confusion_matrix(prediction, y_test)
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

	print(f'accuracy score: {metrics.accuracy_score(y_test, prediction)}')

log_regression(features, data)
log_regression("Lag2", data, split=True)