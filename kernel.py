import numpy as np 
import pandas as pd 
# Input data files should be placed in the "../input/" directory.
import warnings
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

INFO_PATH = '../input/heroes_information.csv'
POWER_PATH = './../input/super_hero_powers.csv'

warnings.filterwarnings('ignore')


def main():

	# read superhero info csv
	heros_info = pd.read_csv(INFO_PATH)

	# index human/non-human to 1/0 values
	heros_info['Human'] = heros_info['Race'].apply(lambda i: 1 if i == 'Human' else 0)

	# drop the unnamed values
	heros_info.drop(['Unnamed: 0'], inplace=True, axis=1)

	# Get weight and height median and fill the NA values
	heros_info['Weight'].fillna(heros_info['Weight'].median(), inplace=True)
	heros_info['Publisher'].fillna(heros_info['Publisher'].mode()[0], inplace=True)

	# Filling the data of height and weight with median values
	imp = Imputer(missing_values=-99.0, strategy='median', axis=0)
	heros_info["Height"]=imp.fit_transform(heros_info[["Height"]])
	heros_info["Weight"]=imp.fit_transform(heros_info[["Weight"]])

	# ...............................
	# Reading powers csv
	heros_power = pd.read_csv(POWER_PATH)

	power_cat_columns = heros_power.columns.drop("hero_names")
	# index categorical data of hero powers
	heros_power_dummies = pd.get_dummies(heros_power, columns=power_cat_columns)

	info_cat_columns = ['Gender', 'Eye color', 'Hair color', 'Publisher', 'Skin color', 'Alignment']
	# index categorical data of hero info
	heros_info_dummies = pd.get_dummies(heros_info, columns=info_cat_columns)
	
	# ..................................
	# join data files on name column
	heros = pd.merge(heros_info_dummies, heros_power_dummies, left_on=['name'], right_on=['hero_names'], how='inner')

	# ..................................
	# Prepare input and output variables X, y
	X_columns_drop = ['name', 'hero_names', 'Race', 'Human']
	X, y = heros.drop(X_columns_drop, axis=1), heros['Human']

	# ....................................
	# Initialize a stratified split of our dataset for the validation process
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	# Initialize the classifier with the default parameters 
	rfc = RandomForestClassifier(random_state=42, max_depth=8, n_jobs=-1, oob_score=True)
	# Train it on the training set
	results = cross_val_score(rfc, X, y, cv=skf)
	# Evaluate the accuracy on the test set
	print("RandomForest CV accuracy score: {:.2f}%".format(results.mean()*100))
	
	# Calculating confution matrix for randomforest, more description in readme
	y_pred = cross_val_predict(rfc, X, y, cv=5)
	conf_mat = confusion_matrix(y, y_pred)
	print("RandomForest ConfusionMatrix: {}".format(conf_mat))

	# ............................
	kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
	# Initialize the classifier with the default parameters 
	lgr = LogisticRegression(C=2)
	# Train it on the training set
	results = cross_val_score(lgr, X, y, cv=skf)
	# Evaluate the accuracy on the test set
	print("Logistic Regression CV accuracy score: {:.2f}%".format(results.mean()*100))

	y_pred = cross_val_predict(lgr, X, y, cv=5)
	conf_mat = confusion_matrix(y, y_pred)
	print("LogisticRegression ConfusionMatrix: {}".format(conf_mat))


if __name__ == '__main__':
    main()