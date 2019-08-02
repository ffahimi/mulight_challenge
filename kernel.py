import numpy as np 
import pandas as pd 
# Input data files should be placed in the "../input/" directory.
import warnings
from sklearn.preprocessing import Imputer


INFO_PATH = '../input/heroes_information.csv'

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

	# Reading powers csv
	heros_power = pd.read_csv('./../input/super_hero_powers.csv')

	power_cat_columns = heros_power.columns.drop("hero_names")
	# index categorical data of hero powers
	heros_power_dummies = pd.get_dummies(heros_power, columns=power_cat_columns)

	info_cat_columns = ['Gender', 'Eye color', 'Hair color', 'Publisher', 'Skin color', 'Alignment']
	# index categorical data of hero info
	heros_info_dummies = pd.get_dummies(heros_info, columns=info_cat_columns)
	
	# join data files on name column
	heros = pd.merge(heros_info_dummies, heros_power_dummies, left_on=['name'], right_on=['hero_names'], how='inner')

	# Prepare input and output variables X, y
	X_columns_drop = ['name', 'hero_names', 'Race', 'Human']
	X, y = heros.drop(X_columns_drop, axis=1), heros['Human']



if __name__ == '__main__':
    main()