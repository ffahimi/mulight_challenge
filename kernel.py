import numpy as np 
import pandas as pd 
# Input data files should be placed in the "../input/" directory.
import warnings

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

	print(heros_info.columns[heros_info.isnull().any()])
	
if __name__ == '__main__':
    main()