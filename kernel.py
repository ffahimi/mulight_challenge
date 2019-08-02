import numpy as np 
import pandas as pd 
# Input data files should be placed in the "../input/" directory.
import warnings

INFO_PATH = '../input/heroes_information.csv'

warnings.filterwarnings('ignore')


def main():
	heros_info = pd.read_csv(INFO_PATH)
	print(heros_info['Race'])


if __name__ == '__main__':
    main()