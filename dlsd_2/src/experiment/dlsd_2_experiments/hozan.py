
import pandas as pd 

PATH_DATA = '/hartensa/data_other/hozan_S1186.csv'

def main():
	df = pd.read_csv(PATH_DATA,sep=";", converters={'Density': lambda x: float(x.replace('.','').replace(',','.'))}, header=0)
	idx_density = 7
	idx_timestamp = 1
	new_df = pd.DataFrame(df.iloc[:,idx_density].values)
	new_df.index = df.iloc[:,idx_timestamp]
	new_df.columns = ['1186']
	new_df.iloc
	new_df.to_csv('/hartensa/data_other/hozan_S1186_for_model.csv')

if __name__ == '__main__':
	main()