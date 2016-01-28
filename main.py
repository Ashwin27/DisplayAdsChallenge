import pandas as pd
import os

dir = os.path.dirname(__file__)

data = pd.read_csv(dir + 'Data/dac_sample.txt', sep='\t')
print data.head()

