import pandas as pd
import os
import seaborn

dir = os.path.dirname(__file__)

data = pd.read_csv(dir + 'Data/dac_sample.txt', sep='\t', header=None)
print data.head()

X = data.ix[:, 1:]
Y = data.ix[:, 0]
