import os
import numpy as np
import pandas as pd

# READ DATA FILES
path = 'data'
print(os.path.abspath(path))
filename_read = os.path.join(path, "Admission_Predict_Ver1.1_small_data_set_for_Linear_Regression.csv")
data = pd.read_csv(filename_read, na_values=['NA', '?'])

# dropping useless data field
outpath = os.path.join(path, "out.csv")

# Correct column names without spaces
data.columns = ['SerialNo', 'GRE Score', 'TOEFL Score', 'UniversityRating', 'SOP', 'LOR', 'CGPA', 'Research',
                'ChanceOfAdmit']

print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))
data = data.drop(['SerialNo'], axis=1)

# dropping all rows that contain missing values
data = data.replace('?',np.NaN)

print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))

print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))
