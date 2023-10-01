import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


if __name__ == '__main__':

    # READ DATA FILES
    path = 'data'
    filename_read = os.path.join(path, "Admission_Predict_Ver1.1_small_data_set_for_Linear_Regression.csv")
    data = pd.read_csv(filename_read, na_values=['NA', '?'])

    # Dropping useless data field and its corresponding data
    outpath = os.path.join(path, "out.csv")

    print('Number of instances = %d' % (data.shape[0]))
    print('Number of attributes = %d' % (data.shape[1]))

    # Correct column names without spaces
    data.columns = ['SerialNo', 'GREScore', 'TOEFLScore', 'UniversityRating', 'SOP', 'LOR', 'CGPA', 'Research',
                    'ChanceOfAdmit']

    # Drop the 'SerialNo' column and its corresponding data
    data = data.drop(['SerialNo'], axis=1)

    # Dropping all rows that contain missing values
    data = data.replace('?', np.NaN)

    print('Number of instances = %d' % (data.shape[0]))
    print('Number of attributes = %d' % (data.shape[1]))

    print('Number of missing values:')
    for col in data.columns:
        print('\t%s: %d' % (col, data[col].isna().sum()))

    # dropping all rows that contain missing values
    data = data.dropna(axis='rows')

    # Writing modified data frame to csv file
    data.to_csv(outpath, index=False)  # Set index=False to avoid writing row numbers to the file

    # Read the existing CSV file into a Pandas DataFrame
    filename_read = os.path.join(path, 'out.csv')
    data = pd.read_csv(filename_read)

    # Define the number of random data points you want to generate
    num_random_points = 1000  # Change this to the desired number of data points

    # Initialize lists to store data for each column
    GREScore = []
    TOEFLScore = []
    UniversityRating = []
    SOP = []
    LOR = []
    CGPA = []
    Research = []
    ChanceOfAdmit = []

    # Generate random data points and append them to the respective lists
    for _ in range(num_random_points):
        GREScore.append(random.randint(260, 340))
        TOEFLScore.append(random.randint(80, 120))
        UniversityRating.append(random.randint(1, 5))
        SOP.append(random.randint(1, 5))
        LOR.append(random.randint(1, 5))
        CGPA.append(round(random.randint(2, 4), 2))
        Research.append(random.randint(0, 1))
        ChanceOfAdmit.append(round(random.randint(0, 1)))

    # Create a DataFrame from the generated data
    random_data = pd.DataFrame({
        'GREScore': GREScore,
        'TOEFLScore': TOEFLScore,
        'UniversityRating': UniversityRating,
        'SOP': SOP,
        'LOR': LOR,
        'CGPA': CGPA,
        'Research': Research,
        'ChanceOfAdmit': ChanceOfAdmit
    })

    # Create a DataFrame from the new data
    new_columns = ['SOP', 'LOR', 'CGPA', 'Research', 'ChanceOfAdmit', 'GREScore', 'TOEFLScore']
    new_df = pd.DataFrame(random_data, columns=new_columns)

    # Concatenate the new data with the existing DataFrame
    data = pd.concat([data, new_df], ignore_index=True)

    # Save the modified DataFrame back to the same CSV file
    data.to_csv(filename_read, index=False)

    # outliers
    data.boxplot(figsize=(20, 10))
    plt.figure(figsize=(20, 10))
    data.boxplot(showfliers=True)
    plt.title('Boxplot with Outliers')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    # plt.show()

    # Duplicate Data
    dups = data.duplicated()
    print('Number of duplicate rows = %d' % (dups.sum()))

    print('Number of rows before discarding duplicates = %d' % (data.shape[0]))
    data2 = data.drop_duplicates()
    print('Number of rows after discarding duplicates = %d' % (data2.shape[0]))

    # shuffle data
    df = data
    np.random.seed(38)  # uncomment this line to get the same shuffle each time

    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)
    # use inplace=False

    # sort the dataframes
    df = df.sort_values(by='GREScore', ascending=True)
    
    # processing data --- Start
    # remove_outliers for GREScore
    gre_score_column_mean = data['GREScore'].mean()
    gre_score_column_std = data['GREScore'].std()

    print(gre_score_column_mean)
    print(gre_score_column_std)

    # remove_outliers for GREScore
    print("Length before gre outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'GREScore', 2)
    print("Length after gre outliers dropped: {}".format(len(df)))

    # remove_outliers for TOEFLScore
    print("Length before TOEFLScore outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'TOEFLScore', 2)
    print("Length after TOEFLScore outliers dropped: {}".format(len(df)))

    # remove_outliers for UniversityRating
    print("Length before UniversityRating outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'TOEFLScore', 2)
    print("Length after UniversityRating outliers dropped: {}".format(len(df)))

    # remove_outliers for SOP
    print("Length before SOP outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'SOP', 2)
    print("Length after SOP outliers dropped: {}".format(len(df)))

    # remove_outliers for LOR
    print("Length before LOR outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'LOR', 2)
    print("Length after LOR outliers dropped: {}".format(len(df)))

    # remove_outliers for CGPA
    print("Length before CGPA outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'CGPA', 2)
    print("Length after CGPA outliers dropped: {}".format(len(df)))

    # remove_outliers for Research
    print("Length before Research outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'Research', 2)
    print("Length after Research outliers dropped: {}".format(len(df)))

    # remove_outliers for ChanceOfAdmit
    print("Length before ChanceOfAdmit outliers dropped: {}".format(len(df)))
    remove_outliers(df, 'ChanceOfAdmit', 2)
    print("Length after ChanceOfAdmit outliers dropped: {}".format(len(df)))
