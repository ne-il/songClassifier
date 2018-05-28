Ce projet vise à construire un algorithme capable d'estimer la date de sortie d'un morceau de musique à partir de descripeurs calulés à partir de la bande son de ce morceau. Le jeu de données à utiliser est disponible dans le répertoire projets de l'UV.

Le fichier SongApp.csv contient la base d'apprentissage la première colonne correspond à l'année de sortie. Les 90 colonnes restantes correpondent au descripteurs. Vous serez évalué sur le second fichier SongTst.csv qui ne contient que les descripteurs de morceaus qui ne sont pas présent dans la base d'apprentissage.

En plus du notebook détaillant votre travail, vous devrez fournir un fichier resultats_nom1nom2....txt avec une date par ligne correspondant aux prédiction de votre algorithme sur la base de test.





```python
import pip
import numpy as np
import pandas as pd
import sklearn
import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error


```


```python
song_app = pd.read_csv("../../SongApp.csv",header=-1,)
```


```python
# clean data
df_song = song_app.dropna()
df_song.__delitem__(0)
labels_column = df_song.iloc[0:,0]
labels = df_song.values[0:,0]
labels
df_song.__delitem__(1)
df_song['labels'] = labels_column
df_song.head()
```

    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>...</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
      <th>91</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>48.73215</td>
      <td>18.42930</td>
      <td>70.32679</td>
      <td>12.94636</td>
      <td>-10.32437</td>
      <td>-24.83777</td>
      <td>8.76630</td>
      <td>-0.92019</td>
      <td>18.76548</td>
      <td>4.59210</td>
      <td>...</td>
      <td>-19.68073</td>
      <td>33.04964</td>
      <td>42.87836</td>
      <td>-9.90378</td>
      <td>-32.22788</td>
      <td>70.49388</td>
      <td>12.04941</td>
      <td>58.43453</td>
      <td>26.92061</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50.95714</td>
      <td>31.85602</td>
      <td>55.81851</td>
      <td>13.41693</td>
      <td>-6.57898</td>
      <td>-18.54940</td>
      <td>-3.27872</td>
      <td>-2.35035</td>
      <td>16.07017</td>
      <td>1.39518</td>
      <td>...</td>
      <td>26.05866</td>
      <td>-50.92779</td>
      <td>10.93792</td>
      <td>-0.07568</td>
      <td>43.20130</td>
      <td>-115.00698</td>
      <td>-0.05859</td>
      <td>39.67068</td>
      <td>-0.66345</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48.24750</td>
      <td>-1.89837</td>
      <td>36.29772</td>
      <td>2.58776</td>
      <td>0.97170</td>
      <td>-26.21683</td>
      <td>5.05097</td>
      <td>-10.34124</td>
      <td>3.55005</td>
      <td>-6.36304</td>
      <td>...</td>
      <td>-171.70734</td>
      <td>-16.96705</td>
      <td>-46.67617</td>
      <td>-12.51516</td>
      <td>82.58061</td>
      <td>-72.08993</td>
      <td>9.90558</td>
      <td>199.62971</td>
      <td>18.85382</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.97020</td>
      <td>42.20998</td>
      <td>67.09964</td>
      <td>8.46791</td>
      <td>-15.85279</td>
      <td>-16.81409</td>
      <td>-12.48207</td>
      <td>-9.37636</td>
      <td>12.63699</td>
      <td>0.93609</td>
      <td>...</td>
      <td>-55.95724</td>
      <td>64.92712</td>
      <td>-17.72522</td>
      <td>-1.49237</td>
      <td>-7.50035</td>
      <td>51.76631</td>
      <td>7.88713</td>
      <td>55.66926</td>
      <td>28.74903</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50.54767</td>
      <td>0.31568</td>
      <td>92.35066</td>
      <td>22.38696</td>
      <td>-25.51870</td>
      <td>-19.04928</td>
      <td>20.67345</td>
      <td>-5.19943</td>
      <td>3.63566</td>
      <td>-4.69088</td>
      <td>...</td>
      <td>-50.69577</td>
      <td>26.02574</td>
      <td>18.94430</td>
      <td>-0.33730</td>
      <td>6.09352</td>
      <td>35.18381</td>
      <td>5.00283</td>
      <td>-11.02257</td>
      <td>0.02263</td>
      <td>2001</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 91 columns</p>
</div>




```python
num_columns = df_song.columns.shape[0]
columns = list(range(num_columns - 1))
columns_name = list(map(str,columns)) + ['labels']
df_song.columns = columns_name
df_song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>48.73215</td>
      <td>18.42930</td>
      <td>70.32679</td>
      <td>12.94636</td>
      <td>-10.32437</td>
      <td>-24.83777</td>
      <td>8.76630</td>
      <td>-0.92019</td>
      <td>18.76548</td>
      <td>4.59210</td>
      <td>...</td>
      <td>-19.68073</td>
      <td>33.04964</td>
      <td>42.87836</td>
      <td>-9.90378</td>
      <td>-32.22788</td>
      <td>70.49388</td>
      <td>12.04941</td>
      <td>58.43453</td>
      <td>26.92061</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50.95714</td>
      <td>31.85602</td>
      <td>55.81851</td>
      <td>13.41693</td>
      <td>-6.57898</td>
      <td>-18.54940</td>
      <td>-3.27872</td>
      <td>-2.35035</td>
      <td>16.07017</td>
      <td>1.39518</td>
      <td>...</td>
      <td>26.05866</td>
      <td>-50.92779</td>
      <td>10.93792</td>
      <td>-0.07568</td>
      <td>43.20130</td>
      <td>-115.00698</td>
      <td>-0.05859</td>
      <td>39.67068</td>
      <td>-0.66345</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48.24750</td>
      <td>-1.89837</td>
      <td>36.29772</td>
      <td>2.58776</td>
      <td>0.97170</td>
      <td>-26.21683</td>
      <td>5.05097</td>
      <td>-10.34124</td>
      <td>3.55005</td>
      <td>-6.36304</td>
      <td>...</td>
      <td>-171.70734</td>
      <td>-16.96705</td>
      <td>-46.67617</td>
      <td>-12.51516</td>
      <td>82.58061</td>
      <td>-72.08993</td>
      <td>9.90558</td>
      <td>199.62971</td>
      <td>18.85382</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.97020</td>
      <td>42.20998</td>
      <td>67.09964</td>
      <td>8.46791</td>
      <td>-15.85279</td>
      <td>-16.81409</td>
      <td>-12.48207</td>
      <td>-9.37636</td>
      <td>12.63699</td>
      <td>0.93609</td>
      <td>...</td>
      <td>-55.95724</td>
      <td>64.92712</td>
      <td>-17.72522</td>
      <td>-1.49237</td>
      <td>-7.50035</td>
      <td>51.76631</td>
      <td>7.88713</td>
      <td>55.66926</td>
      <td>28.74903</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50.54767</td>
      <td>0.31568</td>
      <td>92.35066</td>
      <td>22.38696</td>
      <td>-25.51870</td>
      <td>-19.04928</td>
      <td>20.67345</td>
      <td>-5.19943</td>
      <td>3.63566</td>
      <td>-4.69088</td>
      <td>...</td>
      <td>-50.69577</td>
      <td>26.02574</td>
      <td>18.94430</td>
      <td>-0.33730</td>
      <td>6.09352</td>
      <td>35.18381</td>
      <td>5.00283</td>
      <td>-11.02257</td>
      <td>0.02263</td>
      <td>2001</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 91 columns</p>
</div>




```python
df_song.shape
```




    (463715, 91)



# Let's Split before balance


```python
# x_data = df_resampled.sample(100000)
x_data = df_song

y = x_data['labels']
X = x_data.iloc[:, 0:89]

```


```python
# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (417343, 89)
    (46372, 89)
    (417343,)
    (46372,)



```python
reunited = pd.DataFrame(X_train)
labels = pd.DataFrame(y_train)
reunited['labels'] = labels
reunited.shape
```




    (417343, 90)



# LET'S BALANCE THE DATASET


```python
# value_count = song_app['labels'].value_counts()
df = reunited
# sort the dataframe
df.sort_values('labels', inplace=True)
# set the index to be this and don't drop
df.set_index(keys=['labels'], drop=False,inplace=True)
# get a list of names
labels=df['labels'].unique().tolist()
# now we can perform a lookup on a 'view' of the dataframe

df_list = []
for label in labels:
    label_df = df.loc[df['labels'] == label]
    df_list.append(label_df)

# joe = df.loc[df['labels'] == 2008]


# nb_instance_target = 5210
nb_instance_target = 15210

# print('before reseampling:')
# print df_list[88]
# print(len(df_list[88]))

new_df_list = []
for df in df_list:
    if len(df) > nb_instance_target:
        df_majority_downsampled = resample(df, 
                                 replace=False,    # sample without replacement
                                 n_samples=nb_instance_target,     # to match minority class
                                 random_state=123)
        new_df_list.append(df_majority_downsampled)
    elif len(df) < nb_instance_target:
        df_minority_upsampled = resample(df, 
                                 replace=True,     # sample with replacement
                                 n_samples= nb_instance_target,    # to match majority class
                                 random_state=123)
        new_df_list.append(df_minority_upsampled)

# print('After reseampling:')
# print(new_df_list[88])
# print(len(new_df_list[88]))   

df_resampled = pd.concat(new_df_list)
df_resampled = df_resampled.reset_index(drop=True)

print (reunited.shape)
print (df_resampled.shape)
```

    (417343, 90)
    (1353690, 90)


# TEST WITH BALANCED


```python
# x_data = df_resampled.sample(100000)
X_balanced = df_resampled

y_balanced = X_balanced['labels']
X_balanced = X_balanced.iloc[:, 0:89]
```


```python
# split into training and test data
# X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.1, random_state=42)
X_train, y_train = X_balanced, y_balanced

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (1353690, 89)
    (46372, 89)
    (1353690,)
    (46372,)



```python
from sklearn import preprocessing
# X_train_scaled = preprocessing.scale(X_train)
# y_train_scaled = preprocessing.scale(y_train)
```


```python
clf = RandomForestRegressor(n_jobs=5, n_estimators=50, verbose=2)
model = clf.fit(X_train, y_train)
```

    building tree 1 of 50
    building tree 3 of 50building tree 4 of 50
    building tree 2 of 50
    
    building tree 5 of 50
    building tree 6 of 50
    building tree 7 of 50building tree 8 of 50
    
    building tree 9 of 50
    building tree 10 of 50
    building tree 11 of 50
    building tree 12 of 50
    building tree 13 of 50
    building tree 14 of 50
    building tree 15 of 50
    building tree 16 of 50
    building tree 17 of 50
    building tree 18 of 50
    building tree 19 of 50
    building tree 20 of 50
    building tree 21 of 50
    building tree 22 of 50
    building tree 23 of 50
    building tree 24 of 50
    building tree 25 of 50
    building tree 26 of 50
    building tree 27 of 50
    building tree 28 of 50
    building tree 29 of 50
    building tree 30 of 50
    building tree 31 of 50
    building tree 32 of 50
    building tree 33 of 50
    building tree 34 of 50
    building tree 35 of 50


    [Parallel(n_jobs=5)]: Done  31 tasks      | elapsed: 22.7min


    building tree 36 of 50
    building tree 37 of 50
    building tree 38 of 50
    building tree 39 of 50
    building tree 40 of 50
    building tree 41 of 50
    building tree 42 of 50
    building tree 43 of 50
    building tree 44 of 50
    building tree 45 of 50
    building tree 46 of 50
    building tree 47 of 50
    building tree 48 of 50
    building tree 49 of 50
    building tree 50 of 50


    [Parallel(n_jobs=5)]: Done  50 out of  50 | elapsed: 32.7min finished


# On a tous mis a 15 000 et on en entrainer le model sur l integralite des 1 500 000 entree 

# Maintenant on va tester sur les 40 000 trucs de test qu on a pas du tout utiliser pour entrainer le model


```python
y_predict_test = model.predict(X_test)
```

    [Parallel(n_jobs=5)]: Done  31 tasks      | elapsed:    0.9s
    [Parallel(n_jobs=5)]: Done  50 out of  50 | elapsed:    1.5s finished



```python
def log_errors(y_test, predicted):
    mae = mean_absolute_error(y_test, predicted)
    mse = mean_squared_error(y_test, predicted, multioutput='raw_values')
    r2 = r2_score(y_test, predicted)
    medae = median_absolute_error(y_test, predicted)
    print('mae ',mae)
    print('mse ',mse)
    print('r2 score ', r2)
    print('median_ae ',medae)
```


```python
log_errors(y_test=y_test,predicted=y_predict_test)
```

    ('mae ', 7.518526463013394)
    ('mse ', array([95.79563938]))
    ('r2 score ', 0.19995074203161456)
    ('median_ae ', 6.0)



```python
list_test = np.around(y_test)

# list_predict = predict_y.astype('int')
list_predict = np.around(y_predict_test)

# print list_test[0]
# print list_predict[0]
print(classification_report(list_predict, list_test ))
```

                 precision    recall  f1-score   support
    
         1922.0       0.00      0.00      0.00         0
         1925.0       0.00      0.00      0.00         0
         1926.0       0.00      0.00      0.00         0
         1927.0       0.00      0.00      0.00         0
         1928.0       0.00      0.00      0.00         0
         1929.0       0.00      0.00      0.00         0
         1930.0       0.00      0.00      0.00         0
         1931.0       0.00      0.00      0.00         0
         1932.0       0.00      0.00      0.00         0
         1934.0       0.00      0.00      0.00         0
         1935.0       0.00      0.00      0.00         1
         1936.0       0.00      0.00      0.00         1
         1937.0       0.00      0.00      0.00         0
         1938.0       0.00      0.00      0.00         1
         1939.0       0.00      0.00      0.00         0
         1940.0       0.00      0.00      0.00         0
         1941.0       0.00      0.00      0.00         2
         1942.0       0.00      0.00      0.00         0
         1944.0       0.00      0.00      0.00         1
         1945.0       0.00      0.00      0.00         0
         1946.0       0.00      0.00      0.00         1
         1947.0       0.00      0.00      0.00         1
         1948.0       0.00      0.00      0.00         1
         1949.0       0.00      0.00      0.00         1
         1950.0       0.00      0.00      0.00         0
         1951.0       0.00      0.00      0.00         1
         1952.0       0.00      0.00      0.00         0
         1953.0       0.00      0.00      0.00         1
         1954.0       0.00      0.00      0.00         1
         1955.0       0.00      0.00      0.00         1
         1956.0       0.00      0.00      0.00         1
         1957.0       0.00      0.00      0.00         1
         1958.0       0.00      0.00      0.00         0
         1959.0       0.00      0.00      0.00         1
         1960.0       0.00      0.00      0.00         2
         1961.0       0.00      0.00      0.00         2
         1962.0       0.00      0.00      0.00         3
         1963.0       0.00      0.00      0.00         6
         1964.0       0.00      0.00      0.00         6
         1965.0       0.00      0.00      0.00         5
         1966.0       0.00      0.00      0.00         5
         1967.0       0.00      0.00      0.00         9
         1968.0       0.00      0.00      0.00        12
         1969.0       0.00      0.00      0.00         7
         1970.0       0.00      0.00      0.00        14
         1971.0       0.00      0.00      0.00        12
         1972.0       0.00      0.00      0.00        28
         1973.0       0.00      0.04      0.01        25
         1974.0       0.00      0.00      0.00        23
         1975.0       0.00      0.00      0.00        33
         1976.0       0.00      0.00      0.00        65
         1977.0       0.00      0.00      0.00        66
         1978.0       0.00      0.00      0.00        91
         1979.0       0.00      0.01      0.00       147
         1980.0       0.01      0.02      0.01       192
         1981.0       0.01      0.01      0.01       291
         1982.0       0.02      0.01      0.02       403
         1983.0       0.02      0.02      0.02       496
         1984.0       0.03      0.01      0.02       649
         1985.0       0.04      0.02      0.03       780
         1986.0       0.05      0.02      0.03      1015
         1987.0       0.05      0.02      0.03      1129
         1988.0       0.08      0.03      0.04      1464
         1989.0       0.09      0.03      0.05      1505
         1990.0       0.10      0.03      0.05      1845
         1991.0       0.07      0.03      0.04      1925
         1992.0       0.09      0.03      0.05      2181
         1993.0       0.07      0.03      0.04      2089
         1994.0       0.09      0.04      0.05      2415
         1995.0       0.08      0.04      0.05      2388
         1996.0       0.06      0.03      0.04      2529
         1997.0       0.07      0.04      0.05      2561
         1998.0       0.08      0.04      0.05      2814
         1999.0       0.09      0.05      0.06      2747
         2000.0       0.08      0.05      0.06      2996
         2001.0       0.08      0.06      0.07      2835
         2002.0       0.09      0.07      0.08      2781
         2003.0       0.09      0.09      0.09      2436
         2004.0       0.06      0.09      0.07      1799
         2005.0       0.03      0.09      0.04      1024
         2006.0       0.02      0.16      0.03       399
         2007.0       0.00      0.18      0.01        98
         2008.0       0.00      0.12      0.00         8
         2009.0       0.00      0.00      0.00         0
         2010.0       0.00      0.00      0.00         0
    
    avg / total       0.07      0.05      0.05     46372
    


    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)



```python
# x_data = df_resampled.reset_index(drop=True)
x_data = df_song
# x_pif = df_resampled.sample(500)
x_pif = x_data
y_pif = x_pif['labels']
x_pif = x_pif.iloc[:, 0:89]
y_predict_pif = model.predict(x_pif)
log_errors(y_test=y_pif,predicted=y_predict_pif)
```

    ('mae ', 3.892168735140922)
    ('mse ', array([37.0023209]))
    ('r2 score ', 0.6908186890594709)
    ('median_ae ', 2.2200000000000273)



```python
list_test = np.around(y_pif)

# list_predict = predict_y.astype('int')
list_predict = np.around(y_predict_pif)

# print list_test[0]
# print list_predict[0]
print(classification_report(list_predict, list_test ))
```

                 precision    recall  f1-score   support
    
         1922.0       1.00      1.00      1.00         6
         1924.0       1.00      1.00      1.00         5
         1925.0       1.00      1.00      1.00         7
         1926.0       1.00      1.00      1.00        19
         1927.0       1.00      1.00      1.00        40
         1928.0       1.00      1.00      1.00        48
         1929.0       1.00      1.00      1.00        79
         1930.0       1.00      1.00      1.00        38
         1931.0       1.00      1.00      1.00        31
         1932.0       1.00      1.00      1.00        11
         1933.0       1.00      1.00      1.00         6
         1934.0       1.00      1.00      1.00        28
         1935.0       1.00      1.00      1.00        24
         1936.0       1.00      1.00      1.00        22
         1937.0       1.00      1.00      1.00        25
         1938.0       1.00      1.00      1.00        19
         1939.0       1.00      1.00      1.00        35
         1940.0       1.00      1.00      1.00        14
         1941.0       1.00      1.00      1.00        31
         1942.0       1.00      1.00      1.00        21
         1943.0       1.00      0.93      0.96        14
         1944.0       1.00      0.82      0.90        17
         1945.0       1.00      0.93      0.96        29
         1946.0       1.00      0.97      0.98        30
         1947.0       1.00      0.96      0.98        57
         1948.0       1.00      0.97      0.99        39
         1949.0       1.00      0.98      0.99        54
         1950.0       1.00      0.98      0.99        59
         1951.0       1.00      0.95      0.98        65
         1952.0       1.00      0.97      0.98        67
         1953.0       1.00      0.98      0.99       124
         1954.0       1.00      0.96      0.98       112
         1955.0       1.00      0.97      0.99       260
         1956.0       1.00      0.99      1.00       538
         1957.0       1.00      1.00      1.00       561
         1958.0       1.00      0.99      0.99       528
         1959.0       1.00      0.98      0.99       553
         1960.0       1.00      0.99      0.99       402
         1961.0       1.00      0.99      1.00       526
         1962.0       1.00      0.98      0.99       592
         1963.0       1.00      0.99      0.99       873
         1964.0       1.00      0.99      0.99       858
         1965.0       1.00      0.99      0.99      1010
         1966.0       0.99      0.99      0.99      1209
         1967.0       0.96      0.98      0.97      1486
         1968.0       0.93      0.96      0.95      1607
         1969.0       0.88      0.93      0.90      1871
         1970.0       0.84      0.89      0.87      2012
         1971.0       0.90      0.85      0.88      1974
         1972.0       0.87      0.88      0.88      2061
         1973.0       0.81      0.86      0.84      2187
         1974.0       0.90      0.80      0.85      2227
         1975.0       0.85      0.86      0.85      2274
         1976.0       0.93      0.80      0.86      2251
         1977.0       0.87      0.83      0.85      2353
         1978.0       0.81      0.80      0.81      2626
         1979.0       0.76      0.74      0.75      2941
         1980.0       0.80      0.69      0.74      3240
         1981.0       0.79      0.66      0.72      3359
         1982.0       0.73      0.61      0.67      3955
         1983.0       0.76      0.55      0.64      4210
         1984.0       0.79      0.50      0.61      4832
         1985.0       0.77      0.46      0.58      5311
         1986.0       0.72      0.42      0.53      6322
         1987.0       0.63      0.39      0.48      7401
         1988.0       0.60      0.33      0.43      9111
         1989.0       0.54      0.31      0.40     10317
         1990.0       0.49      0.27      0.35     12139
         1991.0       0.42      0.25      0.32     12841
         1992.0       0.40      0.23      0.29     15119
         1993.0       0.34      0.20      0.26     16011
         1994.0       0.31      0.18      0.23     18445
         1995.0       0.28      0.17      0.21     19240
         1996.0       0.26      0.15      0.19     21892
         1997.0       0.25      0.14      0.18     23437
         1998.0       0.25      0.13      0.17     26598
         1999.0       0.18      0.11      0.14     26242
         2000.0       0.18      0.10      0.13     29271
         2001.0       0.16      0.11      0.13     28751
         2002.0       0.16      0.11      0.13     29547
         2003.0       0.13      0.12      0.12     26327
         2004.0       0.09      0.11      0.10     22745
         2005.0       0.06      0.11      0.08     16655
         2006.0       0.03      0.10      0.05     11645
         2007.0       0.01      0.07      0.02      6004
         2008.0       0.01      0.06      0.01      2606
         2009.0       0.00      0.02      0.00      1686
         2010.0       0.18      1.00      0.30      1499
         2011.0       1.00      1.00      1.00         1
    
    avg / total       0.33      0.24      0.27    463715
    


# TEST WITH UNBALANCED


```python
# x_data = df_resampled.sample(100000)
df_song ['labels'] = df_song.index
df_song = df_song.reset_index(drop=True)
df_song.head()
x_data = df_song

labels = x_data['labels']
x_data = x_data.iloc[:, 0:89]
labels
```

    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46.15136</td>
      <td>66.08336</td>
      <td>-44.98747</td>
      <td>-4.59621</td>
      <td>2.79035</td>
      <td>0.24898</td>
      <td>4.34126</td>
      <td>-8.37581</td>
      <td>1.12948</td>
      <td>-9.39140</td>
      <td>...</td>
      <td>-153.70438</td>
      <td>34.93246</td>
      <td>25.14792</td>
      <td>48.56976</td>
      <td>54.46778</td>
      <td>-274.15587</td>
      <td>16.24013</td>
      <td>78.75953</td>
      <td>-27.18012</td>
      <td>1922</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43.68703</td>
      <td>39.49153</td>
      <td>-14.03862</td>
      <td>26.93750</td>
      <td>17.67678</td>
      <td>11.39290</td>
      <td>4.74967</td>
      <td>4.51429</td>
      <td>-9.64881</td>
      <td>-6.04461</td>
      <td>...</td>
      <td>-53.45490</td>
      <td>-43.25854</td>
      <td>145.20986</td>
      <td>2.00722</td>
      <td>125.91492</td>
      <td>-81.66189</td>
      <td>1.32684</td>
      <td>129.29909</td>
      <td>-7.95933</td>
      <td>1922</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.58633</td>
      <td>-159.68217</td>
      <td>160.66121</td>
      <td>35.99286</td>
      <td>13.20156</td>
      <td>9.03191</td>
      <td>-24.80241</td>
      <td>6.71746</td>
      <td>-6.86047</td>
      <td>-1.63318</td>
      <td>...</td>
      <td>-34.35791</td>
      <td>-18.00954</td>
      <td>16.68930</td>
      <td>-10.65305</td>
      <td>-4.19842</td>
      <td>61.22316</td>
      <td>-4.09660</td>
      <td>-119.28517</td>
      <td>4.75980</td>
      <td>1922</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39.96727</td>
      <td>41.88455</td>
      <td>-29.88949</td>
      <td>39.92319</td>
      <td>26.77335</td>
      <td>27.63348</td>
      <td>-14.13202</td>
      <td>-3.81200</td>
      <td>2.78896</td>
      <td>-3.14920</td>
      <td>...</td>
      <td>-214.49901</td>
      <td>267.35328</td>
      <td>396.87548</td>
      <td>-5.91930</td>
      <td>180.85378</td>
      <td>-285.43979</td>
      <td>37.66831</td>
      <td>111.37413</td>
      <td>-7.55992</td>
      <td>1922</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.96435</td>
      <td>64.51294</td>
      <td>4.82804</td>
      <td>15.56643</td>
      <td>27.77890</td>
      <td>-1.90337</td>
      <td>0.47859</td>
      <td>1.69384</td>
      <td>13.71945</td>
      <td>4.40608</td>
      <td>...</td>
      <td>-109.38127</td>
      <td>-88.11439</td>
      <td>157.30257</td>
      <td>8.55069</td>
      <td>9.77216</td>
      <td>-310.16497</td>
      <td>16.87295</td>
      <td>134.25595</td>
      <td>-37.19454</td>
      <td>1922</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 91 columns</p>
</div>




```python
x_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
    </tr>
    <tr>
      <th>labels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1922</th>
      <td>46.15136</td>
      <td>66.08336</td>
      <td>-44.98747</td>
      <td>-4.59621</td>
      <td>2.79035</td>
      <td>0.24898</td>
      <td>4.34126</td>
      <td>-8.37581</td>
      <td>1.12948</td>
      <td>-9.39140</td>
      <td>...</td>
      <td>3.52974</td>
      <td>-153.70438</td>
      <td>34.93246</td>
      <td>25.14792</td>
      <td>48.56976</td>
      <td>54.46778</td>
      <td>-274.15587</td>
      <td>16.24013</td>
      <td>78.75953</td>
      <td>-27.18012</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>43.68703</td>
      <td>39.49153</td>
      <td>-14.03862</td>
      <td>26.93750</td>
      <td>17.67678</td>
      <td>11.39290</td>
      <td>4.74967</td>
      <td>4.51429</td>
      <td>-9.64881</td>
      <td>-6.04461</td>
      <td>...</td>
      <td>-13.53449</td>
      <td>-53.45490</td>
      <td>-43.25854</td>
      <td>145.20986</td>
      <td>2.00722</td>
      <td>125.91492</td>
      <td>-81.66189</td>
      <td>1.32684</td>
      <td>129.29909</td>
      <td>-7.95933</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>37.58633</td>
      <td>-159.68217</td>
      <td>160.66121</td>
      <td>35.99286</td>
      <td>13.20156</td>
      <td>9.03191</td>
      <td>-24.80241</td>
      <td>6.71746</td>
      <td>-6.86047</td>
      <td>-1.63318</td>
      <td>...</td>
      <td>-6.63962</td>
      <td>-34.35791</td>
      <td>-18.00954</td>
      <td>16.68930</td>
      <td>-10.65305</td>
      <td>-4.19842</td>
      <td>61.22316</td>
      <td>-4.09660</td>
      <td>-119.28517</td>
      <td>4.75980</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>39.96727</td>
      <td>41.88455</td>
      <td>-29.88949</td>
      <td>39.92319</td>
      <td>26.77335</td>
      <td>27.63348</td>
      <td>-14.13202</td>
      <td>-3.81200</td>
      <td>2.78896</td>
      <td>-3.14920</td>
      <td>...</td>
      <td>20.58504</td>
      <td>-214.49901</td>
      <td>267.35328</td>
      <td>396.87548</td>
      <td>-5.91930</td>
      <td>180.85378</td>
      <td>-285.43979</td>
      <td>37.66831</td>
      <td>111.37413</td>
      <td>-7.55992</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>40.96435</td>
      <td>64.51294</td>
      <td>4.82804</td>
      <td>15.56643</td>
      <td>27.77890</td>
      <td>-1.90337</td>
      <td>0.47859</td>
      <td>1.69384</td>
      <td>13.71945</td>
      <td>4.40608</td>
      <td>...</td>
      <td>0.04034</td>
      <td>-109.38127</td>
      <td>-88.11439</td>
      <td>157.30257</td>
      <td>8.55069</td>
      <td>9.77216</td>
      <td>-310.16497</td>
      <td>16.87295</td>
      <td>134.25595</td>
      <td>-37.19454</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>




```python
# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-200-17032f1f6a2b> in <module>()
          1 # split into training and test data
    ----> 2 X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.1, random_state=42)
          3 print(X_train.shape)
          4 print(X_test.shape)
          5 print(y_train.shape)


    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/sklearn/model_selection/_split.pyc in train_test_split(*arrays, **options)
       2029         test_size = 0.25
       2030 
    -> 2031     arrays = indexable(*arrays)
       2032 
       2033     if shuffle is False:


    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/sklearn/utils/validation.pyc in indexable(*iterables)
        227         else:
        228             result.append(np.array(X))
    --> 229     check_consistent_length(*result)
        230     return result
        231 


    /Users/neil/Code/song_classifier/venv/lib/python2.7/site-packages/sklearn/utils/validation.pyc in check_consistent_length(*arrays)
        202     if len(uniques) > 1:
        203         raise ValueError("Found input variables with inconsistent numbers of"
    --> 204                          " samples: %r" % [int(l) for l in lengths])
        205 
        206 


    ValueError: Found input variables with inconsistent numbers of samples: [463715, 1353690]



```python
from sklearn import preprocessing
X_train_scaled = preprocessing.scale(X_train)
y_train_scaled = preprocessing.scale(y_train)
```


```python
clf = RandomForestRegressor(n_jobs=5, n_estimators=50)
unbalanced_model = clf.fit(X_train_scaled, y_train_scaled)
```


```python
X_test_scaled = preprocessing.scale(X_test)
y_test_scaled = preprocessing.scale(y_test)
predicted = unbalanced_model.predict(X_test_scaled)
```


```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
```


```python
def log_errors(y_test, predicted):
    mae = mean_absolute_error(y_test, predicted)
    mse = mean_squared_error(y_test, predicted, multioutput='raw_values')
    r2 = r2_score(y_test, predicted)
    medae = median_absolute_error(y_test, predicted)
    print('mae ',mae)
    print('mse ',mse)
    print('r2 score ', r2)
    print('median_ae ',medae)
```


```python
log_errors(y_test=y_test_scaled,predicted=predicted)
```

    mae  0.593008904188
    mse  [ 0.69729548]
    r2 score  0.30270452098
    median_ae  0.419795584823



```python
# x_data = df_resampled.reset_index(drop=True)
x_data = df_song
# x_pif = df_resampled.sample(500)
x_pif = x_data
y_pif = x_pif['labels']
x_pif = x_pif.iloc[:, 0:89]
y_predict_pif = model.predict(x_pif)
log_errors(y_test=y_pif,predicted=y_predict_pif)

list_test = np.around(y_pif)

# list_predict = predict_y.astype('int')
list_predict = np.around(y_predict_pif)

# print list_test[0]
# print list_predict[0]
print(classification_report(list_predict, list_test ))
```

# On fait tourner le model sur SongTest avec le model entraine avec les 1 500 000 entrees


```python
song_test = pd.read_csv("../datasets/SongTst.csv",header=-1,)
```


```python
song_test.shape
song_test.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>49.94357</td>
      <td>21.47114</td>
      <td>73.07750</td>
      <td>8.74861</td>
      <td>-17.40628</td>
      <td>-13.09905</td>
      <td>-25.01202</td>
      <td>-12.23257</td>
      <td>7.83089</td>
      <td>...</td>
      <td>13.01620</td>
      <td>-54.40548</td>
      <td>58.99367</td>
      <td>15.37344</td>
      <td>1.11144</td>
      <td>-23.08793</td>
      <td>68.40795</td>
      <td>-1.82223</td>
      <td>-27.46348</td>
      <td>2.26327</td>
    </tr>
    <tr>
      <th>1</th>
      <td>463715.0</td>
      <td>52.67814</td>
      <td>-2.88914</td>
      <td>43.95268</td>
      <td>-1.39209</td>
      <td>-14.93379</td>
      <td>-15.86877</td>
      <td>1.19379</td>
      <td>0.31401</td>
      <td>-4.44235</td>
      <td>...</td>
      <td>-5.74356</td>
      <td>-42.57910</td>
      <td>-2.91103</td>
      <td>48.72805</td>
      <td>-3.08183</td>
      <td>-9.38888</td>
      <td>-7.27179</td>
      <td>-4.00966</td>
      <td>-68.96211</td>
      <td>-5.21525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>463716.0</td>
      <td>45.74235</td>
      <td>12.02291</td>
      <td>11.03009</td>
      <td>-11.60763</td>
      <td>11.80054</td>
      <td>-11.12389</td>
      <td>-5.39058</td>
      <td>-1.11981</td>
      <td>-7.74086</td>
      <td>...</td>
      <td>-4.70606</td>
      <td>-24.22599</td>
      <td>-35.22686</td>
      <td>27.77729</td>
      <td>15.38934</td>
      <td>58.20036</td>
      <td>-61.12698</td>
      <td>-10.92522</td>
      <td>26.75348</td>
      <td>-5.78743</td>
    </tr>
    <tr>
      <th>3</th>
      <td>463717.0</td>
      <td>52.55883</td>
      <td>2.87222</td>
      <td>27.38848</td>
      <td>-5.76235</td>
      <td>-15.35766</td>
      <td>-15.01592</td>
      <td>-5.86893</td>
      <td>-0.31447</td>
      <td>-5.06922</td>
      <td>...</td>
      <td>-8.35215</td>
      <td>-16.86791</td>
      <td>-10.58277</td>
      <td>40.10173</td>
      <td>-0.54005</td>
      <td>-11.54746</td>
      <td>-45.35860</td>
      <td>-4.55694</td>
      <td>-43.17368</td>
      <td>-3.33725</td>
    </tr>
    <tr>
      <th>4</th>
      <td>463718.0</td>
      <td>51.34809</td>
      <td>9.02702</td>
      <td>25.33757</td>
      <td>-6.62537</td>
      <td>0.03367</td>
      <td>-12.69565</td>
      <td>-3.13400</td>
      <td>2.98649</td>
      <td>-6.71750</td>
      <td>...</td>
      <td>-6.87366</td>
      <td>-20.03371</td>
      <td>-66.38940</td>
      <td>50.56569</td>
      <td>0.27747</td>
      <td>67.05657</td>
      <td>-55.58846</td>
      <td>-7.50859</td>
      <td>28.23511</td>
      <td>-0.72045</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 91 columns</p>
</div>




```python
song_test = song_test.iloc[:, 1:]
song_test.head()
x_test = song_test
y_test_predict = model.predict(x_test)
```


```python
y_test_predict = np.round(y_test_predict)
new = list(map(lambda x: int(x),y_test_predict))
new
```




    [1992,
     1984,
     1978,
     1977,
     1985,
     1983,
     1976,
     1983,
     1992,
     1984,
     1990,
     1972,
     1985,
     1986,
     1988,
     1991,
     1993,
     1990,
     1985,
     1986,
     1990,
     1993,
     1988,
     1993,
     1986,
     1989,
     1988,
     1983,
     1991,
     1992,
     1986,
     1993,
     1992,
     1972,
     1992,
     1993,
     1990,
     1988,
     1969,
     1996,
     1985,
     1984,
     1985,
     1980,
     1982,
     1987,
     1986,
     1990,
     1985,
     1986,
     1981,
     1988,
     1990,
     1984,
     1986,
     1984,
     1987,
     1985,
     1984,
     1987,
     1986,
     1984,
     1989,
     1988,
     1987,
     1988,
     1988,
     1986,
     1986,
     1991,
     1966,
     1999,
     1990,
     1991,
     1992,
     1987,
     1986,
     1994,
     1969,
     1990,
     1990,
     1983,
     1991,
     1987,
     1986,
     1986,
     1991,
     1994,
     1990,
     1989,
     1991,
     1978,
     1973,
     1990,
     1990,
     1994,
     1992,
     1989,
     1987,
     1984,
     1988,
     1984,
     1982,
     1982,
     1989,
     1983,
     1991,
     1990,
     1986,
     1983,
     1986,
     1987,
     1993,
     1982,
     1992,
     1993,
     1982,
     1984,
     1986,
     1981,
     1994,
     1983,
     1987,
     1983,
     1989,
     1988,
     1986,
     1993,
     1990,
     1988,
     1991,
     1990,
     1992,
     1982,
     1988,
     1984,
     1984,
     1989,
     1988,
     1992,
     1991,
     1983,
     1983,
     1981,
     1985,
     1987,
     1983,
     1990,
     1988,
     1985,
     1989,
     1987,
     1994,
     1986,
     1985,
     1983,
     1993,
     1984,
     1994,
     1983,
     1986,
     1992,
     1986,
     1985,
     1994,
     1993,
     1990,
     1986,
     1987,
     1992,
     1990,
     1991,
     1994,
     1991,
     1992,
     2000,
     1987,
     1992,
     2001,
     2001,
     1997,
     1990,
     2000,
     1998,
     1994,
     1995,
     1990,
     1995,
     1991,
     1990,
     1990,
     1994,
     1990,
     1993,
     1999,
     2002,
     1999,
     1992,
     1995,
     1995,
     1986,
     1981,
     1981,
     1980,
     1990,
     1984,
     1993,
     1989,
     1991,
     1990,
     1995,
     1987,
     1991,
     1991,
     1991,
     1995,
     1993,
     1995,
     1992,
     1996,
     1994,
     1984,
     1993,
     1995,
     1994,
     1994,
     1992,
     1993,
     1991,
     1994,
     1992,
     1992,
     2001,
     1991,
     1997,
     1997,
     1998,
     1997,
     1997,
     1995,
     1996,
     2000,
     1992,
     1984,
     1992,
     1990,
     1990,
     1994,
     1992,
     1991,
     1999,
     1993,
     1994,
     1990,
     1991,
     1987,
     1988,
     1975,
     1991,
     1993,
     2000,
     1989,
     1988,
     1992,
     1981,
     2000,
     1987,
     1984,
     1993,
     2001,
     1989,
     1996,
     1999,
     1982,
     1990,
     1987,
     2001,
     1992,
     2002,
     1995,
     1991,
     1990,
     1983,
     1987,
     1993,
     1999,
     1985,
     1988,
     1987,
     1998,
     1982,
     1995,
     1990,
     1991,
     1991,
     1995,
     1984,
     1990,
     1991,
     1987,
     1990,
     1991,
     1995,
     1994,
     1994,
     1991,
     2002,
     2002,
     1995,
     1992,
     1987,
     1993,
     1997,
     1988,
     1993,
     1987,
     1991,
     1991,
     1990,
     1977,
     1992,
     2001,
     1992,
     1985,
     1993,
     1995,
     2003,
     1993,
     1976,
     1995,
     1997,
     1975,
     1999,
     1991,
     2000,
     1990,
     1971,
     1993,
     1993,
     1995,
     2001,
     1987,
     2001,
     1997,
     1991,
     1983,
     1991,
     1987,
     1993,
     1984,
     1990,
     1988,
     1986,
     1988,
     1990,
     1986,
     1992,
     1971,
     1991,
     1966,
     1990,
     1991,
     1989,
     1988,
     1985,
     1990,
     1970,
     1986,
     1990,
     1985,
     1995,
     1991,
     1981,
     1992,
     1993,
     1986,
     1989,
     1990,
     1973,
     1990,
     1989,
     1971,
     1992,
     1994,
     2001,
     1993,
     1993,
     1984,
     1984,
     1983,
     1988,
     2001,
     1989,
     1993,
     1985,
     1986,
     1988,
     1985,
     1995,
     1991,
     1990,
     1986,
     1992,
     1994,
     1985,
     1995,
     1976,
     1986,
     1987,
     1983,
     1989,
     1988,
     1995,
     1993,
     1992,
     1987,
     1991,
     1979,
     1987,
     1992,
     1991,
     1996,
     1981,
     2001,
     1992,
     1994,
     1981,
     1996,
     1989,
     1974,
     1993,
     1983,
     1984,
     1982,
     1983,
     1991,
     1987,
     1990,
     1991,
     1981,
     1987,
     1985,
     1985,
     1980,
     1987,
     1987,
     1984,
     1987,
     1989,
     1993,
     1971,
     1994,
     1988,
     1984,
     1973,
     1972,
     1991,
     1992,
     1975,
     1995,
     1995,
     1988,
     1991,
     1975,
     1997,
     1986,
     1988,
     1993,
     1988,
     1993,
     1994,
     1999,
     1986,
     1984,
     1987,
     1991,
     1991,
     1987,
     1984,
     1989,
     1989,
     1992,
     1990,
     1989,
     1992,
     1985,
     1987,
     1989,
     1985,
     1995,
     1996,
     1983,
     1995,
     1988,
     1990,
     1994,
     1991,
     1988,
     1993,
     1982,
     1977,
     1993,
     1997,
     1985,
     1996,
     1990,
     1986,
     1983,
     1994,
     1985,
     1990,
     1985,
     1986,
     1985,
     1985,
     1984,
     1995,
     1986,
     1993,
     1988,
     1988,
     1984,
     1994,
     1996,
     1994,
     1988,
     1987,
     1990,
     1987,
     1991,
     1993,
     1993,
     1987,
     1995,
     1986,
     1975,
     1991,
     1987,
     1989,
     1988,
     1997,
     1984,
     1988,
     1982,
     1988,
     1985,
     1998,
     1994,
     1990,
     1998,
     1996,
     1982,
     1985,
     1989,
     1989,
     1983,
     1990,
     1987,
     1994,
     1991,
     1985,
     1990,
     1982,
     1984,
     1994,
     1991,
     1996,
     1995,
     1989,
     1990,
     1990,
     1988,
     1991,
     1982,
     1985,
     1989,
     1970,
     1982,
     1996,
     1982,
     1991,
     1989,
     1992,
     1971,
     1997,
     1981,
     1985,
     1988,
     1986,
     1991,
     1990,
     1991,
     1986,
     1986,
     1985,
     2003,
     1981,
     1994,
     1991,
     1987,
     1993,
     1996,
     1990,
     1993,
     1987,
     1992,
     1988,
     1991,
     1986,
     1992,
     1998,
     1997,
     1998,
     1986,
     1995,
     1989,
     1992,
     1985,
     1996,
     1994,
     1992,
     1986,
     1994,
     1999,
     1990,
     1987,
     1986,
     1986,
     1992,
     1994,
     1986,
     1994,
     1997,
     1996,
     1993,
     1990,
     1992,
     1995,
     1991,
     1989,
     1985,
     1989,
     1982,
     1992,
     1989,
     1994,
     1993,
     1989,
     1996,
     1987,
     1992,
     1984,
     1986,
     1983,
     1985,
     1986,
     1986,
     1984,
     1988,
     1981,
     1987,
     1994,
     1996,
     1983,
     1984,
     1987,
     1991,
     1992,
     1990,
     1996,
     1991,
     1987,
     1990,
     1985,
     1983,
     1999,
     1995,
     1983,
     1991,
     1986,
     1972,
     1969,
     1990,
     1988,
     1987,
     1985,
     1993,
     1994,
     1991,
     1995,
     1991,
     1974,
     1985,
     1990,
     1987,
     1978,
     1981,
     1988,
     1991,
     1989,
     1990,
     1995,
     1983,
     1989,
     1992,
     1984,
     1994,
     1993,
     1982,
     1986,
     1972,
     1985,
     1992,
     1983,
     1985,
     1988,
     1986,
     1985,
     1990,
     1988,
     1987,
     1987,
     1990,
     1986,
     1987,
     1988,
     1985,
     1991,
     1986,
     1984,
     1993,
     1987,
     1986,
     1988,
     1986,
     1989,
     1987,
     1992,
     1993,
     1995,
     1994,
     1999,
     1992,
     2000,
     1995,
     1991,
     1987,
     1994,
     1999,
     2001,
     1998,
     1993,
     1997,
     1991,
     1995,
     1999,
     2000,
     1994,
     1988,
     1992,
     1985,
     1996,
     2001,
     1994,
     1992,
     1994,
     1972,
     1993,
     1997,
     1986,
     2001,
     2000,
     2000,
     1992,
     1996,
     1992,
     1969,
     1993,
     1996,
     1986,
     1991,
     2001,
     1990,
     1995,
     1989,
     2002,
     1994,
     1997,
     1988,
     2000,
     1986,
     2002,
     1998,
     1984,
     1994,
     1987,
     2000,
     1995,
     1989,
     2000,
     2000,
     1992,
     1988,
     1970,
     1997,
     1990,
     1982,
     1992,
     1988,
     1988,
     1994,
     2000,
     1984,
     1973,
     1984,
     1997,
     1982,
     1986,
     1988,
     1986,
     1994,
     1992,
     1981,
     1987,
     1994,
     1989,
     1997,
     1997,
     1988,
     1983,
     1984,
     1984,
     1992,
     1989,
     1972,
     1987,
     1986,
     1992,
     1990,
     1972,
     1987,
     1972,
     1989,
     1993,
     1987,
     1989,
     1988,
     1993,
     1983,
     1993,
     1981,
     1996,
     1990,
     1988,
     1984,
     1987,
     1990,
     1988,
     1982,
     1988,
     1986,
     1985,
     1986,
     1984,
     1992,
     1986,
     1990,
     1988,
     1996,
     1987,
     1993,
     1990,
     1983,
     1988,
     1989,
     1981,
     1992,
     1987,
     1988,
     1992,
     1999,
     1989,
     1985,
     1986,
     1987,
     1989,
     1989,
     1985,
     1987,
     1985,
     1996,
     1987,
     1986,
     1980,
     1987,
     1993,
     1985,
     1995,
     1989,
     1979,
     1987,
     1986,
     1991,
     1994,
     1988,
     1993,
     1992,
     1987,
     1988,
     1985,
     1990,
     1985,
     1985,
     1994,
     1995,
     1991,
     1995,
     1990,
     1989,
     1984,
     1988,
     1985,
     1995,
     1999,
     1997,
     1986,
     1991,
     1992,
     1982,
     1989,
     1985,
     2000,
     1999,
     1995,
     1992,
     1990,
     1990,
     1995,
     1989,
     1998,
     1986,
     1990,
     1994,
     1987,
     1983,
     1993,
     1999,
     1992,
     1998,
     1991,
     1992,
     1994,
     1993,
     1990,
     1985,
     1988,
     1988,
     1978,
     1995,
     1994,
     1994,
     1992,
     1988,
     1996,
     1978,
     1996,
     1983,
     1972,
     1994,
     1997,
     1998,
     1976,
     1973,
     1986,
     1968,
     1997,
     1991,
     1992,
     1999,
     1995,
     1993,
     1994,
     1992,
     1991,
     1998,
     1995,
     2000,
     1998,
     1986,
     1997,
     1999,
     2001,
     1996,
     2001,
     2000,
     1989,
     1993,
     1992,
     1992,
     1990,
     ...]




```python
type(new)
```




    list




```python
np.savetxt("test_result2.txt", new, delimiter="\n")
```


```python
tryRead = pd.read_csv("test_result2.txt",header=-1,)
tryRead.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1992.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1984.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1978.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1977.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985.0</td>
    </tr>
  </tbody>
</table>
</div>


