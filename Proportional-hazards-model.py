import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from lifelines import CoxPHFitter
import statsmodels.api as sm

#Read Data
data = pd.read_csv("ProjectSchema.csv", encoding= 'unicode_escape')

#Convert Status variable to dummies
data=pd.get_dummies(data, columns=['Status'])
data=data.drop(['Status_H'], axis=1)

dat=data.iloc[:, 1: 45]

#The dummy variable, Status_F, is added as the last column in the dataframe (column 44)
dat.drop(dat.iloc[:, 10:43], inplace = True, axis = 1)

print("dat_colname",dat.columns)

#Label the Target Variable
# cleanup_nums = {"Status": {"H": 0, "F": 1}}
# dat.replace(cleanup_nums, inplace=True)

#Columns data types
dataTypeSeries = dat.dtypes
print('type', dataTypeSeries)

# print("NA-Count=",dat.isnull().sum())

#Select Important Features with_Random_Forest
X=dat
X = X.drop(['Status_F','Month'], axis=1)
y = dat['Status_F']
print("X.shape=",X.shape)
print("y.shape=", y.shape)

RF= RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
)

#3-fold Cross Validation
cv = cross_validate(RF, X, y, cv=3, return_estimator =True)
print(cv['test_score'])
print(cv['test_score'].mean())

#Feature Importance
for idx,estimator in enumerate(cv['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index = X.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)

#Fit Logistic Regression
model = sm.OLS(y, X)
result = model.fit()
print("regression results:\n", result.summary())


#Using Cox Proportional Hazards model
df=dat[['Month', 'Status_F', 'Age(year)', 'Max.rotor.speed,RPM-Avg', 'active power-Avg']]
print("df=",df )

#Survival Analysis_ Cox Proportional Hazard Model
cph = CoxPHFitter()   ## Instantiate the class to create a cph object
cph.fit(df , 'Month', 'Status_F')   ## Fit the data to train the model
cph.print_summary()    ## HAve a look at the significance of the features
cph.plot()

#Survival curves for the selected Turbines
tr_rows = df.iloc[19:27, 2:]
cph.predict_survival_function(tr_rows).plot()
plt.show()
