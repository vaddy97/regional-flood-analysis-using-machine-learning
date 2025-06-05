import timeAdd commentMore actions
start_time = time.time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"E:\Hdrology\FINAL_DATA\AAAAAA\Data\LOG-DATA.csv")
#data = pd.read_csv(r"E:\Hdrology\FINAL_DATA\AAAAAA\Data\InputData-CSV.csv")
#x=data.loc[:,['Area','Slope','ForestFrac','WaterStorage','Urban', 'Rainfall']]


############################# Most related paramaters ########################################
x=data.loc[:,['Watershed Area (km2)','Mean Elevation of the Watershed (m)','Surface Water Storage Area (km2)',
                'Urban Area (km2)', 'Mean Annual Rainfall (mm)', 'Mean Maximum Temperature (K/day)','Mean Solar Radiation (MJ/m2)','Gravelius_Index']]
 
#x=data.loc[:,['Watershed Area (km2)','Slope','Mean Elevation of the Watershed (m)','Fraction of Forest Cover','Surface Water Storage Area (km2)','Urban Area (km2)','Mean Annual Rainfall (mm)',
 #               'Mean Maximum Temperature (K/day)','Mean Minimum Temperature (K/day)','Mean Solar Radiation (MJ/m2)','Mean Humidity','Mean Wind Speed (m/s)','Perimeter (km)','Gravelius_Index','Soil Depth (m)']]
y=data.loc[:,'Q100']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


##################### currently best result for log transformed data #################################
x_train = x.iloc[[1,2,3,4,6,7,8,9,10,11,13,16,17,19,21,22,23,24,26,27,28,29,30,31,32,34,36,37,38]]
x_test = x.iloc[[0,5,12,14,15,18,20,25,33,35]]
y_train = y.iloc[[1,2,3,4,6,7,8,9,10,11,13,16,17,19,21,22,23,24,26,27,28,29,30,31,32,34,36,37,38]]
y_test = y.iloc[[0,5,12,14,15,18,20,25,33,35]]

'''
################################ best accuracy in (case of RFR) on real data ###################################
x_train = x.iloc[[0,2,3,4,5,7,9,10,11,12,13,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,37,38]]
x_test = x.iloc[[1,6,8,14,18,20,27,31,33,35]]
y_train = y.iloc[[0,2,3,4,5,7,9,10,11,12,13,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,37,38]]
y_test = y.iloc[[1,6,8,14,18,20,27,31,33,35]]'''

'''
############## best accuracy in case of SVM #########################################
x_train = x.iloc[[0,1,2,3,4,5,7,8,9,10,1,13,14,15,16,20,22,23,24,25,26,27,28,29,30,31,32,33,34,36]]
x_test = x.iloc[[6,9,11,17,18,19,21,35,37,38]]
y_train = y.iloc[[0,1,2,3,4,5,7,8,9,10,1,13,14,15,16,20,22,23,24,25,26,27,28,29,30,31,32,33,34,36]]
y_test = y.iloc[[6,9,11,17,18,19,21,35,37,38]]'''

'''#########################  not currently use it ###########################################
x_train = x.iloc[[0,1,2,5,6,7,8,10,11,12,13,15,16,17,19,20,21,22,23,24,25,28,29,30,32,33,34,35,37]]
x_test = x.iloc[[3,4,9,14,18,26,27,31,36,38]]
y_train = y.iloc[[0,1,2,5,6,7,8,10,11,12,13,15,16,17,19,20,21,22,23,24,25,28,29,30,32,33,34,35,37]]
y_test = y.iloc[[3,4,9,14,18,26,27,31,36,38]]'''

# Fitting Random Forest Regression to the dataset 
from sklearn.ensemble import RandomForestRegressor
'''
####################   code for finding best parameters   #####################
rf=RandomForestRegressor()
# create regressor object
from sklearn.model_selection import RandomizedSearchCV

n_estimators= [ 200,300,400,500, 600]
bootstrap= [True, False]
max_depth= [30, 50, 70 ]
max_features= ['auto', 'sqrt']
min_samples_leaf= [1, 2, 3]
min_samples_split= [2, 3, 4, 5, 6]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                n_iter = 10, cv =3,verbose=2,n_jobs = -1, random_state=42)

# code for finding the best parameters
rf_random.fit(x_train.values, y_train.values.ravel())
print(rf_random.best_params_)
print("Best score is {}".format(rf_random.best_score_))

#Automatic selection of best hyperparameters
BestP = rf_random.best_params_
regressor = RandomForestRegressor(bootstrap=BestP['bootstrap'],
                                  max_features=BestP['max_features'],
                                  min_samples_leaf=BestP['min_samples_leaf'],
                                  min_samples_split=BestP['min_samples_split'],
                                  n_estimators=BestP['n_estimators'],                                 
                                  max_depth=BestP['max_depth'])'''


regressor = RandomForestRegressor(bootstrap=True,
                                  max_features='auto',
                                  min_samples_leaf=2,
                                  min_samples_split=4,
                                  n_estimators=600,
                                  random_state=20,                                  
                                  max_depth=30)

# fit the regressor with x and y data
regressor.fit(x_train.values, y_train.values.ravel())

y_pred = regressor.predict(x_test)
y_pred1 = regressor.predict(x_train)

Feature = pd.DataFrame({'Feature_names':x.columns,'Importances':regressor.feature_importances_})
print(Feature)


x_plot = y_test
y_plot = y_pred
fig, ax = plt.subplots()
_ = ax.scatter(x_plot, y_plot, c=x_plot, cmap='plasma')
z = np.polyfit(x_plot, y_plot, 1)
p = np.poly1d(z)
ax.set_xlabel('y_test',fontsize=14)
ax.set_ylabel('y_pred',fontsize=14)
plt.plot(x_plot, p(x_plot))
plt.show()


from sklearn import metrics
print('Test MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('Test MSE:', metrics.mean_squared_error(y_test, y_pred))
print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Test R2:', metrics.r2_score(y_test, y_pred) )

from sklearn import metrics
print('Train MAE:', metrics.mean_absolute_error(y_train, y_pred1))
print('Train MSE:', metrics.mean_squared_error(y_train, y_pred1))
print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred1)))
print('Train R2:', metrics.r2_score(y_train, y_pred1))

stop_time = time.time()
print("Simulation completed in %.2fs" % (stop_time - start_time))
Add comment
