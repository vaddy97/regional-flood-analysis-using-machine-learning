import numpy as npAdd commentMore actions
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
data = pd.read_csv(r"E:\Hdrology\FINAL_DATA\AAAAAA\InputData-CSV.csv")
data.head(10)
#x=data.loc[:,['Area','Slope','ForestFrac','WaterStorage','Urban', 'Rainfall']]
#x=data.loc[:,['Watershed Area (km2)','Slope','Fraction of Forest Cover','Surface Water Storage Area (km2)','Urban Area (km2)', 'Mean Annual Rainfall (mm)']]

############################# Most related paramaters ########################################
x=data.loc[:,['Watershed Area (km2)','Mean Elevation of the Watershed (m)','Surface Water Storage Area (km2)',
                'Urban Area (km2)','Mean Annual Rainfall (mm)', 'Mean Maximum Temperature (K/day)','Mean Solar Radiation (MJ/m2)','Gravelius_Index']]
 
#x=data.loc[:,['Watershed Area (km2)','Slope','Mean Elevation of the Watershed (m)','Fraction of Forest Cover','Surface Water Storage Area (km2)','Urban Area (km2)','Mean Annual Rainfall (mm)',
 #               'Mean Maximum Temperature (K/day)','Mean Minimum Temperature (K/day)','Mean Solar Radiation (MJ/m2)','Mean Humidity','Mean Wind Speed (m/s)','Perimeter (km)','Gravelius_Index','Soil Depth (m)']]
y=data.loc[:,'Q100']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

############## best accuracy in case of RFR ######################################### 
x_train = x.iloc[[0,2,3,4,5,7,9,10,11,12,13,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,37,38]]
x_test = x.iloc[[1,6,8,14,18,20,27,31,33,35]]
y_train = y.iloc[[0,2,3,4,5,7,9,10,11,12,13,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,37,38]]
y_test = y.iloc[[1,6,8,14,18,20,27,31,33,35]]




from sklearn.svm import SVR

'''param={ 'kernel' : ['linear','poly','rbf','sigmoid'],
       'C' : [.1,.4,.6,1,2,4,6],}

svr_grid = GridSearchCV(SVR(),param_grid=param,verbose=2,n_jobs = -1)
svr_grid.fit(x_train,y_train)

print(svr_grid.best_params_)
print("Best score is {}".format(svr_grid.best_score_))

BestP = svr_grid.best_params_'''


svr=SVR(kernel = 'linear' ,C = 0.1 )
#svr=SVR(kernel = BestP['kernel'],C =BestP['C'])
svr.fit(x_train, y_train)
svr.score(x_test,y_test)
y_pred = svr.predict(x_test)
y_pred1= svr.predict(x_train)

##########################  Plotting the trend-line   ###############################
x_plot = y_test
y_plot = y_pred
fig, ax = plt.subplots()
_ = ax.scatter(x_plot, y_plot, c=x_plot, cmap='plasma')
z = np.polyfit(x_plot, y_plot, 1)
p = np.poly1d(z)
ax.set_xlabel('y_test',fontsize=15)
ax.set_ylabel('y_pred',fontsize=15)
plt.plot(x_plot, p(x_plot))
plt.show()

##################################################################################################
'''from matplotlib import pyplot as plt
from sklearn import svm

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    #plt.show()

features_names = ['Watershed Area (km2)','Slope','Mean Elevation of the Watershed (m)','Fraction of Forest Cover','Surface Water Storage Area (km2)', 'Urban Area (km2)','Perimeter (km)', 'Mean Annual Rainfall (mm)', 'Gravelius_Index','Soil Depth (m)']
#svr = svm.SVC(kernel='linear')
#svm.fit(x,y)
f_importances(svr.coef_, features_names)'''     
############################################################################################
model = SVR()
# fit the model
model.fit(x, y)
# perform permutation importance

from sklearn.inspection import permutation_importance
results = permutation_importance(model, x, y, scoring='neg_mean_squared_error')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %d, Score: %0.5f' % (i,v))
# plot feature importance
#plt.bar([x for x in range(len(importance))], importance)
#plt.show()

###################################################################################################


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred) )

print('Mean Absolute Error for predicted value :', metrics.mean_absolute_error(y_train, y_pred1))
print('Mean Squared Error for predicted value :', metrics.mean_squared_error(y_train, y_pred1))
print('Root Mean Squared Error for predicted value :', np.sqrt(metrics.mean_squared_error(y_train, y_pred1)))
print('R2 Score for predicted value :', metrics.r2_score(y_train, y_pred1) )
Add comment
