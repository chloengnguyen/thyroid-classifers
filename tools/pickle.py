# First way to do this

model = linear_model.LinearRegression
model.fit(df[['area']], df.price)
model.coef_ 
model.intercept_
model.predict(5000)

#export 
import pickle 
with open('model_pickle', 'wb') as f:
    pickle.dump(model,f) #the trained model is saved into a binary file now 

#import 
with open('model_picke', 'rb') as f:
    mp = pickle.load(f)

mp.predict(5000) # this give the same prediction values 


# Sklearn joblib : more efficient for numpy array 

#export
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl')

#import 
clf = joblib.load('filename.pkl')