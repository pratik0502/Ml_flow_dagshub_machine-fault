import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import dagshub
dagshub.init(repo_owner='pratik0502', repo_name='Ml_flow_dagshub_machine-fault', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/pratik0502/Ml_flow_dagshub_machine-fault.mlflow')
# mlflow.set_tracking_uri('http://ec2-16-170-236-218.eu-north-1.compute.amazonaws.com:5000/')



import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

data = pd.read_csv(r"D:\ML_ops\archive\data.csv")

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

sd = StandardScaler()
x = sd.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
 




# gm = 1.5
# c = 0.1
# kernel = 'poly'



model = SVC()

params_grid ={
  'gm':[0.1,1,1.5,2],
  'C':[0.001,0.001,0.1,1],
  'kernel':['poly','rbf']
  }

grid_serach = GridSearchCV(estimator=model,param_grid=params_grid,cv=5,n_jobs=5,verbose=-1)

mlflow.set_experiment('machine_fault_SVC')

with mlflow.start_run(run_name='svc with kern = poly'):

  grid_serach.fit(x_train,y_train)
  
  best_params = grid_serach.best_params_
  best_score = grid_serach.best_score_

  mlflow.log_param(best_params)
  mlflow.log_metric('accuracy',best_score)
   # to log the data
  train_data = x_train
  train_data['target'] = y_train.iloc[:,-1]

  mlflow.data.from_pandas(train_data,'train_data')

  test_data = x_test
  test_data = y_test.iloc[:,0]

  mlflow.data.from_pandas(test_data,'test_data')

  
  y_pred = model.predict(x_test)

  acc = accuracy_score(y_test,y_pred)

  CM = confusion_matrix(y_test,y_pred)

  fig, ax = plt.subplots()
  ax.matshow(CM, cmap=plt.cm.Blues)
  for (i, j), val in np.ndenumerate(CM):
      ax.text(j, i, f'{val}', ha='center', va='center')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.savefig('confusion_matrix.png')
  plt.close()

  CR = classification_report(y_test,y_pred)

  mlflow.log_metric('accuracy',acc)
  # mlflow.log_metric('Confusion_metrics',CM)
  # mlflow.log_metric('Confusion_report',CR)

  mlflow.sklearn.log_model(grid_serach.best_estimator_,'SVC')

  mlflow.log_artifact('confusion_matrix.png')

  mlflow.log_artifact(__file__)

  mlflow.set_tag('author','pratik')


  print('accuracy',acc)

  print('CM',CM) 

  print('CR',CR)


