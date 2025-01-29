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

# import dagshub
# dagshub.init(repo_owner='pratik0502', repo_name='Ml_flow_dagshub_machine-fault', mlflow=True)

# mlflow.set_tracking_uri('https://dagshub.com/pratik0502/Ml_flow_dagshub_machine-fault.mlflow')
mlflow.set_tracking_uri('http://ec2-16-170-236-218.eu-north-1.compute.amazonaws.com:5000/')



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
 

gm = 1.5
c = 0.1
kernel = 'poly'

mlflow.set_experiment('machine_fault_SVC')

with mlflow.start_run(run_name='svc with kern = poly'):

    model = SVC(gamma=gm,kernel=kernel,C=c)
    




    model.fit(x_train,y_train)

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

    mlflow.log_param('Gamma',gm)
    mlflow.log_param('C',c)
    mlflow.log_param('kernel',kernel)

    mlflow.sklearn.log_model(model,'Decision tree model')

    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_artifact(__file__)

    mlflow.set_tag('author','pratik')


    print('accuracy',acc)

    print('CM',CM) 

    print('CR',CR)


