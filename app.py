from flask.helpers import send_file
from jinja2 import Template
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)





# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
app.config['DFPr'] = pathfordataset
app.config['dataset_name_to_use_for_preprocessing'] = ''





# this is the path to save dataset for single variable LR
pathforonevarLR = "static/Regression/onevarLR"
pathforonevarLRplot = "Regression/onevarLR/plot"
app.config['LR1VAR'] = pathforonevarLR
app.config['LR1VARplot'] = pathforonevarLRplot





# this is the path to save dataset for Decision Tree
pathfordecisiontree = "static/Classification/decisiontree"
pathfordecisiontreeplot = "Classification/decisiontree/plot"
app.config['decisiontreedata'] = pathfordecisiontree
app.config['decisiontreeplot'] = pathfordecisiontreeplot
app.config['dataset_name_to_use_for_DT'] = ''





# this is the path to save dataset for Decision Tree
pathforknn = "static/Classification/knn"
pathforknnplot = "Classification/knn/plot"
app.config['knndata'] = pathforknn
app.config['knnplot'] = pathforknnplot
app.config['dataset_name_to_use_for_knn'] = ''


@app.route('/')
def index():
    return render_template('index.html')


# for data preprocessing

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing/preprocessing.html')



@app.route('/preprocessing/dataadd' , methods = ['GET','POST'])
def dataadd():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        app.config['dataset_name_to_use_for_preprocessing'] = my_dataset.filename
        dataset_path = os.path.join(pathfordataset,secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))           
        df = pd.read_csv(get_dataset)

        
    return render_template('/preprocessing/preprocessing.html'
    ,col = df.columns.tolist()
    )

@app.route('/preprocessing/preprocessing' , methods = ['GET','POST'])
def uploadpreprocess():
    if request.method == 'POST':
        get_dataset = os.path.join(app.config['DFPr'],secure_filename(app.config['dataset_name_to_use_for_preprocessing']))
        feature = request.form.getlist('features')
        labelencode=request.form.getlist('labelencode')
        df = pd.read_csv(get_dataset)
        df = df.fillna(method = 'ffill')
        sc = StandardScaler()
        if len(feature):
            df[feature] = sc.fit_transform(df[feature])
        
        labelkrnewala = LabelEncoder()
        if len(labelencode):
            for item in labelencode:
                df[item] = labelkrnewala.fit_transform(df[item])

        trained_dataset = pd.DataFrame(df)

        trained_dataset.to_csv("static/data-preprocess/new/trained_dataset.csv")

        return render_template('/preprocessing/preprocessing_output.html', data_shape=trained_dataset.shape, table=trained_dataset.head(5).to_html(classes='table table-striped table-dark table-hover x'), dataset_describe=trained_dataset.describe().to_html(classes='table table-striped table-dark table-hover x') )



@app.route('/downloadNewDataset')
def download_file():
    path = "static/data-preprocess/new/trained_dataset.csv"
    return send_file(path,as_attachment=True)


@app.route('/supervised')
def supervised():
    return render_template('supervised/home-supervised.html')  
@app.route('/unsupervised')
def unsupervised():
    return render_template('unsupervised/home-unsupervised.html') 
    
@app.route('/supervised/regression/linearregression')
def regressionLR():
    return render_template('/supervised/regression/linearregression.html') 


@app.route('/supervised/regression/linearregression' ,  methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']        
        dataset_path = os.path.join(pathforonevarLR,secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'],secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method = 'ffill')
        li = list(df.columns)
        x= df.iloc[:, :-1]
        y= df.iloc[:, 1]
        # Splitting the dataset into training and test set.  
        x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3 , random_state = 42)    
        #Fitting the Simple Linear Regression model to the training dataset
        regressor= LinearRegression() 
        regressor.fit(x_train, y_train)
        the_intercept = regressor.intercept_
        the_slope = regressor.coef_
        trainingscore = regressor.score(x_train, y_train)
        testingscore = regressor.score(x_test, y_test)
        # Visualising the  results
        plt.scatter(x, y, color = 'red')
        plt.plot(x, regressor.predict(x), color = 'blue')
        plt.title('{} VS {}'.format(li[0], li[1]))
        plt.xlabel('{}'.format(li[0]))
        plt.ylabel('{}'.format(li[1]))
        fig = plt.gcf()
        img_name = 'data'
        fig.savefig('static/Regression/onevarLR/plot/data.png', dpi = 1500)
        get_plot1 = os.path.join(app.config['LR1VARplot'],'%s.png'%img_name)
        plt.clf()
        return render_template('/supervised/regression/output1varLR.html' , model_name = my_model_name,
         var1 = the_intercept , var2 = the_slope ,  visualize = get_plot1 ,data_shape = df.shape,
         table = df.head(5).to_html(classes='table table-striped table-dark table-hover x'),
         dataset_describe = df.describe().to_html(classes='table table-striped table-dark table-hover x'),
            trainingscore = trainingscore,testingscore = testingscore)



@app.route('/supervised/regression/linearregressionwithMVAR')
def multiregressionLR():
    return render_template('/supervised/regression/linearregressionwithMVAR.html')





@app.route('/supervised/classification/decisiontree')
def decisiontree():
    return render_template('/supervised/classification/decisiontree.html')


@app.route('/supervised/classification/decisiontree/data' , methods = ['GET','POST'])
def decisiontreedataset():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        app.config['dataset_name_to_use_for_DT'] = my_dataset.filename
        dataset_path = os.path.join(pathfordecisiontree,secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['decisiontreedata'],secure_filename(my_dataset.filename))           
        df = pd.read_csv(get_dataset)
        missing  = df.isnull().sum().to_frame('Individual count of missing value') 
        
    return render_template('/supervised/classification/decisiontree.html',
    my_dataset_head = df.head(5).to_html(classes = 'table table-striped table-hover x'), 
    missing = missing.to_html(classes='table table-striped table-hover x')
    ,dataset_describe = df.describe().to_html(classes='table table-striped table-hover x'),col = df.columns.tolist()
    
    )


@app.route('/supervised/classification/decisiontree/train' , methods = ['GET','POST'])
def dttrain():
    if request.method == 'POST':
        get_dataset = os.path.join(app.config['decisiontreedata'],secure_filename(app.config['dataset_name_to_use_for_DT']))
        feature = request.form.getlist('features')
        labelencode = request.form.getlist('label-encoding')
        predictlabel = request.form['predict-label']
        criteria = request.form['criteria']
        name_of_model=request.form.get('name_of_model')
        df = pd.read_csv(get_dataset)
        df = df.fillna(method = 'ffill')
        labelkrnewala = LabelEncoder()
        for item in labelencode:
            df[item] = labelkrnewala.fit_transform(df[item])
        x_feature = pd.DataFrame()
        for item in feature:
            x_feature.insert(0,item,df[item],allow_duplicates = False)
        y = df[predictlabel]
        sc = StandardScaler()
        x_feature = sc.fit_transform(x_feature)
        x_train, x_test, y_train, y_test = train_test_split(x_feature, y, test_size = 0.25, random_state = 5)
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = criteria, random_state = 43)
        classifier.fit(x_train, y_train)
        #this will show accuracy on training data
        result = classifier.score(x_train,y_train)
        #accuracy on test data
        accuracytestdata = classifier.score(x_test,y_test)
        #changing max depth (longest path of the tree)
        train_accuracy=[]
        valid_accuracy=[]
        for depth in range (1,10):
            classifier=DecisionTreeClassifier(max_depth=depth, random_state=10)
            classifier.fit(x_train, y_train)
            train_accuracy.append(classifier.score(x_train,y_train))
            valid_accuracy.append(classifier.score(x_test,y_test))
        frame=pd.DataFrame({'max_depth':range(1,10), 'train_acc':train_accuracy, 'test_acc':valid_accuracy})
        frame.head(9)
        
        plt.figure(figsize=(12,6))
        plt.plot(frame['max_depth'], frame['train_acc'], marker='o')
        plt.plot(frame['max_depth'], frame['test_acc'], marker='o')
        plt.xlabel('depth of tree')
        plt.ylabel('performance')
        fig = plt.gcf()
        img_name = 'decisiontree'
        fig.savefig('static/Classification/decisiontree/plot/decisiontree.png', dpi = 1500)
        get_plot1 = os.path.join(app.config['decisiontreeplot'],'%s.png'%img_name)
        
        plt.clf()
        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        



        asbetweentestandpred = accuracy_score(y_test,y_pred)*100
        cm = confusion_matrix(y_test, y_pred)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        score = f1_score(y_test, y_pred, average='weighted')

        return render_template('/supervised/classification/decisiontreeoutput.html' , name_of_model = name_of_model,
        result = result, accuracytestdata = accuracytestdata,visualize = get_plot1,
        frame = frame.head(9).to_html(classes='table table-striped table-dark table-hover x'),asbetweentestandpred = asbetweentestandpred,
        cm = cm,precision = precision,score = score,recall = recall
        
        )













@app.route('/supervised/classification/knn')
def knn():
    return render_template('/supervised/classification/knn.html')

@app.route('/supervised/classification/knn/data' ,  methods = ['GET','POST'] )
def knndata():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        app.config['dataset_name_to_use_for_knn'] = my_dataset.filename
        dataset_path = os.path.join(pathforknn,secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['knndata'],secure_filename(my_dataset.filename))           
        df = pd.read_csv(get_dataset)
        missing  = df.isnull().sum().to_frame('Individual count of missing value') 
        
    return render_template('/supervised/classification/knn.html',
    my_dataset_head = df.head(5).to_html(classes = 'table table-striped table-hover x'), 
    missing = missing.to_html(classes='table table-striped table-hover x')
    ,dataset_describe = df.describe().to_html(classes='table table-striped table-hover x'),col = df.columns.tolist()
    
    )



@app.route('/supervised/classification/knn/train' , methods = ['GET','POST'])
def knntrain():
    if request.method == 'POST':
        get_dataset = os.path.join(app.config['knndata'],secure_filename(app.config['dataset_name_to_use_for_knn']))
        feature = request.form.getlist('features')
        labelencode = request.form.getlist('label-encoding')
        predictlabel = request.form['predict-label']
        name_of_model=request.form.get('name_of_model')
        df = pd.read_csv(get_dataset)
        df = df.fillna(method = 'ffill')
        labelkrnewala = LabelEncoder()
        for item in labelencode:
            df[item] = labelkrnewala.fit_transform(df[item])
        x_feature = pd.DataFrame()
        for item in feature:
            x_feature.insert(0,item,df[item],allow_duplicates = False)
        y = df[predictlabel]
        sc = StandardScaler()
        x_feature = sc.fit_transform(x_feature)
        x_train, x_test, y_train, y_test = train_test_split(x_feature, y, test_size = 0.25, random_state = 5)
        from sklearn.neighbors import KNeighborsClassifier
        classifier =  KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(x_train, y_train)
        #this will show accuracy on training data
        result = classifier.score(x_train,y_train)
        #accuracy on test data
        accuracytestdata = classifier.score(x_test,y_test)
        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        asbetweentestandpred = (accuracy_score(y_test, y_pred)*100)
        from timeit import default_timer as timer
        start = timer()
        classifier.fit(x_train, y_train)
        end = timer()
        timeelapsed = end- start
        cm = confusion_matrix(y_test, y_pred)
        return render_template('/supervised/classification/knnoutput.html' , name_of_model = name_of_model,result = result,
        accuracytestdata = accuracytestdata,timeelapsed = timeelapsed,cm=cm,asbetweentestandpred = asbetweentestandpred
        )
        


if __name__ == '__main__':
    app.run(debug=True)