from django.shortcuts import render

import pandas as pd
import numpy as np
import random
import copy
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def index(req):
    return render(req, 'index.html')

def predict(request):
    df = pd.read_csv("C:\\Users\\visa\\Desktop\\Personality-Prediction-System-master\\Personality-Prediction-System-master\\train dataset.csv")
    # print(df)
    df1 = pd.read_csv("C:\\Users\\visa\\Desktop\\Personality-Prediction-System-master\\Personality-Prediction-System-master\\test.csv")
    print(df1.head())
    dataset=[]
    gender = df['Gender']
    age = df['Age']
    openness = df['openness']
    neuroticism = df['neuroticism']
    conscientiousness = df['conscientiousness']
    agreeableness = df['agreeableness']
    extraversion = df['extraversion']
    label = df['Personality (Class label)']

    dataset1=[]
    gender1 = df1['Gender']
    age1 = df1['Age']
    openness1 = df1['openness']
    neuroticism1= df1['neuroticism']
    conscientiousness1 = df1['conscientiousness']
    agreeableness1 = df1['agreeableness']
    extraversion1 = df1['extraversion']
    label1 = df1['Personality (Class label)']

    test_length = len(gender1)
    print(test_length)

    gender = gender.tolist() + gender1.tolist()
    age = age.tolist() + age1.tolist()
    openness = openness.tolist() + openness1.tolist()
    neuroticism = neuroticism.tolist() + neuroticism1.tolist()
    conscientiousness = conscientiousness.tolist() + conscientiousness1.tolist()
    agreeableness = agreeableness.tolist() + agreeableness1.tolist()
    extraversion = extraversion.tolist() + extraversion1.tolist()
    label = label.tolist() + label1.tolist()

    label = label
    list(set(label))

    length_label = len(label)
    len(label)

    def filter(label,gender,age,openness,neuroticism,conscientiousness,agreeableness,extraversion):
        responsible_list= []
        serious_list=[]
        extraverted_list=[]
        lively_list=[]
        dependable_list=[]
        for i in range(len(label)):
            if label[i]=="responsible":
                temp=[]
                temp.append(gender[i])
                temp.append(age[i])
                temp.append(openness[i])
                temp.append(neuroticism[i])
                temp.append(conscientiousness[i])
                temp.append(agreeableness[i])
                temp.append(extraversion[i])
                responsible_list.append(temp)
            if label[i]=="serious":
                temp=[]
                temp.append(gender[i])
                temp.append(age[i])
                temp.append(openness[i])
                temp.append(neuroticism[i])
                temp.append(conscientiousness[i])
                temp.append(agreeableness[i])
                temp.append(extraversion[i])
                serious_list.append(temp)
            if label[i]=="extraverted":
                temp=[]
                temp.append(gender[i])
                temp.append(age[i])
                temp.append(openness[i])
                temp.append(neuroticism[i])
                temp.append(conscientiousness[i])
                temp.append(agreeableness[i])
                temp.append(extraversion[i])
                extraverted_list.append(temp)
            if label[i]=="lively":
                temp=[]
                temp.append(gender[i])
                temp.append(age[i])
                temp.append(openness[i])
                temp.append(neuroticism[i])
                temp.append(conscientiousness[i])
                temp.append(agreeableness[i])
                temp.append(extraversion[i])
                lively_list.append(temp)
            if label[i]=="dependable":
                temp=[]
                temp.append(gender[i])
                temp.append(age[i])
                temp.append(openness[i])
                temp.append(neuroticism[i])
                temp.append(conscientiousness[i])
                temp.append(agreeableness[i])
                temp.append(extraversion[i])
                dependable_list.append(temp)

        return responsible_list,serious_list,extraverted_list,lively_list,dependable_list
    
    responsible_list,serious_list,extraverted_list,lively_list,dependable_list = filter(label,gender,age,openness,neuroticism,conscientiousness,agreeableness,extraversion)

    dependable_list.append(['Female', 18, 7, 6, 4, 5, 5])

    print(len(responsible_list),len(serious_list),len(extraverted_list),
      len(lively_list),len(dependable_list))

    def random_selection(source_num,label,list_label,target_num):
        label = [i for i in range(source_num)]
        random_label = random.sample(label,target_num)
        
        list_label = [list_label[random_label[i]] for i in range(len(random_label))]
        return list_label

    ###No need to do for responsible class
    # Serious label
    serious_list_main = random_selection(len(serious_list),label,serious_list,len(responsible_list))
    #Extraverted list

    extraverted_list_main = random_selection(len(extraverted_list),label,extraverted_list,len(responsible_list))

    ##Lively list

    lively_list_main = random_selection(len(lively_list),label,lively_list,len(responsible_list))

    ##Dependable List
    dependable_list_main = random_selection(len(dependable_list),label,dependable_list,len(responsible_list))
    responsible_list_main = copy.deepcopy(responsible_list)                                            

    #Converting Male =0 and Female =1
    def converter(data_point):
        temp=[]
        if data_point[0]=='Male':
            temp.append(0)
        else:
            temp.append(1)
        
        data = data_point[1:]
        temp = temp+data
        return temp
    

    def converter_data(data_list):
        temp=[]
        for i in range(len(data_list)):
            temp.append(converter(data_list[i]))
        return temp
        
    converter(['Male', 20, 1, 2, 7, 6, 4])

    responsible_list_main = converter_data(responsible_list_main)

    serious_list_main = converter_data(serious_list_main)
    lively_list_main = converter_data(lively_list_main)
    dependable_list_main = converter_data(dependable_list_main)
    extraverted_list_main = converter_data(extraverted_list_main)

    #resposible =0 extra 1 serious 2 lively 3 dependable 4 
    dataset_main=[]
    for i in responsible_list_main:
        dataset_main.append(i + [0])

    for i in extraverted_list_main:
        
        dataset_main.append(i + [1])
    for i in serious_list_main:
        
        dataset_main.append(i + [2])
    for i in lively_list_main:
        
        dataset_main.append(i + [3])
    for i in dependable_list_main:
        
        dataset_main.append(i + [4])

    #training the dataset_main
    dataset_train = random.sample(dataset_main,len(dataset_main))
    Normalised_attrs = []
    for i in range(len(dataset_train)):
        Normalised_attrs.append(dataset_train[i][1:7])


    scaler = StandardScaler()
    Normalised_attrs = scaler.fit_transform(Normalised_attrs)

    temp=[]
    for i in range(len(Normalised_attrs)):
        temp.append(list([dataset_train[i][0]])+list(Normalised_attrs[i])+list([dataset_train[i][-1]]))

    universal_data = copy.deepcopy(temp)

    data_points=[]
    data_targets=[]
    for i in universal_data:
        data_points.append(i[:7])
        data_targets.append(i[7])

    train = data_points[:709]
    label_train = data_targets[:709]
    test = data_points[709:]
    label_test = data_targets[709:]

    clf = svm.SVC(kernel='rbf', C=1)

    train=np.array(train)
    label_train = np.array(label_train)

    model  = clf.fit(train,label_train)
    label_set  = list(set(label))

    personality={}
    for i in range(5):
        personality[i]=label_set[i]

    predict_label = model.predict(test)
    model.score(test,label_test)*100
    data_point=[request.POST['Gender'], request.POST['Age'], request.POST['Openness'], request.POST['Neuroticism'], request.POST['Conscientiousness'], request.POST['Agreeableness'], request.POST['Extraversion']]

    def predict_datapoint(model,datapoint):
        temp=[]
        if datapoint[0]=="Male":
            temp.append(0)
        else:
            temp.append(1)
        data = scaler.transform([datapoint[1:]])
        temp = temp + data.reshape(data.shape[1]).tolist()
    #     print(temp)
        label = model.predict(np.array([temp]))
        return label[0]

    finaloutput = personality[predict_datapoint(model, data_point)]
    
    return render(request, 'output.html',{'output' : finaloutput})