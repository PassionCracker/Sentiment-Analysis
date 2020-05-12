#### This cell contains the list of tasks
# Task - 1 : To read all files and to make a dictionary called - review_data
# Task - 2 : Splitting the data into 3-folds i.e., 2 folds in training_data and 1 fold for testing_data
# Task - 3 : Converting the data into 8 sets of given features
# Task - 3.0 : Negation tagging for unigrams
# Task - 3a : Unigrams frequncy as feautres
# Task - 3b : Unigrams presence as features
# Task - 3c : Bigrams, presence as features ( Note : Here , no need of negation tagging)
# Task - 3d : Unigrams+Bigrams, presence as features.
# Task - 3e : Unigrams with POS tagging, presence as features
# Task - 3f : Top 2633 Unigrams presence as features

import pandas as pd
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import random
from sklearn import metrics
import nltk
from nltk.tokenize import word_tokenize

review_data = {}
#0 : negative, 1 : positive
review_data[0] = []
review_data[1] = []

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
 
# load all docs in a directory
def process_docs(directory):
    vocab = {}
    index = 1
    for filename in listdir(directory):
        path = directory + '/' + filename
        doc_text = load_doc(path)
        if directory == 'neg':
            review_data[0].append(doc_text)
        else:
            review_data[1].append(doc_text)

process_docs('neg')
process_docs('pos')

#####task-1 done

# Length of each class data is 1000, making it to 3-fold => 333,333,333.
# So, 0-665 : training, 666-999 testing
training_data = {}
testing_data = {}
training_data[0] = []
training_data[1] = []
testing_data[0] = []
testing_data[1] = []
    
for i in range(1000):
    review_data[0][i] = review_data[0][i].replace(' \n',' ')
    if i<666:
        training_data[0].append(review_data[0][i])
        training_data[1].append(review_data[1][i])
    else:
        testing_data[0].append(review_data[0][i])
        testing_data[1].append(review_data[1][i])

#####task-2 done

negation_words = ['not','n\'t']
punctuations = ['.','?','!',',',':',';']
#Negation tagging of words immediatly from next word till punctuation mark
train_x1 = {}
test_x1 = {}
train_x1[0] = []
train_x1[1] = []
test_x1[0] = []
test_x1[1] = []

#for training data
for key in training_data:
    for review in training_data[key]:
        words_list = review.split()
        i=0
        while i<len(words_list):
            k=i
            if negation_words[0] in words_list[i] or negation_words[1] in words_list[i]:
                while words_list[k+1] not in punctuations:
                    words_list[k+1] = "NOT_"+words_list[k+1]
                    k+=1
                i=k+1
            else:
                i+=1
            
        #words_list is now updated
        if key == 0:
            train_x1[0].append(" ".join(words_list))
        else:
            train_x1[1].append(" ".join(words_list)) 
            
#For testing data            
for key in testing_data:
    for review in testing_data[key]:
        words_list = review.split()
        i=0
        while i<len(words_list):
            k=i
            if negation_words[0] in words_list[i] or negation_words[1] in words_list[i]:
                while k+1 < len(words_list) and words_list[k+1] not in punctuations :
                    words_list[k+1] = "NOT_"+words_list[k+1]
                    k+=1
                i=k+1
            else:
                i+=1
            
        #words_list is now updated
        if key == 0:
            test_x1[0].append(" ".join(words_list))
        else:
            test_x1[1].append(" ".join(words_list))             
#########Task - 3.0 done

###Determining "Unigram - frquency" feature based accuracy
cv1 = CountVectorizer(min_df=4)
x_train = train_x1[0]+train_x1[1]
x_test = test_x1[0]+test_x1[1]

y_train = []
y_test = []

for i in range(1332):
    if(i<666):
        y_train.append(0)
    else:
        y_train.append(1)
    if(i<334):
        y_test.append(0)
    elif(i<668):
        y_test.append(1)

train = list(range(1332))
test = list(range(668))
random.shuffle(train)
random.shuffle(test)
x1_train = []
x1_test = []
y1_train = []
y1_test = []
for i in train:
    x1_train.append(x_train[i])
    y1_train.append(y_train[i])
for i in test:
    x1_test.append(x_test[i])
    y1_test.append(y_test[i])

cv1.fit(x1_train)


x1_tra = cv1.transform(x1_train)
x1_tes = cv1.transform(x1_test)
nb1 = MultinomialNB()
lr1 = LogisticRegression(solver='lbfgs')
sv1 = svm.LinearSVC()
nb1.fit(x1_tra,y1_train)
lr1.fit(x1_tra,y1_train)
sv1.fit(x1_tra,y1_train)
y1_nb_pred = nb1.predict(x1_tes)
y1_lr_pred = lr1.predict(x1_tes)
y1_sv_pred = sv1.predict(x1_tes)
print("Unigrams Frequency as features-----------------")
print("F1 Score of Naive bayes for sentiment classifier is ",metrics.accuracy_score(y1_test,y1_nb_pred))
print("F1 Score of Logistic Regression for sentiment classifier is ",metrics.accuracy_score(y1_test,y1_lr_pred))
print("F1 Score of SVM for sentiment classifier is ",metrics.accuracy_score(y1_test,y1_sv_pred))

#########Task - 3a done

x2_train = []
x2_test = []
y2_train = y1_train
y2_test = y1_test

x2_train = x1_tra.toarray()
for i in range(len(x2_train)):
    for j in range(len(x2_train[i])):
        if x2_train[i][j] > 0:
            x2_train[i][j] = 1
x2_test = x1_tes.toarray()
for i in range(len(x2_test)):
    for j in range(len(x2_test[i])):
        if x2_test[i][j] > 0:
            x2_test[i][j] = 1

nb2 = MultinomialNB()
nb2.fit(x2_train,y2_train)
y2_nb_pred = nb2.predict(x2_test)
print("Unigrams presence as features-------------")
print("F1 Score of Naive bayes for sentiment classifier is ",metrics.accuracy_score(y2_test,y2_nb_pred))
lr2 = LogisticRegression(solver='lbfgs')
lr2.fit(x2_train,y2_train)
y2_lr_pred = lr2.predict(x2_test)
print("F1 Score of Logistic Regression for sentiment classifier is ",metrics.accuracy_score(y2_test,y2_lr_pred))
sv2 = svm.LinearSVC()
sv2.fit(x2_train,y2_train)
y2_sv_pred = sv2.predict(x2_test)
print("F1 Score of SVM for sentiment classifier is ",metrics.accuracy_score(y2_test,y2_sv_pred))

###########task - 3b done     

x_train = training_data[0] + training_data[1]
x_test = testing_data[0] + testing_data[1]
x3_train = []
x3_test = []
y3_train = []
y3_test = []
#shufflingggggggggggg
for i in train:
    x3_train.append(x_train[i])
    y3_train.append(y_train[i])
for i in test:
    x3_test.append(x_test[i])
    y3_test.append(y_test[i])


cv3 = CountVectorizer(ngram_range=(2,2))
cv3.fit(x3_train)
x3_train = cv3.transform(x3_train)
x3_test = cv3.transform(x3_test)

nb3 = MultinomialNB()
nb3.fit(x3_train,y3_train)
y3_nb_pred = nb3.predict(x3_test)
print("bigrams presence as featurs-----------")
print("F1 Score of Naive bayes for sentiment classifier is ",metrics.accuracy_score(y3_test,y3_nb_pred))
lr3 = LogisticRegression(solver='lbfgs')
lr3.fit(x3_train,y3_train)
y3_lr_pred = lr3.predict(x3_test)
print("F1 Score of Logistic Regression for sentiment classifier is ",metrics.accuracy_score(y3_test,y3_lr_pred))
sv3 = svm.LinearSVC()
sv3.fit(x3_train,y3_train)
y3_sv_pred = sv3.predict(x3_test)
print("F1 Score of SVM for sentiment classifier is ",metrics.accuracy_score(y3_test,y3_sv_pred))

###########task - 3c done     

x_train = training_data[0] + training_data[1]
x_test = testing_data[0] + testing_data[1]
x4_train = []
x4_test = []
y4_train = []
y4_test = []
#shufflingggggggggggg
for i in train:
    x4_train.append(x_train[i])
    y4_train.append(y_train[i])
for i in test:
    x4_test.append(x_test[i])
    y4_test.append(y_test[i])
cv4 = CountVectorizer(ngram_range=(1,2))
cv4.fit(x4_train)
x4_train = cv4.transform(x4_train)
x4_test = cv4.transform(x4_test)

nb4 = MultinomialNB()
nb4.fit(x4_train,y4_train)
y4_nb_pred = nb4.predict(x4_test)
print("Unigrams + Bigrams presence as features---------")
print("F1 Score of Naive bayes for sentiment classifier is ",metrics.accuracy_score(y4_test,y4_nb_pred))
lr4 = LogisticRegression(solver='lbfgs')
lr4.fit(x4_train,y4_train)
y4_lr_pred = lr4.predict(x4_test)
print("F1 Score of Logistic Regression for sentiment classifier is ",metrics.accuracy_score(y4_test,y4_lr_pred))
sv4 = svm.LinearSVC()
sv4.fit(x4_train,y4_train)
y4_sv_pred = sv4.predict(x4_test)
print("F1 Score of SVM for sentiment classifier is ",metrics.accuracy_score(y4_test,y4_sv_pred))

###########task - 3d done     


def postagger(str):
    string = ''
    tags_list = nltk.pos_tag(word_tokenize(str))
    for word_tag in tags_list:
        word = ''
        word=word+word_tag[0]+'_'+word_tag[1]
        string+=word+' '
    return string



x_train = training_data[0] + training_data[1]
x_test = testing_data[0] + testing_data[1]
x5_tra = []
x5_tes = []
y5_train = []
y5_test = []
#shufflingggggggggggg
for i in train:
    x5_tra.append(x_train[i])
    y5_train.append(y_train[i])
for i in test:
    x5_tes.append(x_test[i])
    y5_test.append(y_test[i])
x5_train = []
x5_test =[]
for item in x5_tra:
    x5_train.append(postagger(item))
for item in x5_tes:
    x5_test.append(postagger(item))

cv5=CountVectorizer()
cv5.fit(x5_train)
x5_train = cv5.transform(x5_train)
x5_test = cv5.transform(x5_test)

nb5 = MultinomialNB()
nb5.fit(x5_train,y5_train)
y5_nb_pred = nb5.predict(x5_test)
print("Unigrams with pos tagging as features---------------")
print("F1 Score of Naive bayes for sentiment classifier is ",metrics.accuracy_score(y5_test,y5_nb_pred))
lr5 = LogisticRegression(solver='lbfgs')
lr5.fit(x5_train,y5_train)
y5_lr_pred = lr5.predict(x5_test)
print("F1 Score of Logistic Regression for sentiment classifier is ",metrics.accuracy_score(y5_test,y5_lr_pred))
sv5 = svm.LinearSVC()
sv5.fit(x5_train,y5_train)
y5_sv_pred = sv5.predict(x5_test)
print("F1 Score of SVM for sentiment classifier is ",metrics.accuracy_score(y5_test,y5_sv_pred))

###########task - 3e done     

x6_train = x1_train
x6_test = x1_test
y6_train = y1_train
y6_test = y1_test
cv6 = CountVectorizer(max_features=2633)
cv6.fit(x1_train)
x6_train = cv6.transform(x1_train)
x6_test = cv6.transform(x1_test)
nb6 = MultinomialNB()
nb6.fit(x6_train,y6_train)
y6_nb_pred = nb6.predict(x6_test)
print("Top 2633 Unigrams as features-------------")
print("F1 Score of Naive bayes for sentiment classifier is ",metrics.accuracy_score(y6_test,y6_nb_pred))
lr6 = LogisticRegression(solver='lbfgs')
lr6.fit(x6_train,y6_train)
y6_lr_pred = lr6.predict(x6_test)
print("F1 Score of Logistic Regression for sentiment classifier is ",metrics.accuracy_score(y6_test,y6_lr_pred))
sv6 = svm.LinearSVC()
sv6.fit(x6_train,y6_train)
y6_sv_pred = sv6.predict(x6_test)
print("F1 Score of SVM for sentiment classifier is ",metrics.accuracy_score(y6_test,y6_sv_pred))

###########task - 3f done    
