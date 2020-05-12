# TASK - 1 : To build a Subjective-Objective classifier, by using 5000+5000 dataset
# TASK - 2 : Constructing a function that performs mincut for a given file
# TASK - 3 : Calling the function over all the 2000 files and creating new modified reviews

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import os
from os import listdir

dataset = {}
#0 - objective sentences
#1 - subjective sentences
dataset[0] = []
dataset[1] = []
def load_to_dataset(filename):
    if(filename == 'objective.txt'):
        key = 0
    else:
        key = 1
    for line in open(filename, encoding = "ISO-8859-1"):
        dataset[key].append(line.strip())

load_to_dataset('subjective.txt')
load_to_dataset('objective.txt')

x_train = dataset[0]+dataset[1]
y_train = []
for i in range(10000):
    if(i<5000):
        y_train.append(0)
    else:
        y_train.append(1)

cv= CountVectorizer()
cv.fit(x_train)
x_train =cv.transform(x_train)
nb=MultinomialNB()
nb.fit(x_train,y_train)
#y_pred=nb.predict(x_train)
#print(metrics.accuracy_score(y_train,y_pred))

#This ->nb<- is the model that can classify given sentence into subjective/objective
###########Task - 1 done 

###Constructing Graph-Mincut based Subjectivity detector
# Steps :
    # 1. Construct Graph for sentences of each document
    # 2. Perform mincut
    # 3. Store the sentences , that are towards "Source" to another document
    # 4. Likewise, modify all the 2000 pos and neg files and perform paper1 on those files

def extract_subjective(filename):
    #returns list of subjective sentences in the given filename
    # Sink -> Objective class, Source -> Subjective class
    sentences = []
    for line in open(filename,encoding = "ISO-8859-1"):
        sentences.append(line.strip())
    G = nx.Graph()
    
    #Drawing individual scores
    for i in range(len(sentences)):
        sent = []
        sent.append(sentences[i])
        sent = cv.transform(sent)
        prob = nb.predict_proba(sent)
        G.add_edge(i,'Source',capacity=prob[0][1])
        G.add_edge(i,'Sink',capacity=prob[0][0])
    
    #Drawing associative scores
    # Fixing Threshold distance between two sentences=3 and constant,c =0.001,function=1/d^2
    c=0.001
    for i in range(len(sentences)):
        for j in range(i+1,len(sentences)):
            if(j-i<=3):
                associativity = c/((j-i)*(j-i))
            else:
                associativity = 0
            G.add_edge(i,j,capacity=associativity)
    # to visualize, 
    #nx.draw_networkx(G, with_label = True)
    #print(list(G.edges(data = True)))
    
    
    #              Now performing min cut on graph to get two classes,
    cut_value,partition = nx.minimum_cut(G,'Source','Sink')
    
    #Partition contains SETS of erachable and non-reachable vertices
    reachable,non_reachable = partition
    subjective_sentences = []
    for index in reachable:
        if(index!='Source'):
            subjective_sentences.append(sentences[index])
    return subjective_sentences

#########Task2 done

def modify_files(directory):
    new_dir = 'new_'+directory
    os.mkdir(new_dir)
    i=0
    for filename in listdir(directory):
        #print("Started ",filename)
        
        path = directory + '/' + filename
        subjective_sentences = extract_subjective(path)
        text = "\n".join(subjective_sentences)
        new_file_prop = new_dir+'/'+'new_'+filename
        f = open(new_file_prop,"w+")
        f.write(text)
        f.close()
        print("Ended ", filename)

modify_files('neg')
modify_files('pos')        

