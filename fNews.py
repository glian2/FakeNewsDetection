import numpy as np
import pandas
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
pandas.set_option('display.max_columns',None)

df = pd.read_csv("news.csv")

labels=df.label

x_train, x_test, y_train, y_test = train_test_split(df['text'],labels, test_size=0.2, random_state=7)
# above function of train_test_split will take the text dataframe and use approximately 20% of the information as the test sampe
# the other 80% will be used to train.  In the case above, the split function will manually split outputs into the 4 formats as specified
# the integer of 7 for random is given for reproduceable outcomes.

###############################################################
# filtering out stop words in English language via TfidfVectorizer
tfidf_vect=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train= tfidf_vect.fit_transform(x_train)
tfidf_test= tfidf_vect.transform(x_test)

##############################################################
# Passive aggressive classifier

# Initialization
paClass = PassiveAggressiveClassifier(max_iter=50)
paClass.fit(tfidf_train,y_train)

# Prediction on test and find accuracy

y_pred = paClass.predict(tfidf_test)
score=accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

name = ['Positive','Negative']
cnm=confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
cnm=pd.DataFrame(cnm, name,name)
print(cnm)