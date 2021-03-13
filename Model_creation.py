import pandas as pd
import re
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

features = None
labels = None
model = None
vect = None

def data_sample():
    df_reader = pd.read_json('Clothing_Shoes_and_Jewelry.json', lines = True, chunksize = 1000000 ) # Reading Amazon Reviews in chunks
    counter = 1
    for chunk in df_reader:                                                  # Taking sample and creating chunks
        new_df = pd.DataFrame(chunk[['overall','reviewText','summary']])		
        new_df1 = new_df[new_df['overall'] == 5].sample(4000)
        new_df2 = new_df[new_df['overall'] == 4].sample(4000)
        new_df3 = new_df[new_df['overall'] == 3].sample(8000)
        new_df4 = new_df[new_df['overall'] == 2].sample(4000)
        new_df5 = new_df[new_df['overall'] == 1].sample(4000)
    
        new_df6 = pd.concat([new_df1,new_df2,new_df3,new_df4,new_df5], axis = 0, ignore_index = True)
    
        new_df6.to_csv(str(counter)+".csv", index = False)
    
        new_df = None
        counter = counter + 1
    filenames = glob('*.csv')
    dataframes = [pd.read_csv(f) for f in filenames]
    frame = pd.concat(dataframes, axis = 0, ignore_index = True)
    frame.to_csv('balanced_review.csv', index = False)      # Creating Final csv for model
    
def data_cleaning():
    global features
    global labels
    df = pd.read_csv('balanced_review.csv')
    df.dropna(inplace = True)
    df = df[df['overall'] != 3]
    df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )
    df.to_csv('balanced_review.csv', index = False)  
        
    corpus = []
    
    for i in range(0, 527386):
    
        review = re.sub('[^a-zA-Z]', ' ', df.iloc[i, 1])
        review = review.lower()
        review = review.split()
        review = [word for word in review if not word in stopwords.words('english')]
        ps =  PorterStemmer()
        review = [ps.stem(word) for word in review]
        review = " ".join(review)
        corpus.append(review)
        
    features = corpus
    labels = df.iloc[:,-1]

def model_build():
    global model
    global vect
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 42 ) 
    vect = TfidfVectorizer(min_df = 5).fit(features_train)
    features_train_vectorized = vect.transform(features_train)
    model = LogisticRegression()
    model.fit(features_train_vectorized, labels_train)

def model_vocab_dump():
    pickle.dump(vect.vocabulary_, open('feature.pkl','wb'))
    with open('pickle_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def main():
    global features
    global labels
    global model
    global vect
    print("Creating data sampling...")
    data_sample()
    print("Cleaning data...")
    data_cleaning()
    print("Building Model...")
    model_build()
    print("Dumping model in a pickle file...")
    model_vocab_dump()
    print("Pickle file is ready to be used...")
    features = None
    labels = None
    model = None
    vect = None
	
if __name__ == '__main__':
    main()
