import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import nltk
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from wordcloud import WordCloud


pd.set_option('display.width', None)
df = pd.read_csv(r"C:\Users\Hedaya_city\Downloads\spam.csv",encoding="latin1")
print(df.head())
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
print(df.head(20))

print("===================================")
print("================>>> Basic Function")
print("number of rows and columns")
print(df.shape)

print("Name of Columns")
print(df.columns)

print("Information about Data")
print(df.info())

print("Statistical Operation")
print(df.describe(include='object'))

print("Data types in Data")
print(df.dtypes)

print("Display the index Range")
print(df.index)

print("Random rows in Dataset")
print(df.sample(5))

print("===================================")
print("================>>> Data Cleaning")
print("Missing Values")
print(df.isnull().sum())

print("The Columns Unnamed: 2, Unnamed: 3, and Unnamed: 4 we don't have any information about them so we drop it .")
print(df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True))


print("Number of Frequent Rows")
print(df.duplicated().sum()) # 403

print("Remove Duplicates")
df = df.drop_duplicates(keep='first')
print(df.shape)

print(" There is Missing Values in Data")
print(df.isnull().sum())

sns.heatmap(df.isnull())
plt.title('Missing Values after Cleaning')
plt.show()

print("===================================")
print("================>>> Exploratory Data Analysis")
print("frequence Values in Label columns")
print(df['label'].value_counts())

print("number of characters")
df['num_characters'] = df['message'].apply(lambda x: len(x))
print(df.head())

print("number of words")
df['num_words'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
print(df.head())

print("number of sentences")
df['num_sentences'] = df['message'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df.head())

print("Statistical Operation in all Data")
Statis_Opera = df[['num_characters','num_words','num_sentences']].describe().round()
print(Statis_Opera)

print("Statistical Operation (Ham)")
Statis_Opera_Ham = df[df['label'] == 'ham'] [['num_characters', 'num_words', 'num_sentences']].describe().round()
print(Statis_Opera_Ham)

print("Statistical Operation (Spam)")
Statis_Opera_Spam = df[df['label'] == 'spam'] [['num_characters', 'num_words', 'num_sentences']].describe().round()
print(Statis_Opera_Spam)

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

numerical_cols = ['label', 'num_characters', 'num_words', 'num_sentences']
correlation_matrix = df[numerical_cols].corr()
print("Correlation Matrix")
print(correlation_matrix)

print("================>>> Text Preprocessing ")
df['lower_message'] = df['message'].str.lower()

df["tokenized_message"] = df['lower_message'].apply(word_tokenize)

df['clean_tokens'] = df['tokenized_message'].apply(
    lambda tokens: [re.sub(r'[^a-zA-Z]', '', word) for word in tokens if word.isalpha()])

stop_words = set(stopwords.words('english'))
df['no_stopwords'] = df['clean_tokens'].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

stemmer = PorterStemmer()
df['stemmed_tokens'] = df['no_stopwords'].apply(
    lambda tokens: [stemmer.stem(word) for word in tokens]
)

df['final_message'] = df['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens))

print("================>>> Visualization of Data")

plt.pie(df['label'].value_counts(),labels=['Ham','Spam'],autopct='%1.0f%%')
plt.title('Frequence Values in Label Columns')
plt.show()

sns.histplot(df[df['label'] == 0] [['num_characters']])
sns.histplot(df[df['label'] == 1] [['num_characters']],color = 'red')
plt.title('Number of Characters in Label Columns')
plt.xlabel("num_characters")
plt.ylabel("Count")
plt.grid()
plt.tight_layout()
plt.legend(['Ham','Spam'])
plt.show()

sns.histplot(df[df['label'] == 0] [['num_words']])
sns.histplot(df[df['label'] == 1] [['num_words']])
plt.title('Number of Words in Label Columns')
plt.xlabel("num_words")
plt.ylabel("Count")
plt.grid()
plt.tight_layout()
plt.legend(['Ham','Spam'])
plt.show()

sns.pairplot(df,hue='label')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title('Correlation Between Numerical Features',fontsize=14)
plt.show()

all_text = " ".join(df['final_message'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of Preprocessed Messages", fontsize=16)
plt.axis("off")
plt.show()

print("================>>> Machine Learning with MLflow")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
print(df.head())

X = df['final_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Spam_Classifier")

with mlflow.start_run():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
        ("nb", MultinomialNB())
    ])

    params = {
        'nb__alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    }
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    mlflow.log_param("model_type", "MultinomialNB")
    mlflow.log_param("alpha", grid.best_params_['nb__alpha'])

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    from sklearn.metrics import precision_score, recall_score, f1_score
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    mlflow.sklearn.log_model(best_model, "spam_classifier_model")

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham','Spam'],
                yticklabels=['Ham','Spam'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
