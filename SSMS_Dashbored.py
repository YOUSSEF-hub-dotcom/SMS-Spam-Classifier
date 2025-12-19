import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk

nltk.download('punkt')

st.set_page_config(page_title="📊 SMS Spam Classifier Dashboard", layout="wide")
st.title("📱 SMS Spam Classifier Dashboard")
st.markdown("---")

df = pd.read_csv(r"C:\Users\Hedaya_city\Downloads\spam.csv", encoding="latin1")
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')

tab1, tab2, tab3 = st.tabs(["📘 Dataset Overview", "📊 EDA & Visualization", "🤖 Model Info"])

with tab1:
    st.header("📘 Dataset Overview")

    st.write("### Dataset Sample")
    st.dataframe(df.head(10))

    st.write("### Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    st.write("### Missing Values Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    st.pyplot(fig)

    st.write("### Label Distribution")
    st.bar_chart(df['label'].value_counts())

with tab2:
    st.header("📊 Exploratory Data Analysis")

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])

    df['num_characters'] = df['message'].apply(len)
    df['num_words'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['message'].apply(lambda x: len(nltk.sent_tokenize(x)))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Label Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(df['label'].value_counts(), labels=['Ham', 'Spam'], autopct='%1.0f%%')
        st.pyplot(fig1)

    with col2:
        st.subheader("Characters Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df[df['label'] == 0]['num_characters'], ax=ax2, label='Ham')
        sns.histplot(df[df['label'] == 1]['num_characters'], ax=ax2, color='red', label='Spam')
        ax2.legend()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Words Distribution")
        fig3, ax3 = plt.subplots()
        sns.histplot(df[df['label'] == 0]['num_words'], ax=ax3, label='Ham')
        sns.histplot(df[df['label'] == 1]['num_words'], ax=ax3, color='red', label='Spam')
        ax3.legend()
        st.pyplot(fig3)

    with col4:
        st.subheader("Correlation Heatmap")
        corr = df[['label', 'num_characters', 'num_words', 'num_sentences']].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Boxplot of Message Length (Characters)")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='label', y='num_characters', data=df, palette='Set2', ax=ax_box)
        ax_box.set_xticklabels(['Ham', 'Spam'])
        st.pyplot(fig_box)

    with col6:
        st.subheader("Words vs Sentences per Message")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(x='num_words', y='num_sentences', hue='label', data=df, alpha=0.7, ax=ax_scatter)
        ax_scatter.set_xlabel("Number of Words")
        ax_scatter.set_ylabel("Number of Sentences")
        st.pyplot(fig_scatter)

    st.subheader("WordCloud of All Messages")
    all_text = " ".join(df['message'])
    wordcloud = WordCloud(width=900, height=400, background_color='white').generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

with tab3:
    st.header("🤖 Model Information")

    try:
        model = joblib.load("spam_classifier_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        st.success("✅ Model & Vectorizer loaded successfully!")
    except:
        st.error("❌ Model files not found. Please run the training script first.")

    st.markdown("### Accuracy & Confusion Matrix (from training results)")
    st.write("Accuracy: **~0.97** (from your training results)")

    cm = [[965, 10], [12, 138]]
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Youssef Mahmoud**")