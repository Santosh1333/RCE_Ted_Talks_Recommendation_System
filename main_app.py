import numpy as np
import pandas as pd
import nltk
import string  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import streamlit as st
import datetime
import pytz

def display_time():
    """Displays the current Indian Standard Time."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    current_time = now.strftime("%I:%M:%S %p")
    st.markdown(current_time)

def set_background(image_path):
    """
    Function to set background image for Streamlit app.
    """
    # Set CSS for the background
    background_css = """
        <style>
        .stApp {
            background-image: url("""" + image_path + """");
            background-size: cover;
        }
        </style>
    """
    # Insert background CSS
    st.markdown(background_css, unsafe_allow_html=True)

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

@st.cache
def recommend_talks_with_sentiment(talk_content, comments, data=df, num_talks=10):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['details'])
    
    # TF-IDF vectorization for input talk content
    talk_content_vector = vectorizer.transform(talk_content)
    
    # Calculate cosine similarity between input talk content and all talks
    similarities = cosine_similarity(talk_content_vector, vectorizer.transform(data['details']))
    
    # Get indices of top similar talks
    top_talk_indices = similarities.argsort(axis=1)[:, -num_talks:].flatten()
    
    # Get comments of top similar talks
    selected_comments = comments.iloc[top_talk_indices]
    
    # Perform sentiment analysis on selected comments
    comment_sentiments = [analyze_sentiment(comment) for comment in selected_comments]
    
    # Combine sentiments with talk data
    data['sentiment_score'] = comment_sentiments
    
    # Sort talks by sentiment score
    recommended_talks = data.sort_values(by='sentiment_score', ascending=False).head(num_talks)
    
    return recommended_talks[['title', 'publushed_date', 'like_count']]

def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity

if __name__ == "__main__":
    set_background('background_image.jpg')  # Change 'background_image.jpg' to the path of your image file
    display_time()
    nltk.download('stopwords')
    nltk.download('punkt')

    df = pd.read_csv('JOINT_ted_video_transcripts_comments_stats.csv')
    df = df.dropna()

    df['details'] = df['title'] + ' ' + df['transcript']
    df['details'] = df['details'].apply(preprocess_text)

    comments = df['comments'].fillna('').astype(str)
    comments = comments.apply(preprocess_text)

    def get_similarities(talk_content, data=df):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(data['details'])
        talk_array1 = vectorizer.transform(talk_content)
        details_array = vectorizer.transform(data['details'])  
        sim = cosine_similarity(talk_array1, details_array)
        return sim.flatten()

    def main():
        st.title('TED Talk Recommendation System')
        talk_content = st.text_input('Enter your talk content:')
        if st.button('Recommend Talks'):
            recommended_titles = recommend_talks_with_sentiment([talk_content], comments)
            st.subheader('Recommended Talks:')
            count = 1  
            for index, row in recommended_titles.iterrows():
                search_query = row['title'].replace(' ', '+')
                google_link = "https://www.google.com/search?q=" + search_query
                st.write(f"{count}) {row['title']} - [Go]({google_link})", unsafe_allow_html=True)
                st.write(f"          Published Date: {row['publushed_date']}, Likes: {int(row['like_count'])}")
                count += 1  

            if st.button('Load More'):
                recommended_titles = recommend_talks_with_sentiment([talk_content], comments, num_talks=20)
                for index, row in recommended_titles.iloc[10:].iterrows():
                    search_query = row['title'].replace(' ', '+')
                    google_link = "https://www.google.com/search?q=" + search_query
                    st.write(f"{count}) {row['title']} - [Go]({google_link})", unsafe_allow_html=True)
                    st.write(f"          Published Date: {row['publushed_date']}, Likes: {int(row['like_count'])}")
                    count += 1  

    if __name__ == '__main__':
        main()
