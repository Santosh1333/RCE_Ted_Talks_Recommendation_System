import streamlit as st
import numpy as np
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import datetime
import pytz

# Custom CSS for sidebar tiles
sidebar_custom_css = """
<style>
.sidebar .sidebar-content {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.sidebar .sidebar-content .block-container {
    margin-top: 20px;
    width: 150px;
    text-align: center;
}
.sidebar .sidebar-content .block-container img {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 50%;
    margin-bottom: 10px;
}
</style>
"""

# Apply custom CSS
st.markdown(sidebar_custom_css, unsafe_allow_html=True)

# Custom CSS for main content area
main_custom_css = """
<style>
body {
    background-color: #cfe8fc; /* Blue background */
}
</style>
"""

# Apply custom CSS
st.markdown(main_custom_css, unsafe_allow_html=True)

# Sidebar data
sidebar_data = {
    "Recommender": {
        "description": "Get personalized TED Talk recommendations.",
        "icon": "https://image.flaticon.com/icons/png/512/2099/2099056.png",
    },
    "Top Choices": {
        "description": "Discover the top trending TED Talks.",
        "icon": "https://image.flaticon.com/icons/png/512/300/300218.png",
    },
    "Explore": {
        "description": "Explore TED Talks by category.",
        "icon": "https://image.flaticon.com/icons/png/512/3628/3628665.png",
    }
}



# Page 1: Recommender
def page_recommender():
    

    if __name__ == "__main__":
        
        nltk.download('stopwords')
        nltk.download('punkt')

        df = pd.read_csv('JOINT_ted_video_transcripts_comments_stats.csv')
        df = df.dropna()

        def preprocess_text(text):
            if pd.isnull(text):
                return ''
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens = nltk.word_tokenize(text)
            text = ' '.join([word for word in tokens if word not in stop_words])
            return text

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

# Function to recommend talks with sentiment analysis
        @st.cache
        def recommend_talks_with_sentiment(talk_content, comments, data=df, num_talks=10):
            cos_similarities = get_similarities(talk_content)
            comment_sentiments = comments.apply(analyze_sentiment).values
            weighted_score = 0.8 * cos_similarities + 0.3 * comment_sentiments
            data['score'] = weighted_score
            recommended_talks = data.sort_values(by='score', ascending=False)
            

            # Sentiment analysis of comments
            comment_sentiments = comments.apply(analyze_sentiment).values
            recommended_talks['sentiment_score'] = comment_sentiments
            
            return recommended_talks[['title', 'publushed_date', 'like_count']].head(num_talks)  

        def analyze_sentiment(comment):
            analysis = TextBlob(comment)
            return analysis.sentiment.polarity

        st.title('TED Talk Recommendation System - Recommender')
        talk_content = st.text_input('Enter your talk content:')
        if st.button('Recommend Talks'):
            recommended_titles = recommend_talks_with_sentiment([talk_content], comments,num_talks=15)
            st.subheader('Recommended Talks:')
            count = 1  
            for index, row in recommended_titles.iterrows():
                search_query = row['title'].replace(' ', '+')
                google_link = "https://www.google.com/search?q=" + search_query
                st.write(f"{count}) {row['title']} - [Go]({google_link})", unsafe_allow_html=True)
                st.write(f"          Published Date: {row['publushed_date']}, Likes: {int(row['like_count'])}")
                count += 1  

# Page 2: Top Talks
def page_top_talks():
    st.title('TED Talk Recommendation System - Top Talks')
    st.subheader('Top Trending TED Talks:')
    
    # Sort the DataFrame by like count and comment count
    top_talks = df.sort_values(by=['like_count', 'comments_count'], ascending=False).head(10)
    
    # Display the top talks
    for index, row in top_talks.iterrows():
        search_query = row['title'].replace(' ', '+')
        google_link = "https://www.google.com/search?q=" + search_query
        st.write(f"- [{row['title']}]({google_link})")
        st.write(f"  Published Date: {row['publushed_date']}, Likes: {int(row['like_count'])}, Comments: {int(row['comments_count'])}")



# Page 3: Explore
def page_explore():
    st.title('TED Talk Recommendation System - Explore')
    st.write("This is the Explore page.")

# Main app
def main():
    def display_time():
        """Displays the current Indian Standard Time."""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        current_time = now.strftime("%I:%M:%S %p")
        st.sidebar.markdown(f"**Current Time:** {current_time}")
    display_time()
    st.sidebar.title("MainMenu")
    selected_page = st.sidebar.selectbox("Go to", ["Recommender", "Top Choices", "Explore"])

    if selected_page == "Recommender":
        page_recommender()
    elif selected_page == "Top Choices":
        page_top_talks()
    elif selected_page == "Explore":
        page_explore()

if __name__ == "__main__":
    main()
