import streamlit as st

# Page 1: Recommender
def page_recommender():
    import numpy as np
    import pandas as pd
    import nltk
    import string  
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from textblob import TextBlob
    import datetime
    import pytz

    def display_time():
        """Displays the current Indian Standard Time."""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        current_time = now.strftime("%I:%M:%S %p")
        st.markdown(current_time)



    if __name__ == "__main__":
        # Change 'background_image.jpg' to the path of your image file
        display_time()
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

        @st.cache
        def recommend_talks_with_sentiment(talk_content, comments, data=df, num_talks=10):
            vectorizer = TfidfVectorizer()
            vectorizer.fit(data['details'])
            talk_content_vector = vectorizer.transform(talk_content)
            details_array = vectorizer.transform(data['details'])  
            sim = cosine_similarity(talk_content_vector, details_array)
            cos_similarities = sim.flatten()
            weighted_score = 0.8 * cos_similarities
            data['score'] = weighted_score
            recommended_talks = data.sort_values(by='score', ascending=False).head(num_talks)
            return recommended_talks[['title', 'publushed_date', 'like_count']]  

        def main():
            st.title('TED Talk Recommendation System - Recommender')
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

# Page 2: Top Talks
def page_top_talks():
    st.title('TED Talk Recommendation System - Top Talks')
    st.write("This is the Top Talks page.")

# Page 3: Explore
def page_explore():
    st.title('TED Talk Recommendation System - Explore')
    st.write("This is the Explore page.")

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Recommender", "Top Talks", "Explore"])

    if page == "Recommender":
        page_recommender()
    elif page == "Top Talks":
        page_top_talks()
    elif page == "Explore":
        page_explore()

if __name__ == "__main__":
    main()
