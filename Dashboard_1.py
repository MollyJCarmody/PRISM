# # Dashboard #1 Notebook (Dashboard_1.ipynb)

# ### This notebook (#1) utilizes the OpenAI API (LLM) and Streamlit for dashboard development and deployment.
# ---

# Access DSA PostgreSQL DBMS: psql -h pgsql.dsa.lan -U mjct4g -d casestdysu24t05

# Google Trends Data Tip<br>
# Key for 'value':<br>
# "Breakout" = A sudden surge in searches. 'extracted_value' is arbitrary.<br>
# "+250%" = An increase of 250% in search interest.<br>
# "100" = Relative popularity (on a 0-100 scale).

# ### Query Data & Develop LLM-Dashboard Pipeline:

import numpy as np
import streamlit as st
import pandas as pd
import json
import os
import openai
import plotly.express as px

# Initialize OpenAI client.
client = openai.OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

# Load datasets.
google_trends_data = pd.read_csv("google_trends_data.csv")
news_articles_data = pd.read_csv("news_articles_data.csv")
reddit_posts_data = pd.read_csv("reddit_posts_data.csv")
reddit_sentiments_data = pd.read_csv("reddit_sentiments_data.csv")
google_reviews_data = pd.read_csv("google_reviews_data.csv")
google_reviews_sentiments_data = pd.read_csv("google_reviews_sentiments_data.csv")

# Convert data types.
google_trends_data["id"] = google_trends_data["id"].astype(int)
google_trends_data["industry"] = google_trends_data["industry"].astype(str)
google_trends_data["sub_industry"] = google_trends_data["sub_industry"].astype(str)
google_trends_data["data_type"] = google_trends_data["data_type"].astype(str)
google_trends_data["is_breakout"] = google_trends_data["is_breakout"].astype(bool)
google_trends_data["keywords"] = google_trends_data["keywords"].astype(str)
google_trends_data["search_volume"] = pd.to_numeric(google_trends_data["search_volume"], errors = "coerce")
google_trends_data["city_name"] = google_trends_data["city_name"].astype(str)
google_trends_data["cluster_number"] = pd.to_numeric(google_trends_data["cluster_number"], errors = "coerce", downcast = "integer")
google_trends_data["added_at"] = pd.to_datetime(google_trends_data["added_at"], errors = "coerce")
google_trends_data["data_source"] = google_trends_data["data_source"].astype(str)

news_articles_data["city_name"] = news_articles_data["city_name"].astype(str)
news_articles_data["data_source"] = news_articles_data["data_source"].astype(str)
news_articles_data["industry"] = news_articles_data["industry"].astype(str)
news_articles_data["sub_industry"] = news_articles_data["sub_industry"].astype(str)
news_articles_data["keywords"] = news_articles_data["keywords"].apply(lambda x: str(x))

reddit_posts_data["interpreted_theme_reddit"] = reddit_posts_data["interpreted_theme_reddit"].apply(
    lambda x: [t.strip() for t in x.split("/") if pd.notnull(x)]
)
reddit_posts_data["industry"] = reddit_posts_data["industry"].astype(str)
reddit_posts_data["sub_industry"] = reddit_posts_data["sub_industry"].astype(str)
reddit_posts_data["data_source"] = reddit_posts_data["data_source"].astype(str)

reddit_sentiments_data["sentiment_reddit"] = reddit_sentiments_data["sentiment_reddit"].astype(str).str.lower()
reddit_sentiments_data["industry"] = reddit_sentiments_data["industry"].astype(str)
reddit_sentiments_data["sub_industry"] = reddit_sentiments_data["sub_industry"].astype(str)
reddit_sentiments_data["data_source"] = reddit_sentiments_data["data_source"].astype(str)

google_reviews_data["interpreted_theme_google"] = google_reviews_data["interpreted_theme_google"].astype(str).str.lower()
google_reviews_data["city_name"] = google_reviews_data["city_name"].astype(str)
google_reviews_data["industry"] = google_reviews_data["industry"].astype(str)
google_reviews_data["sub_industry"] = google_reviews_data["sub_industry"].astype(str)
google_reviews_data["data_source"] = google_reviews_data["data_source"].astype(str)

google_reviews_sentiments_data["sentiment_google"] = google_reviews_sentiments_data["sentiment_google"].astype(str).str.lower()
google_reviews_sentiments_data["company"] = google_reviews_sentiments_data["company"].astype(str)
google_reviews_sentiments_data["city_name"] = google_reviews_sentiments_data["city_name"].astype(str)
google_reviews_sentiments_data["industry"] = google_reviews_sentiments_data["industry"].astype(str)
google_reviews_sentiments_data["sub_industry"] = google_reviews_sentiments_data["sub_industry"].astype(str)
google_reviews_sentiments_data["data_source"] = google_reviews_sentiments_data["data_source"].astype(str)

# Streamlit UI Components:
st.title("PRISM: Platform for Real-Time Insights & Strategic Marketing​")

# Dropdown for selecting data source (with dropdown prompt).
user_prompt = st.selectbox(
    "What would you like to do?",
    [
        "I want to generate content based on what people are currently searching on Google.",
        "I want to generate content inspired by recent news articles.",
        "I want to generate content using user sentiment and competitor insights."
    ]
)

# Map the selection back to data source.
prompt_to_data_source = {
    "I want to generate content based on what people are currently searching on Google.": "Google Trends",
    "I want to generate content inspired by recent news articles." : "News Articles",
    "I want to generate content using user sentiment and competitor insights." : "Reddit & Google Reviews"
}

source = prompt_to_data_source[user_prompt]

if source == "Google Trends":
    df_source = google_trends_data
elif source == "News Articles":
    df_source = news_articles_data
elif source == "Reddit & Google Reviews":
    df_source = pd.concat([
        google_reviews_data[["industry", "sub_industry", "city_name", "interpreted_theme_google"]].assign(company = "N/A", sentiment_google = "N/A", interpreted_theme_reddit = "N/A", sentiment_reddit = "N/A"),
        google_reviews_sentiments_data[["industry", "sub_industry", "city_name", "company", "sentiment_google"]].assign(interpreted_theme_google = "N/A", interpreted_theme_reddit = "N/A", sentiment_reddit = "N/A"),
        reddit_posts_data[["industry", "sub_industry", "interpreted_theme_reddit"]].assign(company = "N/A", sentiment_google = "N/A", interpreted_theme_google = "N/A", city_name = "N/A", sentiment_reddit = "N/A"),
        reddit_sentiments_data[["industry", "sub_industry", "sentiment_reddit"]].assign(company = "N/A", sentiment_google = "N/A", interpreted_theme_google = "N/A", city_name = "N/A", interpreted_theme_reddit = "N/A")
    ], ignore_index = True)

# Dropdowns for filtering.
available_industries = sorted(df_source["industry"].dropna().unique())
industry = st.selectbox("Select Industry:", available_industries)

available_sub_industries = sorted(
    df_source[df_source["industry"] == industry]["sub_industry"].dropna().unique()
)
sub_industry = st.selectbox("Select Sub-Industry:", available_sub_industries)

available_cities = sorted(
    df_source[
        (df_source["industry"] == industry) &
        (df_source["sub_industry"] == sub_industry) &
        (df_source["city_name"] != "N/A")
    ]["city_name"].dropna().unique()
)
city = st.selectbox("Select City:", available_cities)

platform = st.selectbox("Select Content Type/Platform:", [
    "LinkedIn", "Google Ads", "Facebook", "Twitter", "Instagram Ads", "TikTok Ads",
    "SEO Blog Content", "Email Campaigns", "Salesforce", "Outreach"
])

# LLM Model Selection:
llm_model = "gpt-4o-mini"

# Button to generate content.
if st.button("Generate Content"):
    
    if source == "Google Trends":
        # Filter data based on selections.
        filtered_df = df_source[
            (df_source["industry"] == industry) &
            (df_source["sub_industry"] == sub_industry) &
            (df_source["city_name"] == city)
        ]
        
        # Determine comparison city dynamically.
        comparison_city = "St Louis" if city == "Seattle" else "Seattle"
        
        # Aggregate total search volume for the selected city.
        total_search_volume_selected = df_source[
            (df_source["sub_industry"] == sub_industry) &
            (df_source["city_name"] == city)
        ]["search_volume"].sum()
        
        # Aggregate total search volume for the comparison city.
        total_search_volume_comparison = df_source[
            (df_source["sub_industry"] == sub_industry) &
            (df_source["city_name"] == comparison_city)
        ]["search_volume"].sum()
        
        # Display the search volume summary dynamically.
        st.subheader(f"Search Volume Summary for: {sub_industry}")
        st.write(f"**{city}:** {total_search_volume_selected:,.0f}")
        st.write(f"**{comparison_city}:** {total_search_volume_comparison:,.0f}")
        
        # Plotly Bar Chart Comparison:
        volume_df = pd.DataFrame({
            "City": [city, comparison_city],
            "Search Volume": [total_search_volume_selected, total_search_volume_comparison]
        })
        
        fig = px.bar(
            volume_df,
            x = "City",
            y = "Search Volume",
            color = "City",
            color_discrete_map = {
                city: "#F78FB3",
                comparison_city: "#6EC1E4"
            },
            title = f"Search Volume Comparison: {sub_industry}",
            text = "Search Volume"
        )
        
        fig.update_traces(texttemplate = '%{text:,.0f}', textposition = 'outside')
        fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide')
        
        st.plotly_chart(fig)
        
        # Prioritize clusters with highest search volume.
        non_breakout_df = filtered_df[filtered_df["is_breakout"] == False]
        
        if not non_breakout_df.empty:
            # Get all available clusters, or up to 5 if more exist.
            top_clusters = non_breakout_df.groupby("cluster_number")["search_volume"].sum().nlargest(5).index.tolist()
            
            # Collect keywords from all selected clusters, ensuring uniqueness.
            keyword_set = set()
            for cluster in top_clusters:
                cluster_keywords = non_breakout_df[non_breakout_df["cluster_number"] == cluster]["keywords"].values[0]
                top_keywords = cluster_keywords.split(", ")[:5] # Get up to 5 keywords per cluster.
                keyword_set.update(top_keywords) # Use set to avoid duplicates.
                
            keywords = ", ".join(keyword_set) # Convert set to comma-separated string.
        else:
            # If no search volume, fallback to breakout data.
            breakout_df = filtered_df[filtered_df["is_breakout"] == True]
            if not breakout_df.empty:
                keywords = breakout_df["keywords"].values[0]
            else:
                st.error("No relevant data found for this selection.")
                st.stop()
                
        # Related Topics Feature:
        related_topics = ""
        if "topic" in filtered_df["data_type"].values:
            # Extract all topic clusters.
            topic_df = filtered_df[filtered_df["data_type"] == "topic"].copy()
            
            # Check if clustering exists in the data.
            if "cluster_number" in topic_df.columns:
                # Get the top topics from each cluster (up to 5 per cluster).
                topic_clusters = topic_df.groupby("cluster_number")["search_volume"].sum().nlargest(5).index.tolist()
                topic_set = set()
                
                for cluster in topic_clusters:
                    cluster_topics = topic_df[topic_df["cluster_number"] == cluster]["keywords"].values[0]
                    top_cluster_topics = cluster_topics.split(", ")[:5] # Up to 5 topics per cluster.
                    topic_set.update(top_cluster_topics) # Avoid duplicates.
                    
                related_topics = ", ".join(topic_set)
            
            else:
                # If no clustering, just use top topics by search volume.
                top_topics = topic_df.sort_values(by = "search_volume", ascending = False)["keywords"].unique()[:5]
                related_topics = ", ".join(top_topics)

        # LLM Payload:
        payload = {
            "keywords": list(keyword_set)[:10], # Ensure the final keyword list does not exceed 10.
            "industry": industry,
            "ad_medium": "social media",
            "platform": platform
        }
        
        # Generate content using LLM.
        st.write("Generating content...")
        
        try:
            response = client.chat.completions.create(
                model = llm_model,
                messages = [
                    {"role": "system", "content": "You generate marketing content based on regional Google search trends. Use the keywords, but feel free to expand for accuracy."},
                    {"role": "user", "content": json.dumps(payload)}
                ]
            )

            # Extract and display response.
            generated_content = response.choices[0].message.content

            # Append "Users who searched for this also searched for:" only if related topics exist.
            if related_topics:
                generated_content += f"\n\n**Users who searched for this also searched for:** {related_topics}"

            st.subheader("Generated Content")
            st.write(generated_content)

        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
    
    elif source == "News Articles":
        # Filter data.
        filtered_df = df_source[
            (df_source["industry"] == industry) &
            (df_source["sub_industry"] == sub_industry) &
            (df_source["city_name"] == city)
        ]
        
        # Extract top keywords.
        all_keywords = ", ".join(filtered_df["keywords"].tolist())
        keyword_list = list(set(all_keywords.split(", ")))[:10]  # Limit to 10 unique keywords.
        
        # LLM Payload
        payload = {
            "keywords": keyword_list,
            "industry": industry,
            "ad_medium": "social media",
            "platform": platform
        }
        
        st.write("Generating content...")
        
        try:
            response = client.chat.completions.create(
                model = llm_model,
                messages = [
                    {"role": "system", "content": "You generate marketing content inspired by recent news. Use the keywords, but feel free to expand for accuracy."},
                    {"role": "user", "content": json.dumps(payload)}
                ]
            )

            generated_content = response.choices[0].message.content

            st.subheader("Generated Content")
            st.write(generated_content)

        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            
    elif source == "Reddit & Google Reviews":
        reddit_filtered_sentiment = reddit_sentiments_data[
            (reddit_sentiments_data["industry"] == industry) &
            (reddit_sentiments_data["sub_industry"] == sub_industry)
        ]
        
        reddit_filtered_theme = reddit_posts_data[
            (reddit_posts_data["industry"] == industry) &
            (reddit_posts_data["sub_industry"] == sub_industry)
        ]
        
        google_filtered_sentiment = google_reviews_sentiments_data[
            (google_reviews_sentiments_data["industry"] == industry) &
            (google_reviews_sentiments_data["sub_industry"] == sub_industry) &
            (google_reviews_sentiments_data["city_name"] == city)
        ]
        
        google_filtered_theme = google_reviews_data[
            (google_reviews_data["industry"] == industry) &
            (google_reviews_data["sub_industry"] == sub_industry) &
            (google_reviews_data["city_name"] == city)
        ]
        
        # 1. Sentiment Charts
        
        reddit_sentiment_counts = reddit_filtered_sentiment["sentiment_reddit"].value_counts().reset_index()
        reddit_sentiment_counts.columns = ["Sentiment", "Count"]
        
        st.subheader("Reddit Sentiment (Global)")
        fig_reddit = px.bar(
            reddit_sentiment_counts,
            x = "Sentiment",
            y = "Count",
            title = "Reddit Sentiment Distribution (All Users)",
            color = "Sentiment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_reddit)
        
        google_sentiment_counts = google_filtered_sentiment["sentiment_google"].value_counts().reset_index()
        google_sentiment_counts.columns = ["Sentiment", "Count"]
        
        st.subheader(f"Google Reviews Sentiment in {city}")
        fig_google = px.bar(
            google_sentiment_counts,
            x = "Sentiment",
            y = "Count",
            title = f"Google Reviews Sentiment in {city}",
            color = "Sentiment",
            color_discrete_sequence = px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_google)
        
        # 2. Positive Companies in Your Area (from Google)
        
        positive_companies = google_filtered_sentiment[
            google_filtered_sentiment["sentiment_google"] == "positive"
        ]["company"].value_counts().head(3).index.tolist()
        
        if positive_companies:
            st.markdown(f"**Companies with positive sentiment in {city}:** " + ", ".join(positive_companies))
        else:
            st.markdown(f"**No companies with positive sentiment found in {city}.**")

        # 3. LLM Payload

        # Combine top Reddit themes and Google review themes as keywords.
        reddit_keywords = reddit_filtered_theme["interpreted_theme_reddit"].explode().dropna().unique().tolist()
        google_keywords = google_filtered_theme["interpreted_theme_google"].dropna().str.split().explode().str.title().value_counts().index.tolist()
        
        combined_keywords = list(set(reddit_keywords + google_keywords))[:10]
        
        payload = {
            "keywords": combined_keywords,
            "industry": industry,
            "ad_medium": "social media",
            "platform": platform
        }
        
        st.write("Generating content...")
        
        try:
            response = client.chat.completions.create(
                model = llm_model,
                messages = [
                    {"role": "system", "content": "You generate marketing content based on real user sentiment and themes. Use the keywords, but feel free to expand for accuracy."},
                    {"role": "user", "content": json.dumps(payload)}
                ]
            )
            
            generated_content = response.choices[0].message.content
            st.subheader("Generated Content")
            st.write(generated_content)
        
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
