# # Dashboard #1 Notebook (Dashboard_1.ipynb)

# ### This notebook (#1) utilizes the OpenAI API (LLM) and Streamlit for dashboard development and deployment.
# ---

# Access DSA PostgreSQL DBMS: psql -h pgsql.dsa.lan -U mjct4g -d casestdysu24t05

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

# Load dataset.
df_clusters = pd.read_csv("df_clusters.csv")

# Remove unnecessary index column.
df_clusters = df_clusters.drop(columns = ["Unnamed: 0"], errors = "ignore")

# Convert data types.
df_clusters["id"] = df_clusters["id"].astype(int)
df_clusters["industry"] = df_clusters["industry"].astype(str)
df_clusters["sub_industry"] = df_clusters["sub_industry"].astype(str)
df_clusters["data_type"] = df_clusters["data_type"].astype(str)
df_clusters["is_breakout"] = df_clusters["is_breakout"].astype(bool)
df_clusters["keywords"] = df_clusters["keywords"].astype(str)
df_clusters["search_volume"] = pd.to_numeric(df_clusters["search_volume"], errors = "coerce")
df_clusters["city_name"] = df_clusters["city_name"].astype(str)
df_clusters["cluster_number"] = pd.to_numeric(df_clusters["cluster_number"], errors = "coerce", downcast = "integer")
df_clusters["added_at"] = pd.to_datetime(df_clusters["added_at"], errors = "coerce")
df_clusters["data_source"] = df_clusters["data_source"].astype(str) # Will need to edit later.

# Streamlit UI Components:
st.title("Marketing Content Generation")

# Dropdown for selecting data source.
data_source = st.selectbox("Select Data Source:", sorted(df_clusters["data_source"].unique()))

# Dropdowns for filtering.
industry = st.selectbox("Select Industry:", sorted(df_clusters[df_clusters["data_source"] == data_source]["industry"].unique()))
sub_industry = st.selectbox("Select Sub-Industry:", sorted(df_clusters[(df_clusters["data_source"] == data_source) &
                                                                        (df_clusters["industry"] == industry)]["sub_industry"].unique()))
city = st.selectbox("Select City:", sorted(df_clusters[(df_clusters["data_source"] == data_source) &
                                                        (df_clusters["industry"] == industry)]["city_name"].unique()))
platform = st.selectbox("Select Content Type/Platform:", [
    "LinkedIn", "Google Ads", "Facebook", "Twitter", "Instagram Ads", "TikTok Ads",
    "SEO Blog Content", "Email Campaigns", "Salesforce", "Outreach"
])

# LLM Model Selection:
llm_model = "gpt-4o-mini"

# Button to generate content.
if st.button("Generate Content"):

    # Filter data based on selections.
    filtered_df = df_clusters[
        (df_clusters["data_source"] == data_source) &
        (df_clusters["industry"] == industry) &
        (df_clusters["sub_industry"] == sub_industry) &
        (df_clusters["city_name"] == city)
    ]

    # Determine comparison city dynamically.
    comparison_city = "St Louis" if city == "Seattle" else "Seattle"

    # Aggregate total search volume for the selected city.
    total_search_volume_selected = df_clusters[
        (df_clusters["data_source"] == data_source) &
        (df_clusters["sub_industry"] == sub_industry) &
        (df_clusters["city_name"] == city)
    ]["search_volume"].sum()

    # Aggregate total search volume for the comparison city.
    total_search_volume_comparison = df_clusters[
        (df_clusters["data_source"] == data_source) &
        (df_clusters["sub_industry"] == sub_industry) &
        (df_clusters["city_name"] == comparison_city)
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
                {"role": "system", "content": "You are an AI that generates marketing content based on location and industry-specific keywords."},
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
