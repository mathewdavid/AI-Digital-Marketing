import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import base64
from cryptography.fernet import Fernet
import requests

# Configure page settings
st.set_page_config(page_title="AI-Powered Marketing Assistant", layout="wide")

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'exchange_rate' not in st.session_state:
    st.session_state.exchange_rate = 75.0  # Default exchange rate as float
if 'business_info' not in st.session_state:
    st.session_state.business_info = {}

# Generate a Fernet key if it doesn't exist
if 'fernet_key' not in st.session_state:
    st.session_state.fernet_key = Fernet.generate_key()

fernet = Fernet(st.session_state.fernet_key)

# Function to encrypt API key
def encrypt_api_key(api_key):
    return fernet.encrypt(api_key.encode()).decode()

# Function to decrypt API key
def decrypt_api_key(encrypted_api_key):
    return fernet.decrypt(encrypted_api_key.encode()).decode()

# Function to validate API key
@st.cache_data(ttl=3600)
def validate_api_key(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Test")
        return True
    except Exception as e:
        st.error(f"API key validation failed: {str(e)}")
        return False

# Function to generate AI content with caching and error handling
@st.cache_data(ttl=3600)
def generate_ai_content(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        content = response.text

        # Attempt to parse the content as JSON
        try:
            json_content = json.loads(content)
            return json_content
        except json.JSONDecodeError:
            # If JSON parsing fails, attempt to extract JSON from the text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                try:
                    json_content = json.loads(json_str)
                    return json_content
                except json.JSONDecodeError:
                    raise ValueError("Unable to extract valid JSON from the generated content")
            else:
                raise ValueError("No JSON object found in the generated content")
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return {"error": str(e)}

# Function to convert USD to INR
def usd_to_inr(amount_usd):
    return amount_usd * st.session_state.exchange_rate

# Function to fetch the current USD to INR exchange rate
def fetch_exchange_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        st.error(f"Error fetching exchange rate: {str(e)}")
        return None

# API key input and validation
def api_key_input():
    st.title("Welcome to AI-Powered Marketing Assistant")
    st.write("Please enter your Google Gemini API key to get started.")

    api_key = st.text_input("Enter your Google Gemini API key", type="password")

    if st.button("Validate and Start"):
        if validate_api_key(api_key):
            encrypted_key = encrypt_api_key(api_key)
            st.session_state.api_key = encrypted_key
            st.session_state.api_key_validated = True
            st.success("API key validated successfully! Redirecting to the main application...")
            time.sleep(2)
            st.rerun()
        else:
            st.error("Invalid API key. Please try again.")

# Main application
def main_app():
    st.title("AI-Powered Marketing Assistant")

    # Sidebar for navigation and settings
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Choose a module",
                        ["Business Info", "Market Research", "Content Strategy", "Ad Campaign", "Customer Persona",
                         "Performance Analysis"])

        st.title("Settings")
        language = st.selectbox("Language", ["English", "Español", "Français"])
        st.session_state.language = language.lower()[:2]

        # Fetch and display real-time exchange rate
        if st.button("Fetch Current Exchange Rate"):
            with st.spinner("Fetching current USD to INR rate..."):
                current_rate = fetch_exchange_rate()
                if current_rate:
                    st.session_state.exchange_rate = current_rate
                    st.success(f"Current USD to INR rate: {current_rate:.2f}")
                else:
                    st.error("Failed to fetch current rate. Using default rate.")

        exchange_rate = st.number_input("USD to INR Exchange Rate",
                                        value=float(st.session_state.exchange_rate),
                                        step=0.01,
                                        format="%.2f")
        if exchange_rate != st.session_state.exchange_rate:
            st.session_state.exchange_rate = exchange_rate

    # Display the selected page
    if page == "Business Info":
        business_info()
    elif page == "Market Research":
        market_research()
    elif page == "Content Strategy":
        content_strategy()
    elif page == "Ad Campaign":
        ad_campaign()
    elif page == "Customer Persona":
        customer_persona()
    elif page == "Performance Analysis":
        performance_analysis()

# Business Info page
def business_info():
    st.header("Business Information")

    col1, col2 = st.columns(2)

    with col1:
        business_type = st.text_input("Business Type", value=st.session_state.business_info.get('type', ''), placeholder="e.g., E-commerce, SaaS, Retail")
        target_audience = st.text_area("Target Audience", value=st.session_state.business_info.get('audience', ''), placeholder="e.g., Young professionals aged 25-35")
        business_location = st.text_input("Business Location", value=st.session_state.business_info.get('location', ''), placeholder="e.g., New York, USA")

    with col2:
        marketing_goals = st.text_area("Current Marketing Goals", value=st.session_state.business_info.get('goals', ''), placeholder="e.g., Increase brand awareness, Generate leads")
        unique_selling_points = st.text_area("Unique Selling Points", value=st.session_state.business_info.get('usp', ''), placeholder="e.g., Eco-friendly products, 24/7 customer support")
        ad_location = st.text_input("Location for Ads/Marketing", value=st.session_state.business_info.get('ad_location', ''), placeholder="e.g., Global, North America, Europe")

    if st.button("Save Information"):
        st.session_state.business_info = {
            "type": business_type,
            "audience": target_audience,
            "goals": marketing_goals,
            "usp": unique_selling_points,
            "location": business_location,
            "ad_location": ad_location
        }
        st.success("Information saved successfully!")
        st.info("You can now proceed to other modules to leverage this information.")

# Market Research page
def market_research():
    st.header("Market Research and Competitor Insights")

    if not st.session_state.business_info:
        st.warning("Please fill out the Business Information first.")
        return

    st.info("Generating market research based on your business information. This may take a moment...")

    prompt = f"""
    Conduct a comprehensive market research analysis for a {st.session_state.business_info['type']} business.
    Target audience: {st.session_state.business_info['audience']}
    Marketing goals: {st.session_state.business_info['goals']}
    Unique Selling Points: {st.session_state.business_info['usp']}
    Business Location: {st.session_state.business_info['location']}
    Ad Location: {st.session_state.business_info['ad_location']}

    Provide the following in JSON format:
    {{
        "market_trends": ["trend1", "trend2", "trend3", "trend4", "trend5"],
        "key_competitors": [
            {{"name": "Competitor 1", "description": "Brief description"}},
            {{"name": "Competitor 2", "description": "Brief description"}},
            {{"name": "Competitor 3", "description": "Brief description"}}
        ],
        "seo_keywords": [
            {{"keyword": "keyword1", "relevance": 0.9}},
            {{"keyword": "keyword2", "relevance": 0.8}},
            {{"keyword": "keyword3", "relevance": 0.7}},
            {{"keyword": "keyword4", "relevance": 0.6}},
            {{"keyword": "keyword5", "relevance": 0.5}}
        ],
        "social_media_recommendations": [
            {{"platform": "Platform 1", "reason": "Reason for recommendation"}},
            {{"platform": "Platform 2", "reason": "Reason for recommendation"}},
            {{"platform": "Platform 3", "reason": "Reason for recommendation"}}
        ],
        "swot_analysis": {{
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "opportunities": ["opportunity1", "opportunity2"],
            "threats": ["threat1", "threat2"]
        }}
    }}
    """

    research_data = generate_ai_content(prompt)

    if "error" in research_data:
        st.error(research_data["error"])
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Market Trends")
            fig = go.Figure(data=[go.Table(
                header=dict(values=["Trend"]),
                cells=dict(values=[research_data["market_trends"]])
            )])
            st.plotly_chart(fig)

            st.subheader("Key Competitors")
            fig = go.Figure(data=[go.Table(
                header=dict(values=["Competitor", "Description"]),
                cells=dict(values=[
                    [comp["name"] for comp in research_data["key_competitors"]],
                    [comp["description"] for comp in research_data["key_competitors"]]
                ])
            )])
            st.plotly_chart(fig)

        with col2:
            st.subheader("SEO Keywords")
            keywords_df = pd.DataFrame(research_data["seo_keywords"])
            fig = px.bar(keywords_df, x='keyword', y='relevance', title="SEO Keywords Relevance")
            st.plotly_chart(fig)

            st.subheader("Recommended Social Media Platforms")
            for platform in research_data["social_media_recommendations"]:
                st.write(f"- **{platform['platform']}**: {platform['reason']}")

        st.subheader("SWOT Analysis")
        swot_data = research_data["swot_analysis"]
        fig = go.Figure()
        for category, items in swot_data.items():
            fig.add_trace(go.Table(
                header=dict(values=[category.capitalize()]),
                cells=dict(values=[items]),
                domain=dict(x=[0.25 if category in ["strengths", "weaknesses"] else 0.75,
                               0.75 if category in ["strengths", "weaknesses"] else 1.0],
                            y=[0.5 if category in ["strengths", "opportunities"] else 0,
                               1.0 if category in ["strengths", "opportunities"] else 0.5])
            ))
        fig.update_layout(title_text="SWOT Analysis")
        st.plotly_chart(fig)

# Content Strategy page
def content_strategy():
    st.header("Content Strategy and Generation")

    col1, col2 = st.columns(2)

    with col1:
        content_type = st.selectbox("Select content type",
                                    ["Blog Post", "Ad Copy", "Email Campaign", "Social Media Post"])
        topic = st.text_input("Enter the topic or product", placeholder="e.g., New product launch, Summer sale")

    with col2:
        tone = st.select_slider("Select content tone", options=["Formal", "Neutral", "Casual", "Humorous"])
        target_platform = st.selectbox("Select target platform",
                                       ["Website", "Facebook", "Instagram", "LinkedIn", "Twitter"])

    if st.button("Generate Content"):
        if not st.session_state.business_info:
            st.warning("Please fill out the Business Information first.")
            return

        st.info("Generating content based on your inputs. This may take a moment...")

        prompt = f"""
        Create a {content_type} about {topic} for a {st.session_state.business_info['type']} business.
        Tone: {tone}
        Target Platform: {target_platform}
        Target Audience: {st.session_state.business_info['audience']}
        Business Location: {st.session_state.business_info['location']}
        Ad Location: {st.session_state.business_info['ad_location']}
        Unique Selling Points: {st.session_state.business_info['usp']}

        Provide the following in JSON format:
        {{
            "title": "Content Title",
            "content": "Main content body",
            "call_to_action": "Call to action text",
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
            "estimated_reading_time": 5,
            "predicted_performance": {{
                "engagement_rate": 3.5,
                "click_through_rate": 2.1
            }}
        }}
        """

        content_data = generate_ai_content(prompt)

        if "error" in content_data:
            st.error(content_data["error"])
        else:
            st.subheader(content_data["title"])
            st.write(content_data["content"])
            st.write(f"**Call to Action:** {content_data['call_to_action']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Keywords")
                for keyword in content_data["keywords"]:
                    st.write(f"- {keyword}")

            with col2:
                st.metric("Estimated Reading Time", f"{content_data['estimated_reading_time']} minutes")

            with col3:
                st.subheader("Predicted Performance")
                st.write(f"Engagement Rate: {content_data['predicted_performance']['engagement_rate']}%")
                st.write(f"Click-Through Rate: {content_data['predicted_performance']['click_through_rate']}%")

# Ad Campaign page
def ad_campaign():
    st.header("Ad Campaign Builder and Optimizer")

    col1, col2 = st.columns(2)

    with col1:
        campaign_name = st.text_input("Campaign Name", placeholder="e.g., Summer Sale 2023")
        platform = st.selectbox("Ad Platform", ["Google Ads", "Facebook Ads", "LinkedIn Ads"])

    with col2:
        budget = st.number_input("Budget (INR)", min_value=1000, help="Enter your total campaign budget in INR")
        duration = st.number_input("Campaign Duration (days)", min_value=1, max_value=365,
                                   help="Enter the number of days your campaign will run")

    if st.button("Generate Campaign Blueprint"):
        if not st.session_state.business_info:
            st.warning("Please fill out the Business Information first.")
            return

        st.info("Generating campaign blueprint. This may take a moment...")

        prompt = f"""
        Create a detailed ad campaign blueprint for {campaign_name} on {platform}.
        Business type: {st.session_state.business_info['type']}
        Target audience: {st.session_state.business_info['audience']}
        Budget: ₹{budget}
        Duration: {duration} days
        Unique Selling Points: {st.session_state.business_info['usp']}
        Business Location: {st.session_state.business_info['location']}
        Ad Location: {st.session_state.business_info['ad_location']}

        Provide the following in JSON format:
        {{
            "campaign_structure": ["Ad Group 1", "Ad Group 2", "Ad Group 3"],
            "targeting_parameters": {{
                "age_range": "25-45",
                "locations": ["New York", "Los Angeles", "Chicago"],
                "interests": ["Technology", "Innovation"]
            }},
            "ad_copy_suggestions": [
                {{
                    "headline": "Headline 1",
                    "description": "Description 1"
                }},
                {{
                    "headline": "Headline 2",
                    "description": "Description 2"
                }},
                {{
                    "headline": "Headline 3",
                    "description": "Description 3"
                }}
            ],
            "bid_strategy": "CPC",
            "daily_budget": 2000,
            "estimated_kpis": {{
                "CTR": "2.5%",
                "CPC": "₹50",
                "Conversions": 100,
                "ROAS": "250%"
            }},
            "timeline": [
                "Day 1: Launch campaign",
                "Day 7: First performance review",
                "Day 14: Mid-campaign optimization",
                "Day 30: Final performance analysis"
            ],
            "optimization_recommendations": [
                "Adjust bids for high-performing keywords",
                "Refine audience targeting based on initial results",
                "A/B test ad copy variations"
            ]
        }}
        """

        campaign_data = generate_ai_content(prompt)

        if "error" in campaign_data:
            st.error(campaign_data["error"])
        else:
            st.subheader("Campaign Blueprint")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Campaign Structure**")
                for group in campaign_data["campaign_structure"]:
                    st.write(f"- {group}")

                st.write("**Targeting Parameters**")
                for param, value in campaign_data["targeting_parameters"].items():
                    st.write(f"- {param.capitalize()}: {', '.join(value) if isinstance(value, list) else value}")

            with col2:
                st.write("**Ad Copy Suggestions**")
                for i, ad in enumerate(campaign_data["ad_copy_suggestions"], 1):
                    st.write(f"Ad {i}:")
                    st.write(f"Headline: {ad['headline']}")
                    st.write(f"Description: {ad['description']}")
                    st.write("---")

            st.write(f"**Bid Strategy:** {campaign_data['bid_strategy']}")
            st.write(f"**Daily Budget:** ₹{campaign_data['daily_budget']}")

            st.subheader("Estimated KPIs")
            kpi_cols = st.columns(len(campaign_data["estimated_kpis"]))
            for i, (metric, value) in enumerate(campaign_data["estimated_kpis"].items()):
                kpi_cols[i].metric(metric, value)

            st.subheader("Campaign Timeline")
            for event in campaign_data["timeline"]:
                st.write(f"- {event}")

            st.subheader("Optimization Recommendations")
            for rec in campaign_data["optimization_recommendations"]:
                st.write(f"- {rec}")

# Customer Persona page
def customer_persona():
    st.header("Customer Persona and Engagement Module")

    st.subheader("Generate Customer Persona")

    col1, col2 = st.columns(2)

    with col1:
        age_range = st.slider("Age Range", 18, 65, (25, 40))
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        location = st.text_input("Location", placeholder="e.g., New York City")

    with col2:
        interests = st.multiselect("Interests",
                                   ["Technology", "Fashion", "Sports", "Travel", "Food", "Music", "Art", "Finance",
                                    "Health", "Education"])
        income_range = st.select_slider("Income Range", options=["Low", "Medium", "High"])
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    if st.button("Create Persona"):
        if not st.session_state.business_info:
            st.warning("Please fill out the Business Information first.")
            return

        st.info("Generating customer persona. This may take a moment...")

        prompt = f"""
        Create a detailed customer persona for a {st.session_state.business_info['type']} business.
        Age range: {age_range[0]}-{age_range[1]}
        Gender: {gender}
        Location: {location}
        Interests: {', '.join(interests)}
        Income Range: {income_range}
        Education: {education}
        Business Type: {st.session_state.business_info['type']}
        Target Audience: {st.session_state.business_info['audience']}
        Business Location: {st.session_state.business_info['location']}
        Ad Location: {st.session_state.business_info['ad_location']}

        Provide the following in JSON format:
        {{
            "name": "Persona Name",
            "description": "Brief description of the persona",
            "goals": ["Goal 1", "Goal 2", "Goal 3"],
            "challenges": ["Challenge 1", "Challenge 2", "Challenge 3"],
            "preferred_communication_channels": ["Channel 1", "Channel 2", "Channel 3"],
            "buying_behavior": "Description of buying behavior",
            "decision_making_factors": ["Factor 1", "Factor 2", "Factor 3"],
            "brand_affinities": ["Brand 1", "Brand 2", "Brand 3"],
            "technology_usage": "Description of technology usage",
            "engagement_strategy_recommendations": ["Strategy 1", "Strategy 2", "Strategy 3"]
        }}
        """

        persona_data = generate_ai_content(prompt)

        if "error" in persona_data:
            st.error(persona_data["error"])
        else:
            st.subheader(f"Meet {persona_data.get('name', 'Your Persona')}")
            st.write(persona_data.get("description", "No description available."))

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Goals:**")
                for goal in persona_data.get("goals", ["No goals specified."]):
                    st.write(f"- {goal}")

                st.write("**Challenges:**")
                for challenge in persona_data.get("challenges", ["No challenges specified."]):
                    st.write(f"- {challenge}")

                st.write("**Preferred Communication Channels:**")
                for channel in persona_data.get("preferred_communication_channels", ["No channels specified."]):
                    st.write(f"- {channel}")

            with col2:
                st.write("**Decision-making Factors:**")
                for factor in persona_data.get("decision_making_factors", ["No factors specified."]):
                    st.write(f"- {factor}")

                st.write("**Brand Affinities:**")
                for brand in persona_data.get("brand_affinities", ["No brand affinities specified."]):
                    st.write(f"- {brand}")

                st.write("**Technology Usage:**")
                st.write(persona_data.get("technology_usage", "No technology usage information available."))

            st.subheader("Buying Behavior")
            st.write(persona_data.get("buying_behavior", "No buying behavior information available."))

            st.subheader("Engagement Strategy Recommendations")
            for rec in persona_data.get("engagement_strategy_recommendations", ["No recommendations available."]):
                st.write(f"- {rec}")


# Performance Analysis page
def performance_analysis():
    st.header("Campaign Performance Analysis and Reporting")

    data_source = st.radio("Select data source", ["Upload CSV", "Provide Campaign Link"])

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.write(data.head())
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
    else:
        campaign_link = st.text_input("Enter campaign link", placeholder="e.g., https://example.com/campaign")
        if campaign_link:
            st.info(f"Analyzing campaign data from: {campaign_link}")
            # Here you would typically fetch data from the provided link
            # For this example, we'll use dummy data
            data = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=30),
                'Impressions': np.random.randint(1000, 5000, 30),
                'Clicks': np.random.randint(50, 200, 30),
                'Conversions': np.random.randint(5, 20, 30),
                'Spend': np.random.uniform(100, 500, 30)
            })

    if 'data' in locals():
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Impressions", f"{data['Impressions'].sum():,}")
        col2.metric("Total Clicks", f"{data['Clicks'].sum():,}")
        col3.metric("Total Conversions", f"{data['Conversions'].sum():,}")
        col4.metric("Total Spend", f"₹{usd_to_inr(data['Spend'].sum()):,.2f}")

        # Visualize performance metrics
        st.subheader("Performance Over Time")
        fig = px.line(data, x='Date', y=['Impressions', 'Clicks', 'Conversions'], title='Performance Metrics Over Time')
        st.plotly_chart(fig)

        if st.button("Generate AI Analysis and Recommendations"):
            st.info("Generating analysis and recommendations. This may take a moment...")

            prompt = f"""
            Analyze the performance of the campaign based on the following metrics:
            - Total Impressions: {data['Impressions'].sum():,}
            - Total Clicks: {data['Clicks'].sum():,}
            - Total Conversions: {data['Conversions'].sum():,}
            - Total Spend: ₹{usd_to_inr(data['Spend'].sum()):,.2f}
            - Average CTR: {(data['Clicks'].sum() / data['Impressions'].sum() * 100):.2f}%
            - Average Conversion Rate: {(data['Conversions'].sum() / data['Clicks'].sum() * 100):.2f}%
            - Average CPC: ₹{usd_to_inr(data['Spend'].sum() / data['Clicks'].sum()):.2f}

            Provide the following in JSON format:
            {{
                "performance_summary": "Brief summary of overall performance",
                "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
                "areas_for_improvement": ["Area 1", "Area 2"],
                "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],
                "predicted_performance": {{
                    "Impressions": 100000,
                    "Clicks": 5000,
                    "Conversions": 250,
                    "Spend": 200000
                }}
            }}
            """

            analysis_data = generate_ai_content(prompt)

            if "error" in analysis_data:
                st.error(analysis_data["error"])
            else:
                st.subheader("AI-Generated Analysis")
                st.write(analysis_data["performance_summary"])

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Key Insights")
                    for insight in analysis_data["key_insights"]:
                        st.write(f"- {insight}")

                    st.subheader("Areas for Improvement")
                    for area in analysis_data["areas_for_improvement"]:
                        st.write(f"- {area}")

                with col2:
                    st.subheader("Recommendations for Optimization")
                    for rec in analysis_data["recommendations"]:
                        st.write(f"- {rec}")

                    st.subheader("Predicted Performance for Next Month")
                    predicted_df = pd.DataFrame(list(analysis_data["predicted_performance"].items()),
                                                columns=['Metric', 'Value'])
                    fig = px.bar(predicted_df, x='Metric', y='Value', title='Predicted Performance')
                    st.plotly_chart(fig)

        if st.button("Generate Performance Report"):
            report = f"""
            # Campaign Performance Report

            ## Executive Summary

            - Total Impressions: {data['Impressions'].sum():,}
            - Total Clicks: {data['Clicks'].sum():,}
            - Total Conversions: {data['Conversions'].sum():,}
            - Total Spend: ₹{usd_to_inr(data['Spend'].sum()):,.2f}
            - Average CTR: {(data['Clicks'].sum() / data['Impressions'].sum() * 100):.2f}%
            - Average Conversion Rate: {(data['Conversions'].sum() / data['Clicks'].sum() * 100):.2f}%
            - Average CPC: ₹{usd_to_inr(data['Spend'].sum() / data['Clicks'].sum()):.2f}

            ## Performance Trends

            [Include performance trends graph]

            ## Key Insights

            1. [Insert key insight 1]
            2. [Insert key insight 2]
            3. [Insert key insight 3]

            ## Areas for Improvement

            1. [Insert area for improvement 1]
            2. [Insert area for improvement 2]

            ## Recommendations

            1. [Insert recommendation 1]
            2. [Insert recommendation 2]
            3. [Insert recommendation 3]

            ## Next Steps

            1. Review and implement optimization recommendations
            2. Set up A/B tests for underperforming ad variations
            3. Adjust targeting parameters based on top-performing segments
            4. Schedule follow-up performance review in 2 weeks
            """

            st.download_button(
                label="Download Full Report",
                data=report,
                file_name=f"campaign_performance_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

# Main function
def main():
    if not st.session_state.api_key_validated:
        api_key_input()
    else:
        decrypted_key = decrypt_api_key(st.session_state.api_key)
        genai.configure(api_key=decrypted_key)
        main_app()

if __name__ == "__main__":
    main()