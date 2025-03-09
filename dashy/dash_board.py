import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from pymongo import MongoClient

# Page Config
st.set_page_config(page_title="🏨 Hotel, Dining & Reviews Dashboard", layout="wide")

# MongoDB Connection
MONGO_URI = 'mongodb+srv://Data_base:test@cluster1.l2qtf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1'
DB_NAME = "info_asign"
COLL_BOOKING = "booking"
COLL_DINING = "test"
COLL_REVIEWS = "reviews"

@st.cache_data
def load_data(collection_name):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[collection_name]
    data = list(collection.find({}, {'_id': 0}))
    return pd.DataFrame(data) if data else pd.DataFrame()

# Load Data
df_booking = load_data(COLL_BOOKING)
df_dining = load_data(COLL_DINING)
df_reviews = load_data(COLL_REVIEWS)

st.title("🏨 Hotel, Dining & Reviews Dashboard")
st.write(f"✅ Loaded {df_booking.shape[0]} bookings, {df_dining.shape[0]} dining records, and {df_reviews.shape[0]} reviews from MongoDB")

# Tabs for Dashboards
tab1, tab2, tab3 = st.tabs(["🏨 Hotel Booking Insights", "🍽️ Dining Insights", "📝 Reviews Analysis"])

# ----- HOTEL BOOKING INSIGHTS -----
with tab1:
    st.sidebar.header("🏨 Booking Filters")
    if not df_booking.empty:
        df_booking['check_in_date'] = pd.to_datetime(df_booking['check_in_date'], errors='coerce')
        df_booking['check_out_date'] = pd.to_datetime(df_booking['check_out_date'], errors='coerce')
        df_booking['stay_duration'] = (df_booking['check_out_date'] - df_booking['check_in_date']).dt.days

        # Filters
        date_range = st.sidebar.date_input(
            "📅 Filter by Check-In Date Range",
            [df_booking['check_in_date'].min(), df_booking['check_in_date'].max()]
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_booking = df_booking[(df_booking['check_in_date'] >= pd.Timestamp(start_date)) &
                                    (df_booking['check_in_date'] <= pd.Timestamp(end_date))]

        customer_ids = st.sidebar.multiselect(
            "👤 Select Customer IDs",
            df_booking['customer_id'].unique(),
            default=df_booking['customer_id'].unique()[:5]
        )
        if customer_ids:
            df_booking = df_booking[df_booking['customer_id'].isin(customer_ids)]

        # Stay Duration Distribution
        fig_stay_duration = px.histogram(df_booking, x="stay_duration", nbins=10, title="Stay Duration Distribution")
        st.plotly_chart(fig_stay_duration, use_container_width=True)
        if not df_booking['stay_duration'].empty:
            avg_stay = df_booking['stay_duration'].mean()
            max_stay = df_booking['stay_duration'].max()
            st.write(f"🛏️ **Average stay duration: {avg_stay:.2f} days. The longest stay is {max_stay} days. Most customers stay for 1-3 days.**")

        # Point-Based Bookings
        if "booked_through_points" in df_booking.columns:
            points_usage = df_booking["booked_through_points"].value_counts().reset_index()
            points_usage.columns = ["Points Used", "Count"]
            fig_points_usage = px.bar(points_usage, x="Points Used", y="Count", title="Bookings Made with Points")
            st.plotly_chart(fig_points_usage, use_container_width=True)
            max_points_used = points_usage.loc[points_usage['Count'].idxmax(), "Points Used"]
            st.write(f"💡 **Most bookings are made with{'out' if max_points_used == 0 else ''} points usage.**")

        # Check-Ins Over Time
        checkin_trend = df_booking.groupby(df_booking['check_in_date'].dt.date).size().reset_index(name="Count")
        fig_checkin_trend = px.line(checkin_trend, x="check_in_date", y="Count", title="Check-Ins Over Time")
        st.plotly_chart(fig_checkin_trend, use_container_width=True)
        if not checkin_trend['Count'].empty:
            peak_checkin = checkin_trend.loc[checkin_trend['Count'].idxmax()]
            st.write(f"📈 **Peak check-in date: {peak_checkin['check_in_date']} with {peak_checkin['Count']} bookings.**")

# ----- DINING INSIGHTS -----
with tab2:
    st.sidebar.header("🍽️ Dining Filters")
    if not df_dining.empty:
        df_dining['order_time'] = pd.to_datetime(df_dining['order_time'], errors='coerce')
        df_dining['day_of_week'] = df_dining['order_time'].dt.day_name()

        # Filters
        cuisines = st.sidebar.multiselect(
            "🍲 Select Preferred Cuisines",
            df_dining["Preferred Cusine"].dropna().unique(),
            default=df_dining["Preferred Cusine"].dropna().unique()[:3]
        )
        if cuisines:
            df_dining = df_dining[df_dining["Preferred Cusine"].isin(cuisines)]

        dishes = st.sidebar.multiselect(
            "🍽️ Select Dishes",
            df_dining["dish"].dropna().unique()
        )
        if dishes:
            df_dining = df_dining[df_dining["dish"].isin(dishes)]

        min_price, max_price = df_dining["price_for_1"].min(), df_dining["price_for_1"].max()
        price_range = st.sidebar.slider("💰 Select Price Range", float(min_price), float(max_price), (float(min_price), float(max_price)))
        df_dining = df_dining[(df_dining["price_for_1"] >= price_range[0]) & (df_dining["price_for_1"] <= price_range[1])]

        # Cuisine Preferences by Age (Heatmap)
        heatmap_data = pd.pivot_table(df_dining, values="Qty", index="age", columns="Preferred Cusine", aggfunc='sum').fillna(0)
        fig_heatmap = px.imshow(heatmap_data, color_continuous_scale="Blues", title="Cuisine Preferences by Age")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        if not heatmap_data.empty:
            st.write("👨‍🍳 **Heatmap shows cuisine preferences across age groups for better insights.**")

        # Price Variation by Dish
        fig_price_variation = px.box(df_dining, x="dish", y="price_for_1", title="Price Variation by Dish", color="dish")
        st.plotly_chart(fig_price_variation, use_container_width=True)
        most_expensive_dish = df_dining.loc[df_dining["price_for_1"].idxmax(), "dish"]
        st.write(f"💸 **Most expensive dish: {most_expensive_dish}.**")

        # Revenue Over Time
        revenue_trend = df_dining.groupby(df_dining['order_time'].dt.date)['price_for_1'].sum().reset_index()
        revenue_trend.columns = ["Date", "Revenue"]
        fig_revenue = px.line(revenue_trend, x="Date", y="Revenue", title="Revenue Trend Over Time")
        st.plotly_chart(fig_revenue, use_container_width=True)
        if not revenue_trend.empty:
            peak_revenue_date = revenue_trend.loc[revenue_trend["Revenue"].idxmax(), "Date"]
            peak_revenue = revenue_trend.loc[revenue_trend["Revenue"].idxmax(), "Revenue"]
            st.write(f"💰 **Peak revenue: ₹{peak_revenue:.2f} on {peak_revenue_date}.**")
        # 3D Graph: Dishes by Revenue and Day of Week
        fig_3d = px.scatter_3d(
            df_dining,
            x="day_of_week",
            y="dish",
            z="price_for_1",
            color="dish",
            title="3D Visualization of Dishes Revenue by Day",
            labels={"day_of_week": "Day of the Week", "dish": "Dish", "price_for_1": "Price"}
        )
        fig_3d.update_traces(marker=dict(size=8))  # Adjust marker size for better visibility
        st.plotly_chart(fig_3d, use_container_width=True)
        st.write("🎇 **This 3D visualization illustrates the price variations of dishes based on the day of the week, highlighting popular days and expensive dishes.**")

# ----- REVIEWS ANALYSIS -----
with tab3:
    st.sidebar.header("📝 Reviews Filters")
    if not df_reviews.empty:
        # Sentiment Analysis
        def get_sentiment(review):
            if pd.isna(review):
                return "Neutral"
            polarity = TextBlob(review).sentiment.polarity
            return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

        df_reviews["Sentiment"] = df_reviews["Review"].apply(get_sentiment)
        sentiment_counts = df_reviews["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig_sentiment = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution")
        st.plotly_chart(fig_sentiment, use_container_width=True)
        if not sentiment_counts.empty:
            dominant_sentiment = sentiment_counts.loc[sentiment_counts['Count'].idxmax(), "Sentiment"]
            st.write(f"📝 **Dominant sentiment: {dominant_sentiment}.**")

        # Rating Distribution
        fig_rating_distribution = px.histogram(
            df_reviews,
            x="Rating",
            nbins=10,
            title="Rating Distribution",
            color="Rating"
        )
        st.plotly_chart(fig_rating_distribution, use_container_width=True)
        if not df_reviews["Rating"].empty:
            avg_rating = df_reviews["Rating"].mean()
            max_rating = df_reviews["Rating"].max()
            min_rating = df_reviews["Rating"].min()
            st.write(f"⭐ **Average rating: {avg_rating:.2f}. Ratings range from {min_rating} to {max_rating}.**")
