import streamlit as st

# ✅ Page config - must be first Streamlit command
st.set_page_config(page_title="Submit Review", page_icon="📝")

import pandas as pd
import os
import uuid
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
from datetime import datetime

# ✅ Set API Keys (for production, move to environment variables)
os.environ["TOGETHER_API_KEY"] = "4b35bf117de47f85e287445a758437a052494b4235ab5f7c28c9182e9ef4e06b"
PINECONE_API_KEY = 'pcsk_5Q9W8Y_TSK22TSphTofcH6KjpYv7dqDTVUQ83XLbfMFCfaT4uVv8Xy2a3bP8tEEpuitbkD'

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("hotel")

# ✅ Initialize Together Embeddings
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# ✅ Path to Excel file
file_path = "D:\\reviews_data.xlsx"

# ✅ Load existing reviews data
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
else:
    df = pd.DataFrame(columns=[
        "review_id", "display_id", "customer_id", "Rating", "Review",
        "review_date", "review_date_numeric", "stay_status"
    ])

# ✅ Function to get next display_id (sequential 5-digit)
def get_next_display_id():
    if df.empty:
        return 10001  # Start from 10001
    else:
        return int(df["display_id"].max()) + 1

# ✅ Streamlit App Title
st.title("📝 Submit Your Hotel Review")

# ✅ Review Submission Form
with st.form("review_form"):
    customer_id = st.text_input("Customer ID", placeholder="Enter your Customer ID")
    rating = st.slider("Rating (1-10)", 1, 10, 5)
    review_text = st.text_area("Your Review", placeholder="Write your review here...")
    review_date = st.date_input("Review Date", datetime.now())

    # ✅ New field - Customer stay status
    stay_status = st.radio(
        "Are you currently staying at the hotel or have you checked out?",
        ["Currently Staying", "Checked Out"]
    )

    submitted = st.form_submit_button("Submit Review")

# ✅ Handle form submission
if submitted:
    if not review_text.strip():
        st.error("⚠️ Review cannot be empty.")
    elif not customer_id.strip():
        st.error("⚠️ Customer ID cannot be empty.")
    else:
        # ✅ Generate review_id and display_id
        new_review_id = str(uuid.uuid4())
        new_display_id = get_next_display_id()

        # ✅ Convert review date to numeric format
        review_date_numeric = int(review_date.strftime("%Y%m%d"))

        # ✅ Create new review record
        new_review = {
            "review_id": new_review_id,
            "display_id": new_display_id,
            "customer_id": customer_id,
            "Rating": rating,
            "Review": review_text,
            "review_date": review_date.strftime("%Y-%m-%d"),
            "review_date_numeric": review_date_numeric,
            "stay_status": stay_status  # ✅ Add stay status
        }

        # ✅ Save to DataFrame and Excel
        new_df = pd.DataFrame([new_review])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(file_path, index=False)

        st.success(f"✅ Review Submitted Successfully (Review ID: {new_display_id})")

        # ✅ Embed review and upload to Pinecone
        with st.spinner("🔗 Uploading review to vector database..."):
            review_embedding = embeddings.embed_query(review_text)

            index.upsert(vectors=[
                (
                    new_review_id,
                    review_embedding,
                    {
                        "review_id": new_review_id,
                        "display_id": new_display_id,
                        "customer_id": customer_id,
                        "Rating": rating,
                        "review_date": review_date_numeric,
                        "stay_status": stay_status  # ✅ Add stay status to Pinecone metadata
                    }
                )
            ])

        st.success("✅ Review successfully added to Pinecone Vector Database!")

# ✅ Footer
st.markdown("---")
st.markdown('<div style="text-align: center;">Built with ❤️ using Streamlit | 2025</div>', unsafe_allow_html=True)
