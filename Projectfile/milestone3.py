import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
from datetime import datetime
from together import Together
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Hotel Review Summarizer", page_icon="🏨", layout="wide")

os.environ["TOGETHER_API_KEY"] = "4b35bf117de47f85e287445a758437a052494b4235ab5f7c28c9182e9ef4e06b"
PINECONE_API_KEY = 'pcsk_5Q9W8Y_TSK22TSphTofcH6KjpYv7dqDTVUQ83XLbfMFCfaT4uVv8Xy2a3bP8tEEpuitbkD'

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("hotel")
MONGO_URI = 'mongodb+srv://Data_base:test@cluster1.l2qtf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1'
df = pd.read_excel("D:\\reviews_data.xlsx")
COLL_REVIEWS = "reviews"



with st.sidebar:
    st.title("🔍 Search & Filter")
    search_query = st.text_area("Search for Reviews (Sentence/Free Text)", "food quality")
    rating_filter = st.slider("Minimum Rating", 1, 10, 5)
    date_range = st.date_input("Review Date Range", [datetime(2024, 1, 1), datetime(2024, 12, 31)])

start_date = int(date_range[0].strftime("%Y%m%d"))
end_date = int(date_range[1].strftime("%Y%m%d"))

embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
query_embedding = embeddings.embed_query(search_query)

with st.spinner("🔎 Searching reviews..."):
    search_results = index.query(
        vector=query_embedding,
        top_k=15,
        include_metadata=True,
        filter={
            "Rating": {"$gte": rating_filter},
            "review_date": {"$gte": start_date, "$lte": end_date}
        }
    )

matches = search_results.get("matches", [])
matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
matched_reviews_df = df[df["review_id"].isin(matched_ids)]

st.title("🏨 Hotel Review Summarizer")

if matched_reviews_df.empty:
    st.warning("⚠️ No reviews found matching your criteria.")
else:
    st.success(f"✅ Found {len(matched_reviews_df)} matching reviews.")

    for idx, row in matched_reviews_df.iterrows():
        st.markdown(f"""
        <div style="padding: 15px; margin: 10px 0; border-radius: 12px; background: linear-gradient(135deg, #2c3e50, #4ca1af); color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
            <h4>Review #{row['review_id']}</h4>
            <p>⭐ Rating: <b>{row['Rating']}</b></p>
            <p>📅 Date: <b>{row['review_date']}</b></p>
            <p>🗒️ {row['Review']}</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("📄 Generate Summary"):
        with st.spinner("🧠 Generating summary..."):
            client = Together()

            prompt_text = (
                f"Read the following hotel reviews and summarize only opinions related to '{search_query}'. "
                "Classify the summarized points into three sections: 'Positive', 'Negative', and 'Neutral'. "
                "Strictly follow this format:\n\n"
                "Positive:\n- Point 1\n- Point 2\n\nNegative:\n- Point 1\n- Point 2\n\nNeutral:\n- Point 1\n- Point 2\n\n"
                f"{matched_reviews_df['Review'].str.cat(sep=' ')}"
            )

            response = client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[{"role": "user", "content": prompt_text}]
            )

            summary_text = response.choices[0].message.content.strip()

            if "Positive:" in summary_text and "Negative:" in summary_text and "Neutral:" in summary_text:
                positive_part = summary_text.split("Positive:")[1].split("Negative:")[0].strip()
                negative_part = summary_text.split("Negative:")[1].split("Neutral:")[0].strip()
                neutral_part = summary_text.split("Neutral:")[1].strip()

                st.markdown("### 📚 Summary (Filtered to Your Query)")

                st.markdown("#### ✅ Positive Points")
                st.markdown(f"```\n{positive_part}\n```")

                st.markdown("#### ❌ Negative Points")
                st.markdown(f"```\n{negative_part}\n```")

                st.markdown("#### ⚪ Neutral Points")
                st.markdown(f"```\n{neutral_part}\n```")

                positive_count = positive_part.count("- ")
                negative_count = negative_part.count("- ")
                neutral_count = neutral_part.count("- ")

                labels = ["Positive", "Negative", "Neutral"]
                counts = [positive_count, negative_count, neutral_count]
                colors = ["#4CAF50", "#F44336", "#FFC107"]

                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
                ax.axis("equal")
                st.pyplot(fig)

                def create_pdf():
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(200, 10, "Hotel Review Summary", ln=True, align="C")
                    pdf.ln(5)

                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 8, f"Search Query: {search_query}", ln=True)

                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 8, "Positive Points:", ln=True)
                    pdf.set_font("Arial", size=12)
                    for line in positive_part.split("\n"):
                        pdf.cell(200, 6, line, ln=True)

                    pdf.ln(3)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 8, "Negative Points:", ln=True)
                    pdf.set_font("Arial", size=12)
                    for line in negative_part.split("\n"):
                        pdf.cell(200, 6, line, ln=True)

                    pdf.ln(3)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 8, "Neutral Points:", ln=True)
                    pdf.set_font("Arial", size=12)
                    for line in neutral_part.split("\n"):
                        pdf.cell(200, 6, line, ln=True)

                    pdf_bytes = pdf.output(dest='S').encode('latin1')
                    return pdf_bytes

                st.download_button(
                    "📥 Download Summary PDF",
                    data=create_pdf(),
                    file_name="Hotel_Review_Summary.pdf",
                    mime="application/pdf"
                )

            else:
                st.error("❌ Could not parse the summary properly. Please try again.")

st.markdown("---")
st.markdown('<div style="text-align: center;">Built with ❤️| 2025</div>', unsafe_allow_html=True)
