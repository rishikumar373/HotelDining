import streamlit as st
from datetime import date
import pandas as pd
import random
import joblib
import numpy as np
from pymongo import MongoClient
import bcrypt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from sklearn.preprocessing import LabelEncoder

# MongoDB Connection
client = MongoClient("mongodb+srv://Data_base:test@cluster1.l2qtf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")
db = client["info_asign"]
bookings_collection = db["new_bookings"]
users_collection = db["users"]

# Streamlit App
st.set_page_config(page_title="RK Paradise Hotel Booking", page_icon="🏨", layout="centered")

# Email Notification Function
def send_email(recipient, subject, body, banner_path=None):
    sender_email = "rishikumar45628@gmail.com"
    sender_password = "ajyt zwtd vtmz wamx"
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Attach banner image
    if banner_path:
        with open(banner_path, 'rb') as img:
            mime_img = MIMEImage(img.read())
            mime_img.add_header('Content-ID', '<banner>')
            msg.attach(mime_img)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()
    except Exception as e:
        st.error(f"Error sending email: {e}")

# User Authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.customer_id = None
    st.session_state.username = ""
    st.session_state.loyalty_points = 0

if not st.session_state.logged_in:
    st.title("🏨 RK Paradise Login/Signup")
    auth_choice = st.radio("Choose Action", ["Login", "Sign Up"])

    email = st.text_input("Email")
    username = st.text_input("Username") if auth_choice == "Sign Up" else ""
    password = st.text_input("Password", type="password")

    if auth_choice == "Sign Up":
        if st.button("Sign Up"):
            if email and password and username:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                if users_collection.find_one({"email": email}):
                    st.error("User already exists.")
                else:
                    customer_id = random.randint(10001, 99999)
                    users_collection.insert_one({"email": email, "username": username, "password": hashed_password, "customer_id": customer_id, "loyalty_points": 0})
                    st.success("Account created successfully! Please log in.")
            else:
                st.error("Please provide email, username, and password.")

    elif auth_choice == "Login":
        if st.button("Login"):
            if email and password:
                user = users_collection.find_one({"email": email})
                if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.customer_id = user.get("customer_id")
                    st.session_state.username = user.get("username")
                    st.session_state.loyalty_points = user.get("loyalty_points", 0)
                    st.success(f"Logged in successfully as {st.session_state.username}!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
            else:
                st.error("Please provide both email and password.")

if st.session_state.logged_in:
    st.title("🏨 RK Paradise Booking Form")

    # Display logged-in user info
    st.write(f"You are logged in as **{st.session_state.username}** with **{st.session_state.loyalty_points}** loyalty points.")

    # Auto-fetch Customer ID
    customer_id = st.session_state.customer_id
    st.write(f"Your Customer ID: {customer_id}")

    # User Inputs
    name = st.text_input("Enter your name", "")
    check_in_date = st.date_input("Check-in Date", min_value=date.today())
    check_out_date = st.date_input("Check-out Date", min_value=check_in_date)
    age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
    stayers = st.number_input("How many stayers in total?", min_value=1, max_value=3, step=1)
    cuisine_options = ["South Indian", "North Indian", "Multi"]
    preferred_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
    booking_options = ["Yes", "No"]
    preferred_booking = st.selectbox("Do you want to book through points?", booking_options)
    special_requests = st.text_area("Any Special Requests? (Optional)", "")

    # Submit Button
    if st.button("Submit Booking"):
        if name and customer_id:
            # Prepare booking data
            booking_data = {
                'customer_id': int(customer_id),
                'Preferred Cusine': preferred_cuisine,
                'age': age,
                'check_in_date': pd.to_datetime(check_in_date),
                'check_out_date': pd.to_datetime(check_out_date),
                'booked_through_points': 1 if preferred_booking == 'Yes' else 0,
                'number_of_stayers': stayers
            }

            # Feature Engineering
            booking_data['check_in_day'] = booking_data['check_in_date'].dayofweek
            booking_data['check_out_day'] = booking_data['check_out_date'].dayofweek
            booking_data['check_in_month'] = booking_data['check_in_date'].month
            booking_data['check_out_month'] = booking_data['check_out_date'].month
            booking_data['stay_duration'] = (booking_data['check_out_date'] - booking_data['check_in_date']).days

            # Insert into MongoDB
            bookings_collection.insert_one(booking_data)

            # Update Loyalty Points
            if preferred_booking == 'Yes':
                st.session_state.loyalty_points -= 10
            else:
                st.session_state.loyalty_points += 10

            users_collection.update_one(
                {"email": st.session_state.user_email},
                {"$set": {"loyalty_points": st.session_state.loyalty_points}}
            )

            # Load Model & Label Encoder
            model = joblib.load('xgb_best_model.pkl')
            label_encoder = joblib.load('label_encoder.pkl')

            # Align booking_df columns with model features
            booking_df = pd.DataFrame([booking_data])

            # Encode 'Preferred Cusine'
            le = LabelEncoder()
            booking_df['Preferred Cusine'] = le.fit_transform(booking_df['Preferred Cusine'])

            model_features = model.get_booster().feature_names
            for col in model_features:
                if col not in booking_df.columns:
                    booking_df[col] = 0
            booking_df = booking_df[model_features]

            # Prediction
            X = booking_df
            y_pred_prob = model.predict_proba(X)
            dish_names = label_encoder.classes_
            top_3_indices = np.argsort(-y_pred_prob, axis=1)[:, :3]
            top_3_dishes = dish_names[top_3_indices]

            # Generate Coupon Code
            coupon_code = f"RK{random.randint(1000, 9999)}"

            # Display Booking Confirmation
            st.success(f"✅ Booking Confirmed for {name} at RK Paradise (Customer ID: {customer_id})!")
            st.write(f"**Check-in:** {check_in_date}")
            st.write(f"**Check-out:** {check_out_date}")
            st.write(f"**Age:** {age}")
            st.write(f"**Preferred Cusine:** {preferred_cuisine}")
            if special_requests:
                st.write(f"**Special Requests:** {special_requests}")

            # Display Coupon Code
            st.write(f"🎉 Your Coupon Code and Booking details has been Mailed")

            # Recommended Dishes
            st.write("### Recommended Dishes:")
            st.write(f"1. {top_3_dishes[0][0]}")
            st.write(f"2. {top_3_dishes[0][1]}")
            st.write(f"3. {top_3_dishes[0][2]}")

            # Send Email with Banner
            email_body = f"Dear {name},\n\nYour booking at RK Paradise has been confirmed!\n\nCheck-in: {check_in_date}\nCheck-out: {check_out_date}\nPreferred Cuisine: {preferred_cuisine}\n\nRecommended Dishes:\n1. {top_3_dishes[0][0]}\n2. {top_3_dishes[0][1]}\n3. {top_3_dishes[0][2]}\n\nYour Coupon Code: {coupon_code}\n\nThank you for choosing RK Paradise!"
            send_email(st.session_state.user_email, "RK Paradise Booking Confirmation & Coupon Code", email_body, banner_path="rk.jpg")
        else:
            st.warning("⚠️ Please enter your name to proceed!")
