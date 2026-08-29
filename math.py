import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide"
)

# Title and Header
st.title("🚢 Titanic Survival Analysis & ML Prediction")
st.markdown("### Mini Project 1 - Data Cleaning & Machine Learning")

# Primary remote URL
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(DATA_URL)

    # Indian Names Mapping (Strictly Indian Names)
    male_names = [
        "Aarav Sharma",
        "Arjun Mehta",
        "Rohan Gupta",
        "Aditya Patel",
        "Rahul Malhotra",
        "Vikram Rao",
        "Karan Shah",
        "Manish Kumar",
        "Siddharth Jain",
        "Varun Bansal",
        "Devansh Tiwari",
        "Harsh Vardhan",
        "Nikhil Verma",
        "Amitabh Sen",
        "Pranav Joshi",
        "Rajat Chauhan",
        "Sameer Kulkarni",
        "Yash Singhania",
        "Ayush Pandey",
        "Gaurav Saxena",
    ]

    female_names = [
        "Ananya Verma",
        "Diya Kapoor",
        "Priya Singh",
        "Sneha Joshi",
        "Isha Agarwal",
        "Neha Nair",
        "Pooja Mishra",
        "Riya Saxena",
        "Kavya Reddy",
        "Anjali Desai",
        "Tanvi Choudhary",
        "Meera Nambiar",
        "Shreya Ghosh",
        "Pallavi Iyer",
        "Divya Hegde",
        "Roshni Bhatia",
        "Simran Kaur",
        "Kritika Roy",
        "Swati Deshmukh",
        "Payal Mukherjee",
    ]

    m_idx, f_idx = 0, 0
    names = []
    for sex in df["Sex"]:
        if str(sex).strip().lower() == "female":
            names.append(female_names[f_idx % len(female_names)])
            f_idx += 1
        else:
            names.append(male_names[m_idx % len(male_names)])
            m_idx += 1
    df["Name"] = names

    # Data Cleaning & Missing Value Treatment
    df["Age"] = df["Age"].fillna(round(df["Age"].mean(), 2))
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    df["Fare"] = df["Fare"].fillna(0)
    df["Survival_Status"] = df["Survived"].map(
        {1: "Survived", 0: "Not Survived"}
    )

    return df


df = load_and_clean_data()

# 1. Top Summary KPI Cards
total_passengers = len(df)
survived_count = int((df["Survived"] == 1).sum())
not_survived_count = int((df["Survived"] == 0).sum())
survival_rate = round((survived_count / total_passengers) * 100, 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Passengers", total_passengers)
col2.metric("Survived", survived_count)
col3.metric("Not Survived", not_survived_count)
col4.metric("Survival Rate", f"{survival_rate}%")

st.markdown("---")

# 2. Interactive Charts & Graphs
st.subheader("📊 Exploratory Data Analysis")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig_gender = px.histogram(
        df,
        x="Sex",
        color="Survival_Status",
        barmode="group",
        title="Survival Count by Gender",
        color_discrete_map={
            "Survived": "#2ecc71",
            "Not Survived": "#e74c3c",
        },
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with row1_col2:
    fig_pclass = px.histogram(
        df,
        x="Pclass",
        color="Survival_Status",
        barmode="group",
        title="Survival Count by Passenger Class",
        color_discrete_map={
            "Survived": "#2ecc71",
            "Not Survived": "#e74c3c",
        },
    )
    st.plotly_chart(fig_pclass, use_container_width=True)

fig_age = px.histogram(
    df,
    x="Age",
    color="Survival_Status",
    nbins=30,
    title="Age Distribution vs Survival",
    color_discrete_map={"Survived": "#2ecc71", "Not Survived": "#e74c3c"},
)
st.plotly_chart(fig_age, use_container_width=True)

st.markdown("---")

# 3. Live ML Prediction Tool
st.subheader("🤖 Live Passenger Survival Predictor")

# Train Logistic Regression model
X = df[["Pclass", "Age"]].copy()
X["Sex_Code"] = df["Sex"].map({"male": 0, "female": 1})
y = df["Survived"]

model = LogisticRegression()
model.fit(X[["Sex_Code", "Pclass", "Age"]], y)

pred_c1, pred_c2, pred_c3 = st.columns(3)

with pred_c1:
    input_sex = st.selectbox("Select Gender", ["male", "female"])
with pred_c2:
    input_pclass = st.selectbox("Select Passenger Class (Pclass)", [1, 2, 3])
with pred_c3:
    input_age = st.slider("Select Age", min_value=1, max_value=80, value=25)

if st.button("Predict Survival"):
    sex_val = 1 if input_sex == "female" else 0
    pred = model.predict([[sex_val, input_pclass, input_age]])[0]
    prob = model.predict_proba([[sex_val, input_pclass, input_age]])[0][1]

    if pred == 1:
        st.success(
            f"🎉 **Passenger is predicted to SURVIVE!** (Survival Probability: {round(prob * 100, 2)}%)"
        )
    else:
        st.error(
            f"⚠️ **Passenger is predicted NOT to survive.** (Survival Probability: {round(prob * 100, 2)}%)"
        )

st.markdown("---")

# 4. Cleaned Indian Records Table
st.subheader("📋 Cleaned Indian Passenger Records")
st.dataframe(
    df[
        [
            "PassengerId",
            "Name",
            "Sex",
            "Age",
            "Pclass",
            "Fare",
            "Survival_Status",
        ]
    ],
    use_container_width=True,
)
