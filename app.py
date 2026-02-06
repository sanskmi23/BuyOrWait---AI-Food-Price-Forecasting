import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# PAGE CONFIG
st.set_page_config(
    page_title="BuyOrWait",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS (Professional Dark Theme)


st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(180deg, #f8fafc, #eef2ff);
    color: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #e2e8f0;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #0f172a !important;
    font-weight: 500;
}

/* Sidebar select boxes */
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] input {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border-radius: 10px;
    border: 1px solid #cbd5e1;
    padding: 8px;
}

/* Headings */
h1 {
    color: #1e293b;
    font-weight: 900;
}
h2 {
    color: #2563eb;
    font-weight: 800;
}
h3, h4 {
    color: #0f172a;
    font-weight: 700;
}

/* Paragraph */
p {
    color: #334155;
    font-size: 16px;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

/* Recommendation / Success box */
div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid #cbd5e1;
}

/* Buttons (Quick Questions) */
.stButton>button {
    background: #16a34a;
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 10px 16px;
    border: none;
    transition: 0.2s;
}

.stButton>button:hover {
    background: #15803d;
    transform: scale(1.03);
}

/* Divider */
hr {
    border: 1px solid #e2e8f0;
}

/* Tables (if any) */
table {
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER (HTML LOGO — NOT st.image)
# --------------------------------------------------
# --------------------------------------------------
# HEADER (LOGO FIXED)
# --------------------------------------------------
col1, col2 = st.columns([1, 6])

with col1:
    st.image("assets/logo.png", width=110)

with col2:
    st.markdown("""
    <div style="display:flex; flex-direction:column; justify-content:center; height:110px;">
        <h1>BuyOrWait</h1>
        <h4>Smart Food Commodity Buying Decision System</h4>
    </div>
    """, unsafe_allow_html=True)
st.markdown("""
<p>
A <b>Green AI</b> based system that predicts vegetable prices, pulse prices, grain prices, etc. and 
finds the cheapest buying day, and helps consumers save money intelligently.
</p>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_2026.csv")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("## 🔎 Selection Panel")

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["State"].unique())
)

veg = st.sidebar.selectbox(
    "Select Commodity",
    sorted(df[df["State"] == state]["Commodity"].unique())
)

filtered = df[
    (df["State"] == state) &
    (df["Commodity"] == veg)
].sort_values("Arrival_Date")

if len(filtered) < 7:
    st.warning("Not enough historical data.")
    st.stop()

# --------------------------------------------------
# MODEL
# --------------------------------------------------
filtered["day_index"] = np.arange(len(filtered))
X = filtered[["day_index"]]
y = filtered["Modal_Price"]

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(filtered), len(filtered) + 7).reshape(-1, 1)
future_prices = model.predict(future_days)

current_price = y.iloc[-1]
predicted_price = round(future_prices[-1], 0)
pct_change = ((predicted_price - current_price) / current_price) * 100

action = "BUY NOW" if pct_change > 3 else "WAIT" if pct_change < -3 else "STABLE"
# Color for action
if action == "BUY NOW":
    action_color = "#a2ff51dc"   # Green
elif action == "WAIT":
    action_color = "#d00000"   # Red
else:
    action_color = "#f4a261"   # Yellow/Orange

# --------------------------------------------------
# BEST DAY TO BUY
# --------------------------------------------------
future_dates = [
    filtered["Arrival_Date"].iloc[-1] + timedelta(days=i+1)
    for i in range(7)
]

min_price = round(min(future_prices), 0)
best_day = future_dates[np.argmin(future_prices)]
wait_days = (best_day - filtered["Arrival_Date"].iloc[-1]).days

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
st.markdown("## 🤖 Recommendation")
# Action Color (BUY=Green, WAIT=Red, STABLE=Yellow)
if action == "BUY NOW":
    action_color = "#76c115"   # Green
elif action == "WAIT":
    action_color = "#dc2626"   # Red
else:
    action_color = "#facc15"   # Yellow
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"₹{int(current_price)}")
c2.metric("Predicted Price (7 Days)", f"₹{int(predicted_price)}")
with c3:
    st.markdown(
        f"""
        <div style="
            background-color:{action_color};
            padding:18px;
            border-radius:16px;
            text-align:center;
            color:white;
            font-size:22px;
            font-weight:bold;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        ">
            {action}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(f"**Expected price change:** `{pct_change:.2f}%`")

st.markdown("## 📅 Best Day to Buy")
st.success(
    f"""
    💰 Lowest Expected Price: ₹{int(min_price)}  
    📆 Date: {best_day.date()}  
    ⏳ Wait: {wait_days} days
    """
)
# --------------------------------------------------
# AI ASSISTANT (NO API - LIGHTWEIGHT CHATBOT)
# --------------------------------------------------
st.markdown("## 🤖 AI Assistant (Ask Questions)")

st.info("""
💡 **You can ask questions like:**
- What is the trend?
- When will the price be lowest?
- Should I buy now or wait?
- What is the predicted price after 7 days?
- What is the current price?
- How much will the price change?
""")

# Suggested Question Buttons
st.markdown("### 📌 Quick Questions")
q1, q2, q3, q4, q5, q6 = st.columns(6)

if "messages" not in st.session_state:
    st.session_state.messages = []

selected_question = None

with q1:
    if st.button("📈 Trend"):
        selected_question = "What is the trend?"
with q2:
    if st.button("💰 Lowest Price"):
        selected_question = "When will the price be lowest?"
with q3:
    if st.button("🛒 Buy/Wait"):
        selected_question = "Should I buy now or wait?"
with q4:
    if st.button("🔮 Prediction"):
        selected_question = "What is the predicted price after 7 days?"
with q5:
    if st.button("📌 Current"):
        selected_question = "What is the current price?"
with q6:
    if st.button("📊 Change %"):
        selected_question = "How much will the price change?"

# Input box
user_input = st.text_input("Type your question here:", value=selected_question if selected_question else "")

def ai_assistant(question):
    q = question.lower()

    if "trend" in q or "increase" in q or "decrease" in q:
        if pct_change > 0:
            return f"📈 The trend is increasing. Expected change is {pct_change:.2f}% in the next 7 days."
        elif pct_change < 0:
            return f"📉 The trend is decreasing. Expected change is {pct_change:.2f}% in the next 7 days."
        else:
            return "📊 The price trend is stable with minimal change."

    elif "lowest" in q or "minimum" in q or "cheap" in q or "cheapest" in q or "low" in q:
        return f"💰 The lowest predicted price is ₹{int(min_price)} on 📅 {best_day.date()}."

    elif "buy" in q or "wait" in q or "recommend" in q:
        return f"🛒 Recommendation: **{action}**.\n\nCurrent price is ₹{int(current_price)} and predicted price after 7 days is ₹{int(predicted_price)}."

    elif "predict" in q or "forecast" in q or "future" in q:
        return f"🔮 Predicted price after 7 days is ₹{int(predicted_price)}."

    elif "current" in q or "today" in q or "now" in q:
        return f"📌 Current price is ₹{int(current_price)}."

    elif "change" in q or "%" in q or "percentage" in q:
        return f"📊 Expected price change is {pct_change:.2f}% in the next 7 days."

    elif "state" in q:
        return f"📍 Selected state: **{state}**."

    elif "vegetable" in q or "commodity" in q:
        return f"🥦 Selected commodity: **{veg}**."

    else:
        return (
            "❓ I can answer:\n"
            "- Trend\n"
            "- Lowest price day\n"
            "- Buy now or wait\n"
            "- Predicted price\n"
            "- Current price"
        )

# Generate response
if user_input:
    response = ai_assistant(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("AI", response))

# Show conversation
st.markdown("### 💬 Chat History")
for sender, msg in st.session_state.messages[-10:]:
    if sender == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")

st.divider()

# --------------------------------------------------
# PRICE TREND (NOW PERFECTLY NORMAL)
# --------------------------------------------------
st.markdown("## 📈 Price Trend")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered["Arrival_Date"], filtered["Modal_Price"], label="Historical")
ax.plot(future_dates, future_prices, linestyle="--", label="Predicted")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
# SUSTAINABILITY & GREEN AI FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
    🌱 <b>Sustainability & Green AI</b><br><br>
    BuyOrWait promotes responsible consumption by reducing panic buying,
    minimizing food waste, and encouraging data-driven purchasing decisions.
    <br><br>
    The system uses lightweight predictive models that consume minimal
    computational power, aligning with the principles of Green AI —
    efficiency, transparency, and environmental responsibility.
    </div>
    """,
    unsafe_allow_html=True
)
