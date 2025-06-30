import pandas as pd, json, os, random, joblib
from textblob import TextBlob
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import resample

# ---------- CONFIG ----------
RAW_CSV = "sample.csv"
TAGGED_CSV = "tagged_interactions.csv"
NBA_JSON = "nba_outputs.json"
RESULT_CSV = "result.csv"
MODEL_FILE = "nba_channel_classifier.joblib"
NUM_CUSTOMERS = 1000

# ---------- 1. Data Preparation ----------
df = pd.read_csv(RAW_CSV, low_memory=False).drop_duplicates('tweet_id').dropna(subset=['text'])
cust = df[df['inbound'] == True]
supp = df[df['inbound'] == False]
df = pd.merge(cust, supp, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_customer', '_support'))

# ---------- 2. Feature Engineering ----------
df['customer_sentiment'] = df['text_customer'].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
df['sentiment_label'] = df['customer_sentiment'].apply(
    lambda s: 'positive' if s > 0.3 else 'negative' if s < -0.3 else 'neutral'
)

def classify(text):
    text = str(text).lower()
    keys = {
        "refund": ["refund", "money back", "charged"],
        "login issue": ["login", "password", "authentication"],
        "delivery issue": ["delivery", "package", "tracking"],
        "cancellation": ["cancel", "unsubscribe"],
        "technical issue": ["app", "bug", "crash", "error"]
    }
    for k, v in keys.items():
        if any(word in text for word in v): return k
    return "other"

df['nature_of_support_request'] = df['text_customer'].apply(classify)

# Initial rule-based channel assignment
def choose_channel(s, t):
    if s < -0.4 or t in ['refund', 'delivery issue']: return "scheduling_phone_call"
    if s > 0.3: return "email_reply"
    return "twitter_dm_reply"

df['channel'] = df.apply(lambda r: choose_channel(r['customer_sentiment'], r['nature_of_support_request']), axis=1)
df['resolved'] = df['text_customer'].str.lower().apply(
    lambda t: any(k in t for k in ['thank you', 'thanks', 'resolved', 'fixed', 'got it', 'sorted'])
)
df.to_csv(TAGGED_CSV, index=False)

# ---------- 3. Model Training ----------
train_df = df[df['resolved'] == False].copy()

# Create balanced training set
min_count = train_df['channel'].value_counts().min()
balanced_df = pd.concat([
    train_df[train_df['channel'] == c].sample(min_count, random_state=42)
    for c in train_df['channel'].unique()
])

# Feature combination
balanced_df['input'] = (
    balanced_df['text_customer'] + ' issue: ' + balanced_df['nature_of_support_request'] +
    ' sentiment: ' + balanced_df['sentiment_label']
)

X = balanced_df['input']
y = balanced_df['channel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_FILE)

# ---------- 4. NBA Generation ----------
nba_records, csv_rows = [], []
model = joblib.load(MODEL_FILE)

df_nba = train_df.drop_duplicates('author_id_customer').head(NUM_CUSTOMERS)

def gen_msg(c, t):
    return {
        "scheduling_phone_call": f"We understand your concern about {t}. Let’s resolve this quickly over a call.",
        "email_reply": f"Thank you for reaching out regarding {t}. We've sent you an email."
    }.get(c, f"We’re on it! Please DM us more details about your {t}.")

def predict_status(c, s): 
    return "resolved" if c == "email_reply" and s == "positive" else \
           "escalated" if c == "scheduling_phone_call" and s == "negative" else \
           "pending customer reply"

def make_chat(row):
    parts = []
    if pd.notna(row['text_customer']): parts.append("Customer: " + row['text_customer'])
    if pd.notna(row['text_support']): parts.append("Support_agent: " + row['text_support'])
    return "\n".join(parts)

for _, r in df_nba.iterrows():
    new_input = (
        r['text_customer'] + ' issue: ' + r['nature_of_support_request'] +
        ' sentiment: ' + r['sentiment_label']
    )
    
    ch = model.predict([new_input])[0]
    proba = model.predict_proba([new_input])[0]
    confidence = round(max(proba), 4)
    
    msg = gen_msg(ch, r['nature_of_support_request'])
    time = (datetime.utcnow() + timedelta(minutes=random.randint(5, 30))).isoformat() + "Z"
    
    reasoning = f"ML model used sentiment '{r['sentiment_label']}' and issue '{r['nature_of_support_request']}' to choose {ch}."
    status = predict_status(ch, r['sentiment_label'])
    chat = make_chat(r)

    chat_len = len(str(r['text_customer']).split())
    high_priority = (
        r['nature_of_support_request'] in ['refund', 'delivery issue'] and
        r['sentiment_label'] == 'negative'
    )

    nba_records.append({
        "customer_id": str(r['author_id_customer']),
        "channel": ch,
        "send_time": time,
        "message": msg,
        "reasoning": reasoning
    })

    csv_rows.append({
        "customer_id": str(r['author_id_customer']),
        "channel": ch,
        "send_time": time,
        "message": msg,
        "reasoning": reasoning,
        "chat_log": chat,
        "issue_status": status,
        "confidence_score": confidence,
        "chat_length": chat_len,
        "high_priority": high_priority
    })

# ---------- 5. Save Results ----------
with open(NBA_JSON, 'w') as f:
    json.dump(nba_records, f, indent=2)

df_result = pd.DataFrame(csv_rows)
df_result = df_result.sort_values(by='send_time')
df_result.to_csv(RESULT_CSV, index=False)

print(f"\n NBA records saved to: {NBA_JSON}")
print(f" Full results saved to: {RESULT_CSV}")
