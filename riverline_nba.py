import pandas as pd, json, os, random
from textblob import TextBlob
from datetime import datetime, timedelta

# ---------- CONFIG ----------
RAW_CSV = "twcs.csv"
VISUAL_DIR = "visuals"
INTERACTIONS_CSV = "interaction_table.csv"
TAGGED_CSV = "tagged_interactions.csv"
NBA_JSON = "nba_outputs.json"
RESULT_CSV = "result.csv"
NUM_CUSTOMERS = 1000
os.makedirs(VISUAL_DIR, exist_ok=True)

# ---------- 1. Data Preparation ----------
df = pd.read_csv(RAW_CSV, low_memory=False).drop_duplicates('tweet_id').dropna(subset=['text'])
cust = df[df['inbound'] == True]
supp = df[df['inbound'] == False]
df = pd.merge(cust, supp, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_customer', '_support'))
df.to_csv(INTERACTIONS_CSV, index=False)

# ---------- 2. Feature Engineering ----------
df['customer_sentiment'] = df['text_customer'].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
df['sentiment_label'] = df['customer_sentiment'].apply(lambda s: 'positive' if s > 0.3 else 'negative' if s < -0.3 else 'neutral')

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
df['resolved'] = df['text_customer'].str.lower().apply(lambda t: any(k in t for k in ['thank you', 'thanks', 'resolved', 'fixed', 'got it', 'sorted']))
df.to_csv(TAGGED_CSV, index=False)

# ---------- 3. NBA Generation ----------
df = df[df['resolved'] == False].drop_duplicates('author_id_customer').head(NUM_CUSTOMERS)

def choose_channel(s, t):
    if s < -0.4 or t in ['refund', 'delivery issue']: return "scheduling_phone_call"
    if s > 0.3: return "email_reply"
    return "twitter_dm_reply"

def gen_msg(c, t):
    return {
        "scheduling_phone_call": f"We understand your concern about {t}. Let’s resolve this quickly over a call.",
        "email_reply": f"Thank you for reaching out regarding {t}. We've sent you an email."
    }.get(c, f"We’re on it! Please DM us more details about your {t}.")

def predict_status(c, s): 
    return "resolved" if c == "email_reply" and s == "positive" else "escalated" if c == "scheduling_phone_call" and s == "negative" else "pending customer reply"

def make_chat(row):
    parts = []
    if pd.notna(row['text_customer']): parts.append("Customer: " + row['text_customer'])
    if pd.notna(row['text_support']): parts.append("Support_agent: " + row['text_support'])
    return "\n".join(parts)

nba_records, csv_rows = [], []

for _, r in df.iterrows():
    ch = choose_channel(r['customer_sentiment'], r['nature_of_support_request'])
    msg = gen_msg(ch, r['nature_of_support_request'])
    time = (datetime.utcnow() + timedelta(minutes=random.randint(5, 30))).isoformat() + "Z"
    reasoning = f"Based on {r['sentiment_label']} sentiment and support topic '{r['nature_of_support_request']}', {ch} is best."
    status = predict_status(ch, r['sentiment_label'])
    chat = make_chat(r)

    nba_records.append({
        "customer_id": str(r['author_id_customer']),
        "channel": ch, "send_time": time,
        "message": msg, "reasoning": reasoning
    })

    csv_rows.append({
        "customer_id": str(r['author_id_customer']),
        "channel": ch, "send_time": time,
        "message": msg, "reasoning": reasoning,
        "chat_log": chat, "issue_status": status
    })

# ---------- 4. Save Outputs ----------
with open(NBA_JSON, 'w') as f: json.dump(nba_records, f, indent=2)
pd.DataFrame(csv_rows).to_csv(RESULT_CSV, index=False)

print(f"NBA records: {len(nba_records)} saved to {NBA_JSON}")
print(f"Results saved to: {RESULT_CSV}")
