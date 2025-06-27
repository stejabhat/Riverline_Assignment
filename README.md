# Riverline_Assignment

Riverline ML Assignment – Next Best Action System
Overview
This repository contains a complete implementation of the Riverline Technical Assignment. The objective is to build a Next-Best-Action (NBA) system that intelligently suggests follow-up actions for unresolved customer support issues, based on behavior analysis and sentiment detection.

The solution is divided into four key stages:

Data Pipeline

User Behavior Observation

NBA Rule Engine

Evaluation & Output Generation

File Structure
graphql
Copy code
.
├── twcs.csv                     # Raw dataset (Customer Support on Twitter)
├── visuals/                     # Generated visualizations (if enabled)
├── interaction_table.csv        # Cleaned interaction table
├── tagged_interactions.csv      # Tagged data with sentiment and issue type
├── nba_outputs.json             # NBA response outputs (JSON format)
├── result.csv                   # Final result CSV (1000 customer records)
├── riverline_nba.py             # Single script to run the entire pipeline
├── requirements.txt             # Required Python packages
├── README.md                    # Project documentation
How to Run
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Ensure the following files are present in the project directory:

twcs.csv (Twitter Customer Support dataset)

Run the main script

bash
Copy code
python riverline_nba.py
This script will:

Load and clean the dataset

Generate customer-agent interaction pairs

Extract sentiment and classify support issues

Filter unresolved tickets

Generate NBA messages and reasoning

Export outputs to nba_outputs.json and result.csv

Assumptions & Design Decisions
Sentiment Analysis: Used TextBlob for quick sentiment scoring on customer messages.

Support Issue Classification: Simple keyword-based rule engine to detect request types like login issue, refund, delivery, etc.

NBA Rule Engine: Based on a ruleset that considers sentiment and issue type to choose the best communication channel:

Negative sentiment or critical issue → Phone Call

Positive sentiment → Email

Neutral or other → Twitter DM

Resolved Detection: Heuristic matching of phrases like "thank you", "resolved", "got it" in customer messages.

Idempotency: Intermediate CSVs are saved and reused unless source changes.

Output Files
nba_outputs.json: Contains NBA suggestions in the required JSON format.

result.csv: Contains a row-wise version of NBA output with added chat_log and issue_status fields.

Sample result.csv Columns:
customer_id	channel	send_time	message	reasoning	chat_log	issue_status
123456	email_reply	2025-07-01T10:15:00Z	...	...	...	resolved

Metrics & Summary
Resolved tickets detected (and excluded): Automatically counted based on sentiment keywords.

1000 unresolved customers: Filtered and used for NBA prediction.

Issue Status Predicted: resolved, pending customer reply, escalated based on sentiment and channel.

Optional Enhancements (Not Implemented)
Personality-aware messaging using Reddit MBTI corpus

ML-based classification of support requests

LLM-based message generation

Submission Checklist
 riverline_nba.py script (main pipeline)

 requirements.txt (minimal dependencies)

 nba_outputs.json and result.csv

 This README explaining design and usage

 (Optional) Loom video explanation

Credits & Resources
Customer Support on Twitter (Kaggle)

TextBlob for sentiment analysis
