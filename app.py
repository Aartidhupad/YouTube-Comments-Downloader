# app.py
from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
import io
import re
import requests
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# --- Setup ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Ensure VADER lexicon is available (download quietly if needed)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    logging.info("Downloading vader_lexicon...")
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# --- Helpers ---
def extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video id from many URL formats.
    Returns 11-character video id or None.
    """
    if not url:
        return None

    # common patterns
    patterns = [
        r"(?:v=|v\/|embed\/|youtu\.be\/|\/v\/)([A-Za-z0-9_-]{11})",
        r"([A-Za-z0-9_-]{11})$"  # fallback: last 11 chars
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

def fetch_all_comments(video_id: str, api_key: str, max_results_per_page: int = 100) -> list:
    """
    Use YouTube Data API v3 commentThreads endpoint (via requests).
    Returns list of comment strings. Raises exceptions on API errors.
    """
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "textFormat": "plainText",
        "key": api_key,
        "maxResults": max_results_per_page
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "fetchora/1.0"})

    while True:
        resp = session.get(base_url, params=params, timeout=15)
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError("Invalid JSON response from YouTube API")

        if resp.status_code != 200:
            # bubble up API error (message + reason if available)
            err = data.get("error", {})
            message = err.get("message", f"HTTP {resp.status_code}")
            raise RuntimeError(f"YouTube API error: {message}")

        items = data.get("items", [])
        for it in items:
            try:
                txt = it["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(txt)
            except KeyError:
                continue

        next_token = data.get("nextPageToken")
        if not next_token:
            break

        params["pageToken"] = next_token

    return comments

def analyze_comments(comments: list) -> pd.DataFrame:
    """
    Perform VADER sentiment analysis and return a DataFrame with columns:
    'comment' (string) and 'sentiment' (0 or 1).
    """
    rows = []
    for c in comments:
        score = sia.polarity_scores(c)["compound"]
        label = 1 if score >= 0 else 0
        rows.append({"comment": c, "sentiment": label})
    return pd.DataFrame(rows)

# --- Routes ---
@app.route("/")
def home():
    # make sure templates/index.html exists in ./templates
    return render_template("index.html")

@app.route("/fetch", methods=["POST"])
def fetch_route():
    """
    Expects JSON:
    {
      "api_key": "<you api key>",
      "video_url": "<youtube link or id>",
      "format": "csv" | "json" | "xlsx" | "html"
    }
    Returns a file download.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    api_key = data.get("api_key")
    video_url = data.get("video_url")
    file_format = (data.get("format") or "csv").lower()

    if not api_key:
        return jsonify({"error": "Missing api_key"}), 400
    if not video_url:
        return jsonify({"error": "Missing video_url"}), 400
    if file_format not in {"csv", "json", "xlsx", "html"}:
        return jsonify({"error": "Unsupported format"}), 400

    vid = extract_video_id(video_url)
    if not vid:
        return jsonify({"error": "Could not extract video id. Provide a full YouTube URL or video id."}), 400

    try:
        logging.info("Fetching comments for video id: %s", vid)
        comments = fetch_all_comments(vid, api_key)

        logging.info("Fetched %d comments", len(comments))
        df = analyze_comments(comments)

        # Prepare in-memory file for download
        if file_format == "xlsx":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name="yt_comments.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        else:
            # CSV / JSON / HTML -> pandas writes text. convert to bytes before sending.
            if file_format == "csv":
                text = df.to_csv(index=False)
                filename = "yt_comments.csv"
                mimetype = "text/csv"
            elif file_format == "json":
                text = df.to_json(orient="records", force_ascii=False)
                filename = "yt_comments.json"
                mimetype = "application/json"
            else:  # html
                text = df.to_html(index=False)
                filename = "yt_comments.html"
                mimetype = "text/html"

            buffer = io.BytesIO()
            buffer.write(text.encode("utf-8"))
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=filename, mimetype=mimetype)

    except Exception as exc:
        logging.exception("Error during /fetch")
        return jsonify({"error": str(exc)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
