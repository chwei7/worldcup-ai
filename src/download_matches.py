import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("FOOTBALL_DATA_TOKEN")

if not token:
    print("找不到 API Token")
    raise SystemExit

url = "https://api.football-data.org/v4/competitions/WC/matches?season=2022"

headers = {
    "X-Auth-Token": token.strip()
}

response = requests.get(
    url,
    headers=headers,
    timeout=30
)

print("狀態碼：", response.status_code)

if response.status_code != 200:
    print("下載失敗：", response.text)
    raise SystemExit

data = response.json()
matches = data.get("matches", [])

rows = []

for match in matches:
    score = match.get("score", {})
    full_time = score.get("fullTime", {})

    rows.append(
        {
            "date": match.get("utcDate"),
            "status": match.get("status"),
            "stage": match.get("stage"),
            "home_team": match.get("homeTeam", {}).get("name"),
            "away_team": match.get("awayTeam", {}).get("name"),
            "home_score": full_time.get("home"),
            "away_score": full_time.get("away"),
        }
    )

df = pd.DataFrame(rows)

df.to_csv(
    "data/worldcup_matches.csv",
    index=False,
    encoding="utf-8-sig"
)

print("下載成功")
print("比賽數量：", len(df))
print("檔案位置：data/worldcup_matches.csv")