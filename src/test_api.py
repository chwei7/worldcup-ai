import os

import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("FOOTBALL_DATA_TOKEN")

if not token:
    print("找不到 API Token")
    raise SystemExit

token = token.strip()

print("Token 長度：", len(token))

url = "https://api.football-data.org/v4/competitions"

headers = {
    "X-Auth-Token": token
}

response = requests.get(
    url,
    headers=headers,
    timeout=30
)

print("狀態碼：", response.status_code)
print("伺服器訊息：", response.text)

if response.status_code == 200:
    data = response.json()
    competitions = data.get("competitions", [])

    print("\nAPI 連線成功")
    print("找到賽事數量：", len(competitions))

    for competition in competitions[:10]:
        print(
            competition.get("name"),
            "-",
            competition.get("code")