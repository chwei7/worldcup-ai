import os

from dotenv import load_dotenv

load_dotenv()

token = os.getenv("FOOTBALL_DATA_TOKEN")

if token:
    print("API Token 讀取成功")
    print("Token 長度：", len(token))
else:
    print("找不到 API Token")