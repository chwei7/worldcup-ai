
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# =========================================================
# 基本設定
# =========================================================

MATCH_FILE = "data/results.csv"
MODEL_FILE = "models/worldcup_model.joblib"

RECENT_MATCHES = 5
HEAD_TO_HEAD_LIMIT = 5


# =========================================================
# 球隊中文、英文名稱
# =========================================================

TEAM_NAME_ZH_TO_EN = {
    "阿根廷": "Argentina",
    "法國": "France",
    "巴西": "Brazil",
    "德國": "Germany",
    "日本": "Japan",
    "南韓": "South Korea",
    "西班牙": "Spain",
    "英格蘭": "England",
    "葡萄牙": "Portugal",
    "荷蘭": "Netherlands",
    "義大利": "Italy",
    "比利時": "Belgium",
    "烏拉圭": "Uruguay",
    "克羅埃西亞": "Croatia",
    "墨西哥": "Mexico",
    "美國": "United States",
    "加拿大": "Canada",
    "摩洛哥": "Morocco",
    "瑞士": "Switzerland",
    "丹麥": "Denmark",
    "塞內加爾": "Senegal",
    "澳洲": "Australia",
    "伊朗": "Iran",
    "沙烏地阿拉伯": "Saudi Arabia",
    "波蘭": "Poland",
    "塞爾維亞": "Serbia",
    "喀麥隆": "Cameroon",
    "迦納": "Ghana",
    "厄瓜多": "Ecuador",
    "哥斯大黎加": "Costa Rica",
    "威爾斯": "Wales",
    "突尼西亞": "Tunisia",
    "卡達": "Qatar",
    "智利": "Chile",
    "哥倫比亞": "Colombia",
    "巴拉圭": "Paraguay",
    "秘魯": "Peru",
    "委內瑞拉": "Venezuela",
    "瑞典": "Sweden",
    "挪威": "Norway",
    "芬蘭": "Finland",
    "冰島": "Iceland",
    "奧地利": "Austria",
    "捷克": "Czech Republic",
    "希臘": "Greece",
    "土耳其": "Turkey",
    "烏克蘭": "Ukraine",
    "俄羅斯": "Russia",
    "蘇格蘭": "Scotland",
    "愛爾蘭": "Republic of Ireland",
    "北愛爾蘭": "Northern Ireland",
    "羅馬尼亞": "Romania",
    "匈牙利": "Hungary",
    "斯洛伐克": "Slovakia",
    "斯洛維尼亞": "Slovenia",
    "阿爾及利亞": "Algeria",
    "埃及": "Egypt",
    "奈及利亞": "Nigeria",
    "象牙海岸": "Ivory Coast",
    "南非": "South Africa",
    "紐西蘭": "New Zealand",
    "中國": "China PR",
    "北韓": "North Korea",
    "牙買加": "Jamaica",
    "巴拿馬": "Panama",
    "宏都拉斯": "Honduras",
    "以色列": "Israel",
    "阿聯酋": "United Arab Emirates",
    "伊拉克": "Iraq",
    "約旦": "Jordan",
}


TEAM_NAME_EN_TO_ZH = {
    english: chinese
    for chinese, english in TEAM_NAME_ZH_TO_EN.items()
}


# =========================================================
# 賽事中文名稱
# =========================================================

TOURNAMENT_ZH = {
    "Friendly": "國際友誼賽",
    "FIFA World Cup": "世界盃",
    "FIFA World Cup qualification": "世界盃資格賽",
    "UEFA Euro": "歐洲國家盃",
    "UEFA Euro qualification": "歐洲國家盃資格賽",
    "Copa América": "美洲盃",
    "Copa América qualification": "美洲盃資格賽",
    "AFC Asian Cup": "亞洲盃",
    "AFC Asian Cup qualification": "亞洲盃資格賽",
    "African Cup of Nations": "非洲國家盃",
    "African Cup of Nations qualification": "非洲國家盃資格賽",
    "CONCACAF Gold Cup": "中北美洲及加勒比海金盃",
    "CONCACAF Nations League": "中北美洲及加勒比海國家聯賽",
    "UEFA Nations League": "歐洲國家聯賽",
    "Confederations Cup": "洲際國家盃",
    "Olympic Games": "奧運足球賽",
    "Asian Games": "亞洲運動會足球賽",
}


# =========================================================
# AI 特徵中文名稱
# =========================================================

FEATURE_NAME_ZH = {
    "home_win_rate": "第一隊最近勝率",
    "away_win_rate": "第二隊最近勝率",
    "home_draw_rate": "第一隊最近平手率",
    "away_draw_rate": "第二隊最近平手率",
    "home_loss_rate": "第一隊最近敗率",
    "away_loss_rate": "第二隊最近敗率",
    "home_avg_goals_for": "第一隊最近場均進球",
    "away_avg_goals_for": "第二隊最近場均進球",
    "home_avg_goals_against": "第一隊最近場均失球",
    "away_avg_goals_against": "第二隊最近場均失球",
    "home_avg_goal_difference": "第一隊最近平均淨勝球",
    "away_avg_goal_difference": "第二隊最近平均淨勝球",
    "diff_win_rate": "兩隊最近勝率差距",
    "diff_avg_goals_for": "兩隊最近場均進球差距",
    "diff_avg_goals_against": "兩隊最近場均失球差距",
    "diff_goal_difference": "兩隊最近淨勝球差距",
    "h2h_home_win_rate": "第一隊近期交手勝率",
    "h2h_draw_rate": "近期交手平手率",
    "h2h_away_win_rate": "第二隊近期交手勝率",
    "h2h_matches": "近期交手場數",
    "h2h_win_rate_difference": "兩隊近期交手勝率差距",
}


# =========================================================
# 輔助函式
# =========================================================

def display_team_name(english_name):
    return TEAM_NAME_EN_TO_ZH.get(
        english_name,
        english_name,
    )


def translate_tournament(name):
    if pd.isna(name):
        return ""

    name = str(name).strip()

    return TOURNAMENT_ZH.get(
        name,
        name,
    )


def team_result(goals_for, goals_against):
    if goals_for > goals_against:
        return 1

    if goals_for < goals_against:
        return -1

    return 0


def result_text(result):
    if result == 1:
        return "勝"

    if result == 0:
        return "和"

    return "敗"


# =========================================================
# 載入資料與模型
# =========================================================

@st.cache_data
def load_matches():
    matches = pd.read_csv(
        MATCH_FILE,
        parse_dates=["date"],
    )

    matches = matches.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    )

    matches["home_team"] = (
        matches["home_team"]
        .astype(str)
        .str.strip()
    )

    matches["away_team"] = (
        matches["away_team"]
        .astype(str)
        .str.strip()
    )

    matches["home_score"] = pd.to_numeric(
        matches["home_score"],
        errors="coerce",
    )

    matches["away_score"] = pd.to_numeric(
        matches["away_score"],
        errors="coerce",
    )

    matches = matches.dropna(
        subset=[
            "home_score",
            "away_score",
        ]
    )

    return matches.sort_values(
        "date"
    ).reset_index(drop=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# =========================================================
# 建立球隊歷史資料
# =========================================================

@st.cache_data
def build_histories(matches):
    histories = {}

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        histories.setdefault(home_team, [])
        histories.setdefault(away_team, [])

        histories[home_team].append(
            {
                "date": match["date"],
                "opponent": away_team,
                "home_away": "主場",
                "goals_for": home_score,
                "goals_against": away_score,
                "result": team_result(
                    home_score,
                    away_score,
                ),
                "tournament": match.get(
                    "tournament",
                    "",
                ),
            }
        )

        histories[away_team].append(
            {
                "date": match["date"],
                "opponent": home_team,
                "home_away": "客場",
                "goals_for": away_score,
                "goals_against": home_score,
                "result": team_result(
                    away_score,
                    home_score,
                ),
                "tournament": match.get(
                    "tournament",
                    "",
                ),
            }
        )

    return histories


# =========================================================
# 計算最近比賽統計
# =========================================================

def calculate_stats(history):
    if not history:
        return {
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "avg_goal_difference": 0.0,
        }

    count = len(history)

    wins = sum(
        game["result"] == 1
        for game in history
    )

    draws = sum(
        game["result"] == 0
        for game in history
    )

    losses = sum(
        game["result"] == -1
        for game in history
    )

    goals_for = sum(
        game["goals_for"]
        for game in history
    )

    goals_against = sum(
        game["goals_against"]
        for game in history
    )

    return {
        "win_rate": wins / count,
        "draw_rate": draws / count,
        "loss_rate": losses / count,
        "avg_goals_for": goals_for / count,
        "avg_goals_against": goals_against / count,
        "avg_goal_difference": (
            goals_for - goals_against
        ) / count,
    }


# =========================================================
# 建立最近五場比賽表格
# =========================================================

def create_recent_matches_table(history):
    rows = []

    for game in reversed(history):
        game_date = game.get(
            "date",
            "",
        )

        if pd.notna(game_date):
            game_date = pd.to_datetime(
                game_date
            ).strftime("%Y-%m-%d")
        else:
            game_date = ""

        opponent = display_team_name(
            game.get(
                "opponent",
                "",
            )
        )

        tournament = translate_tournament(
            game.get(
                "tournament",
                "",
            )
        )

        score = (
            f"{int(game['goals_for'])}"
            f"："
            f"{int(game['goals_against'])}"
        )

        rows.append(
            {
                "日期": game_date,
                "對手": opponent,
                "主客場": game.get(
                    "home_away",
                    "",
                ),
                "比數": score,
                "結果": result_text(
                    game["result"]
                ),
                "賽事": tournament,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 取得兩隊交手紀錄
# =========================================================

def get_head_to_head(
    matches,
    home_team,
    away_team,
):
    condition1 = (
        (matches["home_team"] == home_team)
        & (matches["away_team"] == away_team)
    )

    condition2 = (
        (matches["home_team"] == away_team)
        & (matches["away_team"] == home_team)
    )

    head_to_head = matches[
        condition1 | condition2
    ].copy()

    return (
        head_to_head.sort_values(
            "date",
            ascending=False,
        )
        .head(HEAD_TO_HEAD_LIMIT)
    )


def calculate_h2h_stats(
    h2h_matches,
    home_team,
    away_team,
):
    home_wins = 0
    away_wins = 0
    draws = 0

    for _, match in h2h_matches.iterrows():
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        if home_score == away_score:
            draws += 1
            continue

        if home_score > away_score:
            winner = match["home_team"]
        else:
            winner = match["away_team"]

        if winner == home_team:
            home_wins += 1

        elif winner == away_team:
            away_wins += 1

    count = len(h2h_matches)

    if count == 0:
        return {
            "h2h_home_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_matches": 0,
            "h2h_win_rate_difference": 0.0,
        }

    home_win_rate = home_wins / count
    away_win_rate = away_wins / count

    return {
        "h2h_home_win_rate": home_win_rate,
        "h2h_draw_rate": draws / count,
        "h2h_away_win_rate": away_win_rate,
        "h2h_matches": count,
        "h2h_win_rate_difference": (
            home_win_rate - away_win_rate
        ),
    }


# =========================================================
# 建立模型需要的特徵
# =========================================================

def create_features(
    home_stats,
    away_stats,
    h2h_stats,
    feature_columns,
):
    values = {
        "home_win_rate":
            home_stats["win_rate"],

        "away_win_rate":
            away_stats["win_rate"],

        "home_draw_rate":
            home_stats["draw_rate"],

        "away_draw_rate":
            away_stats["draw_rate"],

        "home_loss_rate":
            home_stats["loss_rate"],

        "away_loss_rate":
            away_stats["loss_rate"],

        "home_avg_goals_for":
            home_stats["avg_goals_for"],

        "away_avg_goals_for":
            away_stats["avg_goals_for"],

        "home_avg_goals_against":
            home_stats["avg_goals_against"],

        "away_avg_goals_against":
            away_stats["avg_goals_against"],

        "home_avg_goal_difference":
            home_stats["avg_goal_difference"],

        "away_avg_goal_difference":
            away_stats["avg_goal_difference"],

        "diff_win_rate": (
            home_stats["win_rate"]
            - away_stats["win_rate"]
        ),

        "diff_avg_goals_for": (
            home_stats["avg_goals_for"]
            - away_stats["avg_goals_for"]
        ),

        "diff_avg_goals_against": (
            home_stats["avg_goals_against"]
            - away_stats["avg_goals_against"]
        ),

        "diff_goal_difference": (
            home_stats["avg_goal_difference"]
            - away_stats["avg_goal_difference"]
        ),

        "h2h_home_win_rate":
            h2h_stats["h2h_home_win_rate"],

        "h2h_draw_rate":
            h2h_stats["h2h_draw_rate"],

        "h2h_away_win_rate":
            h2h_stats["h2h_away_win_rate"],

        "h2h_matches":
            h2h_stats["h2h_matches"],

        "h2h_win_rate_difference":
            h2h_stats["h2h_win_rate_difference"],
    }

    features = pd.DataFrame([values])

    return features.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )


# =========================================================
# SHAP AI 解釋
# =========================================================

def format_feature_value(
    feature_name,
    value,
):
    if feature_name == "h2h_matches":
        return f"{int(round(value))} 場"

    if (
        "rate" in feature_name
        or feature_name
        in {
            "diff_win_rate",
            "h2h_win_rate_difference",
        }
    ):
        return f"{value * 100:.1f}%"

    return f"{value:.2f}"


def get_shap_class_values(
    model,
    features,
    predicted_class,
):
    explainer = shap.TreeExplainer(model)

    try:
        shap_output = explainer(features)
        shap_values = shap_output.values

    except Exception:
        shap_values = explainer.shap_values(
            features
        )

    class_list = list(model.classes_)

    if predicted_class not in class_list:
        raise ValueError(
            "模型找不到預測類別"
        )

    class_index = class_list.index(
        predicted_class
    )

    if isinstance(shap_values, list):
        return np.asarray(
            shap_values[class_index]
        )[0]

    shap_array = np.asarray(
        shap_values
    )

    if shap_array.ndim == 2:
        return shap_array[0]

    if shap_array.ndim == 3:
        # 格式：
        # 資料筆數、特徵數、類別數
        if (
            shap_array.shape[0] == len(features)
            and shap_array.shape[1]
            == len(features.columns)
        ):
            return shap_array[
                0,
                :,
                class_index,
            ]

        # 格式：
        # 類別數、資料筆數、特徵數
        if shap_array.shape[0] == len(
            class_list
        ):
            return shap_array[
                class_index,
                0,
                :,
            ]

        # 格式：
        # 資料筆數、類別數、特徵數
        if shap_array.shape[1] == len(
            class_list
        ):
            return shap_array[
                0,
                class_index,
                :,
            ]

    raise ValueError(
        f"無法辨識 SHAP 格式："
        f"{shap_array.shape}"
    )


def get_model_explanations(
    model,
    features,
    predicted_class,
    team1_zh,
    team2_zh,
    limit=5,
):
    shap_values = get_shap_class_values(
        model,
        features,
        predicted_class,
    )

    if predicted_class == 2:
        target_text = f"{team1_zh}獲勝"

    elif predicted_class == 0:
        target_text = f"{team2_zh}獲勝"

    else:
        target_text = "平手"

    rows = []

    for index, feature_name in enumerate(
        features.columns
    ):
        impact = float(
            shap_values[index]
        )

        value = float(
            features.iloc[0][feature_name]
        )

        if impact >= 0:
            influence = (
                f"提高「{target_text}」的預測"
            )

        else:
            influence = (
                f"降低「{target_text}」的預測"
            )

        rows.append(
            {
                "feature": feature_name,
                "原因": FEATURE_NAME_ZH.get(
                    feature_name,
                    feature_name,
                ),
                "目前數值": format_feature_value(
                    feature_name,
                    value,
                ),
                "影響": influence,
                "impact": impact,
                "absolute_impact": abs(
                    impact
                ),
            }
        )

    rows.sort(
        key=lambda item: item[
            "absolute_impact"
        ],
        reverse=True,
    )

    return rows[:limit]


def build_plain_language_reason(
    item,
    team1_zh,
    team2_zh,
):
    feature = item["feature"]
    value = item["目前數值"]
    influence = item["影響"]

    descriptions = {
        "home_win_rate":
            f"{team1_zh}最近勝率為{value}",

        "away_win_rate":
            f"{team2_zh}最近勝率為{value}",

        "home_draw_rate":
            f"{team1_zh}最近平手率為{value}",

        "away_draw_rate":
            f"{team2_zh}最近平手率為{value}",

        "home_loss_rate":
            f"{team1_zh}最近敗率為{value}",

        "away_loss_rate":
            f"{team2_zh}最近敗率為{value}",

        "home_avg_goals_for":
            f"{team1_zh}最近場均進球為{value}",

        "away_avg_goals_for":
            f"{team2_zh}最近場均進球為{value}",

        "home_avg_goals_against":
            f"{team1_zh}最近場均失球為{value}",

        "away_avg_goals_against":
            f"{team2_zh}最近場均失球為{value}",

        "home_avg_goal_difference":
            f"{team1_zh}最近平均淨勝球為{value}",

        "away_avg_goal_difference":
            f"{team2_zh}最近平均淨勝球為{value}",

        "diff_win_rate":
            f"兩隊最近勝率差距為{value}",

        "diff_avg_goals_for":
            f"兩隊最近場均進球差距為{value}",

        "diff_avg_goals_against":
            f"兩隊最近場均失球差距為{value}",

        "diff_goal_difference":
            f"兩隊最近淨勝球差距為{value}",

        "h2h_home_win_rate":
            f"{team1_zh}近期交手勝率為{value}",

        "h2h_draw_rate":
            f"兩隊近期交手平手率為{value}",

        "h2h_away_win_rate":
            f"{team2_zh}近期交手勝率為{value}",

        "h2h_matches":
            f"模型參考了兩隊最近{value}交手",

        "h2h_win_rate_difference":
            f"兩隊近期交手勝率差距為{value}",
    }

    beginning = descriptions.get(
        feature,
        f"{item['原因']}為{value}",
    )

    return (
        f"{beginning}，"
        f"因此會{influence}。"
    )


# =========================================================
# Streamlit 網頁設定
# =========================================================

st.set_page_config(
    page_title="AI 足球勝率預測",
    page_icon="⚽",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f3f6f9;
    }

    h1 {
        text-align: center;
        color: #1f4e79;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555555;
        margin-bottom: 30px;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
    }

    .reason-card {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("⚽ AI 足球勝率預測")

st.markdown(
    """
    <div class="subtitle">
        選擇兩支球隊，查看 AI 預測、近期狀態與判斷原因
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 主程式
# =========================================================

try:
    matches = load_matches()
    histories = build_histories(matches)

    model_bundle = load_model()

    model = model_bundle["model"]
    feature_columns = model_bundle[
        "feature_columns"
    ]

    team_options = sorted(
        [
            chinese_name
            for chinese_name, english_name
            in TEAM_NAME_ZH_TO_EN.items()
            if english_name in histories
        ]
    )

    if len(team_options) < 2:
        st.error(
            "資料庫中可使用的球隊不足。"
        )
        st.stop()

    team_column1, team_column2 = st.columns(2)

    with team_column1:
        team1_zh = st.selectbox(
            "選擇第一支球隊",
            team_options,
            index=0,
        )

    with team_column2:
        team2_zh = st.selectbox(
            "選擇第二支球隊",
            team_options,
            index=1,
        )

    if st.button("開始預測"):
        if team1_zh == team2_zh:
            st.error(
                "請選擇兩支不同的球隊。"
            )

        else:
            home_team = TEAM_NAME_ZH_TO_EN[
                team1_zh
            ]

            away_team = TEAM_NAME_ZH_TO_EN[
                team2_zh
            ]

            home_history = histories.get(
                home_team,
                [],
            )[-RECENT_MATCHES:]

            away_history = histories.get(
                away_team,
                [],
            )[-RECENT_MATCHES:]

            home_stats = calculate_stats(
                home_history
            )

            away_stats = calculate_stats(
                away_history
            )

            h2h_matches = get_head_to_head(
                matches,
                home_team,
                away_team,
            )

            h2h_stats = calculate_h2h_stats(
                h2h_matches,
                home_team,
                away_team,
            )

            features = create_features(
                home_stats,
                away_stats,
                h2h_stats,
                feature_columns,
            )

            probabilities = model.predict_proba(
                features
            )[0]

            class_probabilities = dict(
                zip(
                    model.classes_,
                    probabilities,
                )
            )

            away_win = (
                class_probabilities.get(
                    0,
                    0.0,
                )
                * 100
            )

            draw = (
                class_probabilities.get(
                    1,
                    0.0,
                )
                * 100
            )

            home_win = (
                class_probabilities.get(
                    2,
                    0.0,
                )
                * 100
            )

            result_probabilities = {
                f"{team1_zh}勝": home_win,
                "平手": draw,
                f"{team2_zh}勝": away_win,
            }

            most_likely = max(
                result_probabilities,
                key=result_probabilities.get,
            )

            if most_likely == f"{team1_zh}勝":
                predicted_class = 2

            elif most_likely == f"{team2_zh}勝":
                predicted_class = 0

            else:
                predicted_class = 1

            # 預測結果
            st.subheader("AI 預測結果")

            result_column1, result_column2, result_column3 = (
                st.columns(3)
            )

            result_column1.metric(
                f"{team1_zh} 勝",
                f"{home_win:.1f}%",
            )

            result_column2.metric(
                "平手",
                f"{draw:.1f}%",
            )

            result_column3.metric(
                f"{team2_zh} 勝",
                f"{away_win:.1f}%",
            )

            st.success(
                f"最可能結果：{most_likely}"
            )

            # AI 判斷原因
            st.subheader("AI 判斷原因")

            try:
                explanation_rows = (
                    get_model_explanations(
                        model=model,
                        features=features,
                        predicted_class=predicted_class,
                        team1_zh=team1_zh,
                        team2_zh=team2_zh,
                        limit=5,
                    )
                )

                for index, item in enumerate(
                    explanation_rows[:3],
                    start=1,
                ):
                    reason = (
                        build_plain_language_reason(
                            item,
                            team1_zh,
                            team2_zh,
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="reason-card">
                            <strong>原因 {index}</strong>
                            <br>
                            {reason}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander(
                    "查看完整模型影響表"
                ):
                    explanation_table = pd.DataFrame(
                        explanation_rows
                    )

                    st.dataframe(
                        explanation_table[
                            [
                                "原因",
                                "目前數值",
                                "影響",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "AI 判斷原因代表模型如何解讀歷史數據，"
                    "不代表真正的因果關係。"
                )

            except Exception as explanation_error:
                st.warning(
                    "目前無法產生 AI 判斷原因："
                    f"{explanation_error}"
                )

            # 最近五場統計與對手
            st.subheader("最近 5 場數據")

            recent_column1, recent_column2 = (
                st.columns(2)
            )

            with recent_column1:
                st.markdown(
                    f"### {team1_zh}"
                )

                stat1, stat2, stat3 = st.columns(3)

                stat1.metric(
                    "勝率",
                    (
                        f"{home_stats['win_rate'] * 100:.1f}%"
                    ),
                )

                stat2.metric(
                    "平手率",
                    (
                        f"{home_stats['draw_rate'] * 100:.1f}%"
                    ),
                )

                stat3.metric(
                    "敗率",
                    (
                        f"{home_stats['loss_rate'] * 100:.1f}%"
                    ),
                )

                st.write(
                    "場均進球："
                    f"{home_stats['avg_goals_for']:.2f}"
                )

                st.write(
                    "場均失球："
                    f"{home_stats['avg_goals_against']:.2f}"
                )

                st.write(
                    "平均淨勝球："
                    f"{home_stats['avg_goal_difference']:.2f}"
                )

                st.markdown(
                    "#### 最近比賽對手"
                )

                home_recent_table = (
                    create_recent_matches_table(
                        home_history
                    )
                )

                st.dataframe(
                    home_recent_table,
                    use_container_width=True,
                    hide_index=True,
                )

            with recent_column2:
                st.markdown(
                    f"### {team2_zh}"
                )

                stat1, stat2, stat3 = st.columns(3)

                stat1.metric(
                    "勝率",
                    (
                        f"{away_stats['win_rate'] * 100:.1f}%"
                    ),
                )

                stat2.metric(
                    "平手率",
                    (
                        f"{away_stats['draw_rate'] * 100:.1f}%"
                    ),
                )

                stat3.metric(
                    "敗率",
                    (
                        f"{away_stats['loss_rate'] * 100:.1f}%"
                    ),
                )

                st.write(
                    "場均進球："
                    f"{away_stats['avg_goals_for']:.2f}"
                )

                st.write(
                    "場均失球："
                    f"{away_stats['avg_goals_against']:.2f}"
                )

                st.write(
                    "平均淨勝球："
                    f"{away_stats['avg_goal_difference']:.2f}"
                )

                st.markdown(
                    "#### 最近比賽對手"
                )

                away_recent_table = (
                    create_recent_matches_table(
                        away_history
                    )
                )

                st.dataframe(
                    away_recent_table,
                    use_container_width=True,
                    hide_index=True,
                )

            # 交手統計
            st.subheader("近期交手統計")

            h2h_column1, h2h_column2, h2h_column3 = (
                st.columns(3)
            )

            h2h_column1.metric(
                f"{team1_zh} 勝率",
                (
                    f"{h2h_stats['h2h_home_win_rate'] * 100:.1f}%"
                ),
            )

            h2h_column2.metric(
                "平手率",
                (
                    f"{h2h_stats['h2h_draw_rate'] * 100:.1f}%"
                ),
            )

            h2h_column3.metric(
                f"{team2_zh} 勝率",
                (
                    f"{h2h_stats['h2h_away_win_rate'] * 100:.1f}%"
                ),
            )

            st.write(
                "模型參考的直接交手場數："
                f"{h2h_stats['h2h_matches']} 場"
            )

            # 交手紀錄表
            st.subheader("近期交手紀錄")

            if h2h_matches.empty:
                st.info(
                    "找不到兩隊近期交手紀錄。"
                )

            else:
                h2h_table = h2h_matches[
                    [
                        "date",
                        "home_team",
                        "away_team",
                        "home_score",
                        "away_score",
                        "tournament",
                    ]
                ].copy()

                h2h_table["home_team"] = (
                    h2h_table["home_team"].map(
                        display_team_name
                    )
                )

                h2h_table["away_team"] = (
                    h2h_table["away_team"].map(
                        display_team_name
                    )
                )

                h2h_table["tournament"] = (
                    h2h_table["tournament"].map(
                        translate_tournament
                    )
                )

                h2h_table["date"] = (
                    h2h_table["date"].dt.strftime(
                        "%Y-%m-%d"
                    )
                )

                h2h_table["比分"] = (
                    h2h_table[
                        "home_score"
                    ].astype(int).astype(str)
                    + "："
                    + h2h_table[
                        "away_score"
                    ].astype(int).astype(str)
                )

                h2h_table = h2h_table.rename(
                    columns={
                        "date": "日期",
                        "home_team": "主隊",
                        "away_team": "客隊",
                        "tournament": "賽事",
                    }
                )

                h2h_table = h2h_table[
                    [
                        "日期",
                        "主隊",
                        "比分",
                        "客隊",
                        "賽事",
                    ]
                ]

                st.dataframe(
                    h2h_table,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "提醒：預測是依據歷史比賽資料與模型計算，"
                "不能保證實際比賽結果。"
            )


except FileNotFoundError as error:
    st.error(
        f"找不到必要檔案：{error}"
    )


except KeyError as error:
    st.error(
        "模型檔案或資料欄位格式不正確："
        f"{error}"
    )


except Exception as error:
    st.error(
        f"程式發生錯誤：{error}"
    )
```
