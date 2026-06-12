import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


MATCH_FILE = "data/results.csv"
MODEL_FILE = "models/worldcup_model.joblib"

RECENT_MATCHES = 5
HEAD_TO_HEAD_LIMIT = 5


TEAM_NAME_ZH_TO_EN = {
    "阿根廷": "Argentina",
    "法國": "France",
    "巴西": "Brazil",
    "德國": "Germany",
    "日本": "Japan",
    "南韓": "South Korea",
    "韓國": "South Korea",
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
    "澳大利亞": "Australia",
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
    "捷克共和國": "Czech Republic",
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
    "科特迪瓦": "Ivory Coast",
    "南非": "South Africa",
    "紐西蘭": "New Zealand",
    "中國": "China PR",
    "中國隊": "China PR",
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
    "Argentina": "阿根廷",
    "France": "法國",
    "Brazil": "巴西",
    "Germany": "德國",
    "Japan": "日本",
    "South Korea": "南韓",
    "Spain": "西班牙",
    "England": "英格蘭",
    "Portugal": "葡萄牙",
    "Netherlands": "荷蘭",
    "Italy": "義大利",
    "Belgium": "比利時",
    "Uruguay": "烏拉圭",
    "Croatia": "克羅埃西亞",
    "Mexico": "墨西哥",
    "United States": "美國",
    "Canada": "加拿大",
    "Morocco": "摩洛哥",
    "Switzerland": "瑞士",
    "Denmark": "丹麥",
    "Senegal": "塞內加爾",
    "Australia": "澳洲",
    "Iran": "伊朗",
    "Saudi Arabia": "沙烏地阿拉伯",
    "Poland": "波蘭",
    "Serbia": "塞爾維亞",
    "Cameroon": "喀麥隆",
    "Ghana": "迦納",
    "Ecuador": "厄瓜多",
    "Costa Rica": "哥斯大黎加",
    "Wales": "威爾斯",
    "Tunisia": "突尼西亞",
    "Qatar": "卡達",
    "Chile": "智利",
    "Colombia": "哥倫比亞",
    "Paraguay": "巴拉圭",
    "Peru": "秘魯",
    "Venezuela": "委內瑞拉",
    "Sweden": "瑞典",
    "Norway": "挪威",
    "Finland": "芬蘭",
    "Iceland": "冰島",
    "Austria": "奧地利",
    "Czech Republic": "捷克",
    "Greece": "希臘",
    "Turkey": "土耳其",
    "Ukraine": "烏克蘭",
    "Russia": "俄羅斯",
    "Scotland": "蘇格蘭",
    "Republic of Ireland": "愛爾蘭",
    "Northern Ireland": "北愛爾蘭",
    "Romania": "羅馬尼亞",
    "Hungary": "匈牙利",
    "Slovakia": "斯洛伐克",
    "Slovenia": "斯洛維尼亞",
    "Algeria": "阿爾及利亞",
    "Egypt": "埃及",
    "Nigeria": "奈及利亞",
    "Ivory Coast": "象牙海岸",
    "South Africa": "南非",
    "New Zealand": "紐西蘭",
    "China PR": "中國",
    "North Korea": "北韓",
    "Jamaica": "牙買加",
    "Panama": "巴拿馬",
    "Honduras": "宏都拉斯",
    "Israel": "以色列",
    "United Arab Emirates": "阿聯酋",
    "Iraq": "伊拉克",
    "Jordan": "約旦",
}


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
    "h2h_draw_rate": "兩隊近期交手平手率",
    "h2h_away_win_rate": "第二隊近期交手勝率",
    "h2h_matches": "兩隊近期交手場數",
    "h2h_win_rate_difference": "兩隊近期交手勝率差距",
}


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

    return matches.sort_values(
        "date"
    ).reset_index(drop=True)


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
                "goals_for": home_score,
                "goals_against": away_score,
                "result": team_result(
                    home_score,
                    away_score,
                ),
            }
        )

        histories[away_team].append(
            {
                "goals_for": away_score,
                "goals_against": home_score,
                "result": team_result(
                    away_score,
                    home_score,
                ),
            }
        )

    return histories


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


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

        winner = (
            match["home_team"]
            if home_score > away_score
            else match["away_team"]
        )

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

    row = pd.DataFrame([values])

    return row.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )


def format_feature_value(
    feature_name,
    value,
):
    if feature_name == "h2h_matches":
        return f"{int(round(value))} 場"

    if (
        "rate" in feature_name
        or feature_name == "diff_win_rate"
        or feature_name == "h2h_win_rate_difference"
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
            "模型中找不到預測類別"
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
        # 常見新版格式：
        # (資料筆數, 特徵數, 類別數)
        if (
            shap_array.shape[0] == len(features)
            and
            shap_array.shape[1] == len(
                features.columns
            )
        ):
            return shap_array[
                0,
                :,
                class_index,
            ]

        # 另一種可能格式：
        # (類別數, 資料筆數, 特徵數)
        if shap_array.shape[0] == len(
            class_list
        ):
            return shap_array[
                class_index,
                0,
                :,
            ]

        # 另一種可能格式：
        # (資料筆數, 類別數, 特徵數)
        if shap_array.shape[1] == len(
            class_list
        ):
            return shap_array[
                0,
                class_index,
                :,
            ]

    raise ValueError(
        f"無法辨識 SHAP 輸出格式："
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
    class_values = get_shap_class_values(
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

    explanation_rows = []

    for index, feature_name in enumerate(
        features.columns
    ):
        impact = float(
            class_values[index]
        )

        feature_value = float(
            features.iloc[0][feature_name]
        )

        if impact > 0:
            influence_text = (
                f"提高「{target_text}」的預測"
            )
        else:
            influence_text = (
                f"降低「{target_text}」的預測"
            )

        explanation_rows.append(
            {
                "feature": feature_name,
                "原因": FEATURE_NAME_ZH.get(
                    feature_name,
                    feature_name,
                ),
                "目前數值": format_feature_value(
                    feature_name,
                    feature_value,
                ),
                "影響": influence_text,
                "impact": impact,
                "absolute_impact": abs(impact),
            }
        )

    explanation_rows.sort(
        key=lambda item: item[
            "absolute_impact"
        ],
        reverse=True,
    )

    return explanation_rows[:limit]


def build_plain_language_reason(
    item,
    team1_zh,
    team2_zh,
):
    feature = item["feature"]
    value_text = item["目前數值"]
    influence = item["影響"]

    feature_sentences = {
        "home_win_rate":
            f"{team1_zh}最近勝率為{value_text}",

        "away_win_rate":
            f"{team2_zh}最近勝率為{value_text}",

        "home_draw_rate":
            f"{team1_zh}最近平手率為{value_text}",

        "away_draw_rate":
            f"{team2_zh}最近平手率為{value_text}",

        "home_loss_rate":
            f"{team1_zh}最近敗率為{value_text}",

        "away_loss_rate":
            f"{team2_zh}最近敗率為{value_text}",

        "home_avg_goals_for":
            f"{team1_zh}最近場均進球為{value_text}",

        "away_avg_goals_for":
            f"{team2_zh}最近場均進球為{value_text}",

        "home_avg_goals_against":
            f"{team1_zh}最近場均失球為{value_text}",

        "away_avg_goals_against":
            f"{team2_zh}最近場均失球為{value_text}",

        "home_avg_goal_difference":
            f"{team1_zh}最近平均淨勝球為{value_text}",

        "away_avg_goal_difference":
            f"{team2_zh}最近平均淨勝球為{value_text}",

        "diff_win_rate":
            f"兩隊最近勝率差距為{value_text}",

        "diff_avg_goals_for":
            f"兩隊最近場均進球差距為{value_text}",

        "diff_avg_goals_against":
            f"兩隊最近場均失球差距為{value_text}",

        "diff_goal_difference":
            f"兩隊最近淨勝球差距為{value_text}",

        "h2h_home_win_rate":
            f"{team1_zh}近期交手勝率為{value_text}",

        "h2h_draw_rate":
            f"兩隊近期交手平手率為{value_text}",

        "h2h_away_win_rate":
            f"{team2_zh}近期交手勝率為{value_text}",

        "h2h_matches":
            f"模型參考了兩隊最近{value_text}交手",

        "h2h_win_rate_difference":
            f"兩隊近期交手勝率差距為{value_text}",
    }

    beginning = feature_sentences.get(
        feature,
        f"{item['原因']}為{value_text}",
    )

    return f"{beginning}，因此會{influence}。"


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
    '<div class="subtitle">'
    '選擇兩支球隊，查看 AI 預測、近期狀態與判斷原因'
    '</div>',
    unsafe_allow_html=True,
)


try:
    matches = load_matches()
    histories = build_histories(matches)

    bundle = load_model()

    model = bundle["model"]
    feature_columns = bundle[
        "feature_columns"
    ]

    team_options = sorted(
        {
            chinese
            for chinese, english
            in TEAM_NAME_ZH_TO_EN.items()
            if english in histories
        }
    )

    if len(team_options) < 2:
        st.error("可使用的球隊數量不足")
        st.stop()

    left_column, right_column = st.columns(2)

    with left_column:
        team1_zh = st.selectbox(
            "選擇第一支球隊",
            team_options,
            key="team1",
        )

    with right_column:
        team2_zh = st.selectbox(
            "選擇第二支球隊",
            team_options,
            index=1,
            key="team2",
        )

    if st.button("開始預測"):
        if team1_zh == team2_zh:
            st.error(
                "請選擇不同的兩支球隊"
            )

        else:
            home_team = TEAM_NAME_ZH_TO_EN[
                team1_zh
            ]

            away_team = TEAM_NAME_ZH_TO_EN[
                team2_zh
            ]

            home_history = histories[
                home_team
            ][-RECENT_MATCHES:]

            away_history = histories[
                away_team
            ][-RECENT_MATCHES:]

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

            probabilities = (
                model.predict_proba(
                    features
                )[0]
            )

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

            results = {
                f"{team1_zh}勝": home_win,
                "平手": draw,
                f"{team2_zh}勝": away_win,
            }

            most_likely = max(
                results,
                key=results.get,
            )

            if most_likely == f"{team1_zh}勝":
                predicted_class = 2

            elif most_likely == f"{team2_zh}勝":
                predicted_class = 0

            else:
                predicted_class = 1

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
                    reason_text = (
                        build_plain_language_reason(
                            item,
                            team1_zh,
                            team2_zh,
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="reason-card">
                            <strong>原因 {index}</strong><br>
                            {reason_text}
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
                    "AI 原因是根據模型特徵影響程度產生，"
                    "代表模型如何判斷，不代表真正的因果關係。"
                )

            except Exception as explanation_error:
                st.warning(
                    "目前無法產生 AI 判斷原因："
                    f"{explanation_error}"
                )

            st.subheader("最近 5 場數據")

            stats_column1, stats_column2 = (
                st.columns(2)
            )

            with stats_column1:
                st.markdown(
                    f"### {team1_zh}"
                )

                st.write(
                    f"勝率："
                    f"{home_stats['win_rate'] * 100:.1f}%"
                )

                st.write(
                    f"平手率："
                    f"{home_stats['draw_rate'] * 100:.1f}%"
                )

                st.write(
                    f"敗率："
                    f"{home_stats['loss_rate'] * 100:.1f}%"
                )

                st.write(
                    f"場均進球："
                    f"{home_stats['avg_goals_for']:.2f}"
                )

                st.write(
                    f"場均失球："
                    f"{home_stats['avg_goals_against']:.2f}"
                )

                st.write(
                    f"平均淨勝球："
                    f"{home_stats['avg_goal_difference']:.2f}"
                )

            with stats_column2:
                st.markdown(
                    f"### {team2_zh}"
                )

                st.write(
                    f"勝率："
                    f"{away_stats['win_rate'] * 100:.1f}%"
                )

                st.write(
                    f"平手率："
                    f"{away_stats['draw_rate'] * 100:.1f}%"
                )

                st.write(
                    f"敗率："
                    f"{away_stats['loss_rate'] * 100:.1f}%"
                )

                st.write(
                    f"場均進球："
                    f"{away_stats['avg_goals_for']:.2f}"
                )

                st.write(
                    f"場均失球："
                    f"{away_stats['avg_goals_against']:.2f}"
                )

                st.write(
                    f"平均淨勝球："
                    f"{away_stats['avg_goal_difference']:.2f}"
                )

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
                f"模型參考的直接交手場數："
                f"{h2h_stats['h2h_matches']} 場"
            )

            st.subheader("近期交手紀錄")

            if h2h_matches.empty:
                st.info(
                    "找不到兩隊交手紀錄"
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
                    h2h_table[
                        "home_team"
                    ].map(
                        display_team_name
                    )
                )

                h2h_table["away_team"] = (
                    h2h_table[
                        "away_team"
                    ].map(
                        display_team_name
                    )
                )

                h2h_table["tournament"] = (
                    h2h_table[
                        "tournament"
                    ].map(
                        translate_tournament
                    )
                )

                h2h_table["date"] = (
                    h2h_table[
                        "date"
                    ].dt.strftime(
                        "%Y-%m-%d"
                    )
                )

                h2h_table = h2h_table.rename(
                    columns={
                        "date": "日期",
                        "home_team": "主隊",
                        "away_team": "客隊",
                        "home_score": "主隊進球",
                        "away_score": "客隊進球",
                        "tournament": "賽事",
                    }
                )

                st.dataframe(
                    h2h_table,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "提醒：預測根據歷史比賽資料與模型計算，"
                "不能保證實際比賽結果。"
            )

except FileNotFoundError as error:
    st.error(
        f"找不到必要檔案：{error}"
    )

except Exception as error:
    st.error(
        f"程式發生錯誤：{error}"
         )
