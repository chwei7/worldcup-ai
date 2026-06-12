import joblib
import pandas as pd


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
    for chinese, english
    in TEAM_NAME_ZH_TO_EN.items()
}


def convert_team_name(user_input):
    user_input = user_input.strip()

    return TEAM_NAME_ZH_TO_EN.get(
        user_input,
        user_input,
    )


def display_team_name(english_name):
    return TEAM_NAME_EN_TO_ZH.get(
        english_name,
        english_name,
    )


def team_result(
    goals_for,
    goals_against,
):
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
        "avg_goals_for":
            goals_for / count,
        "avg_goals_against":
            goals_against / count,
        "avg_goal_difference": (
            goals_for - goals_against
        ) / count,
    }


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


def build_histories(matches):
    histories = {}

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        home_score = int(
            match["home_score"]
        )

        away_score = int(
            match["away_score"]
        )

        histories.setdefault(
            home_team,
            [],
        )

        histories.setdefault(
            away_team,
            [],
        )

        histories[home_team].append(
            {
                "goals_for": home_score,
                "goals_against":
                    away_score,
                "result": team_result(
                    home_score,
                    away_score,
                ),
            }
        )

        histories[away_team].append(
            {
                "goals_for": away_score,
                "goals_against":
                    home_score,
                "result": team_result(
                    away_score,
                    home_score,
                ),
            }
        )

    return histories


def get_head_to_head(
    matches,
    home_team,
    away_team,
    limit=5,
):
    condition1 = (
        (matches["home_team"] == home_team)
        & (matches["away_team"] == away_team)
    )

    condition2 = (
        (matches["home_team"] == away_team)
        & (matches["away_team"] == home_team)
    )

    h2h = matches[
        condition1 | condition2
    ].copy()

    h2h = h2h.sort_values(
        "date",
        ascending=False,
    )

    return h2h.head(limit)


def calculate_h2h_stats(
    h2h_matches,
    home_team,
    away_team,
):
    home_wins = 0
    away_wins = 0
    draws = 0

    for _, match in h2h_matches.iterrows():
        match_home = match["home_team"]
        match_away = match["away_team"]

        home_score = int(
            match["home_score"]
        )

        away_score = int(
            match["away_score"]
        )

        if home_score == away_score:
            draws += 1
            continue

        if home_score > away_score:
            winner = match_home
        else:
            winner = match_away

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

    home_win_rate = (
        home_wins / count
    )

    draw_rate = (
        draws / count
    )

    away_win_rate = (
        away_wins / count
    )

    return {
        "h2h_home_win_rate":
            home_win_rate,

        "h2h_draw_rate":
            draw_rate,

        "h2h_away_win_rate":
            away_win_rate,

        "h2h_matches":
            count,

        "h2h_win_rate_difference": (
            home_win_rate
            - away_win_rate
        ),
    }


def show_head_to_head(
    h2h_matches,
    home_team,
    away_team,
):
    home_display = display_team_name(
        home_team
    )

    away_display = display_team_name(
        away_team
    )

    print(
        "\n========== "
        f"{home_display} vs "
        f"{away_display} "
        "近期交手 =========="
    )

    if h2h_matches.empty:
        print("找不到兩隊交手紀錄")
        return

    for _, match in h2h_matches.iterrows():
        date_text = (
            match["date"].strftime(
                "%Y-%m-%d"
            )
        )

        match_home_display = (
            display_team_name(
                match["home_team"]
            )
        )

        match_away_display = (
            display_team_name(
                match["away_team"]
            )
        )

        home_score = int(
            match["home_score"]
        )

        away_score = int(
            match["away_score"]
        )

        tournament = str(
            match.get(
                "tournament",
                "",
            )
        )

        print(
            f"{date_text}｜"
            f"{match_home_display} "
            f"{home_score}-"
            f"{away_score} "
            f"{match_away_display}｜"
            f"{tournament}"
        )


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
            home_stats[
                "avg_goals_for"
            ],

        "away_avg_goals_for":
            away_stats[
                "avg_goals_for"
            ],

        "home_avg_goals_against":
            home_stats[
                "avg_goals_against"
            ],

        "away_avg_goals_against":
            away_stats[
                "avg_goals_against"
            ],

        "home_avg_goal_difference":
            home_stats[
                "avg_goal_difference"
            ],

        "away_avg_goal_difference":
            away_stats[
                "avg_goal_difference"
            ],

        "diff_win_rate": (
            home_stats["win_rate"]
            - away_stats["win_rate"]
        ),

        "diff_avg_goals_for": (
            home_stats[
                "avg_goals_for"
            ]
            - away_stats[
                "avg_goals_for"
            ]
        ),

        "diff_avg_goals_against": (
            home_stats[
                "avg_goals_against"
            ]
            - away_stats[
                "avg_goals_against"
            ]
        ),

        "diff_goal_difference": (
            home_stats[
                "avg_goal_difference"
            ]
            - away_stats[
                "avg_goal_difference"
            ]
        ),

        "h2h_home_win_rate":
            h2h_stats[
                "h2h_home_win_rate"
            ],

        "h2h_draw_rate":
            h2h_stats[
                "h2h_draw_rate"
            ],

        "h2h_away_win_rate":
            h2h_stats[
                "h2h_away_win_rate"
            ],

        "h2h_matches":
            h2h_stats[
                "h2h_matches"
            ],

        "h2h_win_rate_difference":
            h2h_stats[
                "h2h_win_rate_difference"
            ],
    }

    row = pd.DataFrame([values])

    return row.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )


def show_stats(
    team_display,
    stats,
):
    print(
        f"\n{team_display} 最近 "
        f"{RECENT_MATCHES} 場："
    )

    print(
        f"勝率："
        f"{stats['win_rate'] * 100:.1f}%"
    )

    print(
        f"平手率："
        f"{stats['draw_rate'] * 100:.1f}%"
    )

    print(
        f"敗率："
        f"{stats['loss_rate'] * 100:.1f}%"
    )

    print(
        f"場均進球："
        f"{stats['avg_goals_for']:.2f}"
    )

    print(
        f"場均失球："
        f"{stats['avg_goals_against']:.2f}"
    )


def main():
    try:
        bundle = joblib.load(
            MODEL_FILE
        )

        model = bundle["model"]

        feature_columns = bundle[
            "feature_columns"
        ]

        matches = load_matches()

        histories = build_histories(
            matches
        )

        print("AI 足球比賽預測程式")
        print("可以輸入中文或英文球隊名稱。")

        home_input = input(
            "\n請輸入第一支球隊："
        )

        away_input = input(
            "請輸入第二支球隊："
        )

        home_team = convert_team_name(
            home_input
        )

        away_team = convert_team_name(
            away_input
        )

        if home_team == away_team:
            print("不能選擇相同球隊")
            return

        if home_team not in histories:
            print(
                f"找不到球隊：{home_input}"
            )
            return

        if away_team not in histories:
            print(
                f"找不到球隊：{away_input}"
            )
            return

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
            limit=HEAD_TO_HEAD_LIMIT,
        )

        h2h_stats = calculate_h2h_stats(
            h2h_matches,
            home_team,
            away_team,
        )

        show_head_to_head(
            h2h_matches,
            home_team,
            away_team,
        )

        home_display = display_team_name(
            home_team
        )

        away_display = display_team_name(
            away_team
        )

        show_stats(
            home_display,
            home_stats,
        )

        show_stats(
            away_display,
            away_stats,
        )

        print("\n近期交手統計：")

        print(
            f"{home_display}勝率："
            f"{h2h_stats['h2h_home_win_rate'] * 100:.1f}%"
        )

        print(
            f"平手率："
            f"{h2h_stats['h2h_draw_rate'] * 100:.1f}%"
        )

        print(
            f"{away_display}勝率："
            f"{h2h_stats['h2h_away_win_rate'] * 100:.1f}%"
        )

        print(
            f"交手場數："
            f"{h2h_stats['h2h_matches']}"
        )

        X = create_features(
            home_stats,
            away_stats,
            h2h_stats,
            feature_columns,
        )

        probabilities = (
            model.predict_proba(X)[0]
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

        print(
            "\n========== "
            "AI 預測 =========="
        )

        print(
            f"{home_display} 勝："
            f"{home_win:.1f}%"
        )

        print(
            f"平手："
            f"{draw:.1f}%"
        )

        print(
            f"{away_display} 勝："
            f"{away_win:.1f}%"
        )

        results = {
            f"{home_display}勝":
                home_win,

            "平手":
                draw,

            f"{away_display}勝":
                away_win,
        }

        most_likely = max(
            results,
            key=results.get,
        )

        print(
            "\n最可能結果：",
            most_likely,
        )

        print("\n模型參考因素：")
        print("- 最近 5 場勝和負")
        print("- 最近 5 場進球與失球")
        print("- 平均淨勝球")
        print("- 兩隊近期直接交手紀錄")

        print(
            "\n提醒：這是模型預測，"
            "不是實際比賽結果保證。"
        )

    except FileNotFoundError as error:
        print(
            "找不到必要檔案：",
            error,
        )

    except Exception as error:
        print(
            "程式發生錯誤：",
            error,
        )


if __name__ == "__main__":
    main()