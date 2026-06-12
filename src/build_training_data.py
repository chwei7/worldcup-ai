import pandas as pd


INPUT_FILE = "data/results.csv"
OUTPUT_FILE = "data/training_data.csv"

RECENT_MATCHES = 5
HEAD_TO_HEAD_LIMIT = 5


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
            "matches_played": 0,
        }

    matches_played = len(history)

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
        "win_rate": wins / matches_played,
        "draw_rate": draws / matches_played,
        "loss_rate": losses / matches_played,
        "avg_goals_for": (
            goals_for / matches_played
        ),
        "avg_goals_against": (
            goals_against / matches_played
        ),
        "avg_goal_difference": (
            goals_for - goals_against
        ) / matches_played,
        "matches_played": matches_played,
    }


def calculate_head_to_head(
    completed_matches,
    home_team,
    away_team,
    limit=5,
):
    head_to_head = []

    for match in reversed(completed_matches):
        match_home = match["home_team"]
        match_away = match["away_team"]

        same_pair = (
            (
                match_home == home_team
                and match_away == away_team
            )
            or
            (
                match_home == away_team
                and match_away == home_team
            )
        )

        if not same_pair:
            continue

        head_to_head.append(match)

        if len(head_to_head) >= limit:
            break

    home_wins = 0
    draws = 0
    away_wins = 0

    for match in head_to_head:
        match_home = match["home_team"]
        match_away = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]

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

    count = len(head_to_head)

    if count == 0:
        return {
            "h2h_home_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_matches": 0,
            "h2h_win_rate_difference": 0.0,
        }

    home_win_rate = home_wins / count
    draw_rate = draws / count
    away_win_rate = away_wins / count

    return {
        "h2h_home_win_rate": home_win_rate,
        "h2h_draw_rate": draw_rate,
        "h2h_away_win_rate": away_win_rate,
        "h2h_matches": count,
        "h2h_win_rate_difference": (
            home_win_rate - away_win_rate
        ),
    }


def main():
    matches = pd.read_csv(
        INPUT_FILE,
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

    matches = matches.sort_values(
        "date"
    ).reset_index(drop=True)

    histories = {}
    completed_matches = []
    training_rows = []

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        home_score = int(
            match["home_score"]
        )

        away_score = int(
            match["away_score"]
        )

        if home_team not in histories:
            histories[home_team] = []

        if away_team not in histories:
            histories[away_team] = []

        home_stats = calculate_stats(
            histories[home_team][
                -RECENT_MATCHES:
            ]
        )

        away_stats = calculate_stats(
            histories[away_team][
                -RECENT_MATCHES:
            ]
        )

        h2h_stats = calculate_head_to_head(
            completed_matches,
            home_team,
            away_team,
            limit=HEAD_TO_HEAD_LIMIT,
        )

        if (
            home_stats["matches_played"] >= 3
            and away_stats["matches_played"] >= 3
        ):
            if home_score > away_score:
                target = 2
            elif home_score == away_score:
                target = 1
            else:
                target = 0

            training_rows.append(
                {
                    "date": match["date"],
                    "home_team": home_team,
                    "away_team": away_team,

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

                    "target": target,
                }
            )

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

        completed_matches.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    training_data = pd.DataFrame(
        training_rows
    )

    training_data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("訓練資料建立完成")
    print("資料筆數：", len(training_data))
    print("檔案位置：", OUTPUT_FILE)

    print("\n結果類別：")
    print("0 = 客隊勝")
    print("1 = 平手")
    print("2 = 主隊勝")

    if not training_data.empty:
        print("\n新增的交手特徵：")
        print(
            training_data[
                [
                    "h2h_home_win_rate",
                    "h2h_draw_rate",
                    "h2h_away_win_rate",
                    "h2h_matches",
                    "h2h_win_rate_difference",
                ]
            ].head()
        )


if __name__ == "__main__":
    main()