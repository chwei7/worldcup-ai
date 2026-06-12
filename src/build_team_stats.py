import pandas as pd

INPUT_FILE = "data/worldcup_matches.csv"
OUTPUT_FILE = "data/worldcup_team_stats.csv"


def main():
    matches = pd.read_csv(INPUT_FILE)

    stats = {}

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]

        # 尚未完成或沒有比分的比賽先跳過
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        for team in [home_team, away_team]:
            if team not in stats:
                stats[team] = {
                    "team": team,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                }

        home_score = int(home_score)
        away_score = int(away_score)

        stats[home_team]["goals_for"] += home_score
        stats[home_team]["goals_against"] += away_score

        stats[away_team]["goals_for"] += away_score
        stats[away_team]["goals_against"] += home_score

        if home_score > away_score:
            stats[home_team]["wins"] += 1
            stats[away_team]["losses"] += 1

        elif away_score > home_score:
            stats[away_team]["wins"] += 1
            stats[home_team]["losses"] += 1

        else:
            stats[home_team]["draws"] += 1
            stats[away_team]["draws"] += 1

    teams = pd.DataFrame(stats.values())

    teams["matches"] = (
        teams["wins"]
        + teams["draws"]
        + teams["losses"]
    )

    teams["win_rate"] = (
        teams["wins"]
        / teams["matches"]
        * 100
    )

    teams["draw_rate"] = (
        teams["draws"]
        / teams["matches"]
        * 100
    )

    teams["loss_rate"] = (
        teams["losses"]
        / teams["matches"]
        * 100
    )

    teams["goal_difference"] = (
        teams["goals_for"]
        - teams["goals_against"]
    )

    teams["avg_goals_for"] = (
        teams["goals_for"]
        / teams["matches"]
    )

    teams["avg_goals_against"] = (
        teams["goals_against"]
        / teams["matches"]
    )

    teams = teams.sort_values(
        by=["win_rate", "goal_difference"],
        ascending=[False, False],
    )

    teams.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("球隊統計建立完成")
    print("球隊數量：", len(teams))
    print("檔案位置：", OUTPUT_FILE)

    print("\n前 10 名球隊：")
    print(
        teams[
            [
                "team",
                "matches",
                "wins",
                "draws",
                "losses",
                "win_rate",
                "goal_difference",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()