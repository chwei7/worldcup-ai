import pandas as pd


DATA_FILE = "data/worldcup_team_stats.csv"


def load_teams():
    teams = pd.read_csv(DATA_FILE)

    teams["team_lower"] = (
        teams["team"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return teams


def find_team(teams, team_name):
    team_name = team_name.strip().lower()

    result = teams[
        teams["team_lower"] == team_name
    ]

    if result.empty:
        return None

    return result.iloc[0]


def show_team(team):
    print(f"\n球隊：{team['team']}")
    print(f"比賽場數：{int(team['matches'])}")
    print(f"勝場：{int(team['wins'])}")
    print(f"平手：{int(team['draws'])}")
    print(f"敗場：{int(team['losses'])}")
    print(f"進球：{int(team['goals_for'])}")
    print(f"失球：{int(team['goals_against'])}")
    print(f"淨勝球：{int(team['goal_difference'])}")
    print(f"勝率：{team['win_rate']:.1f}%")
    print(f"平手率：{team['draw_rate']:.1f}%")
    print(f"敗率：{team['loss_rate']:.1f}%")
    print(f"場均進球：{team['avg_goals_for']:.2f}")
    print(
        f"場均失球："
        f"{team['avg_goals_against']:.2f}"
    )


def calculate_strength(team):
    win_score = team["win_rate"] * 0.45

    goal_difference_score = (
        team["goal_difference"]
        / team["matches"]
        * 15
    )

    attack_score = (
        team["avg_goals_for"] * 15
    )

    defense_score = (
        max(
            0,
            3 - team["avg_goals_against"]
        )
        * 10
    )

    return (
        win_score
        + goal_difference_score
        + attack_score
        + defense_score
    )


def explain_result(team1, team2):
    reasons = []

    if team1["win_rate"] > team2["win_rate"]:
        reasons.append(
            f"{team1['team']} 的勝率較高"
        )
    elif team2["win_rate"] > team1["win_rate"]:
        reasons.append(
            f"{team2['team']} 的勝率較高"
        )

    if (
        team1["avg_goals_for"]
        > team2["avg_goals_for"]
    ):
        reasons.append(
            f"{team1['team']} 的場均進球較高"
        )
    elif (
        team2["avg_goals_for"]
        > team1["avg_goals_for"]
    ):
        reasons.append(
            f"{team2['team']} 的場均進球較高"
        )

    if (
        team1["avg_goals_against"]
        < team2["avg_goals_against"]
    ):
        reasons.append(
            f"{team1['team']} 的場均失球較少"
        )
    elif (
        team2["avg_goals_against"]
        < team1["avg_goals_against"]
    ):
        reasons.append(
            f"{team2['team']} 的場均失球較少"
        )

    if (
        team1["goal_difference"]
        > team2["goal_difference"]
    ):
        reasons.append(
            f"{team1['team']} 的淨勝球較好"
        )
    elif (
        team2["goal_difference"]
        > team1["goal_difference"]
    ):
        reasons.append(
            f"{team2['team']} 的淨勝球較好"
        )

    return reasons[:3]


def compare_teams(team1, team2):
    show_team(team1)
    show_team(team2)

    team1_strength = calculate_strength(team1)
    team2_strength = calculate_strength(team2)

    total_strength = (
        team1_strength + team2_strength
    )

    if total_strength <= 0:
        team1_probability = 50
        team2_probability = 50
    else:
        team1_probability = (
            team1_strength
            / total_strength
            * 100
        )

        team2_probability = (
            team2_strength
            / total_strength
            * 100
        )

    print("\n========== 預測結果 ==========")
    print(
        f"{team1['team']}："
        f"{team1_probability:.1f}%"
    )
    print(
        f"{team2['team']}："
        f"{team2_probability:.1f}%"
    )

    if abs(
        team1_probability - team2_probability
    ) < 3:
        print("\n兩隊實力非常接近")

    elif (
        team1_probability
        > team2_probability
    ):
        print(
            f"\n{team1['team']} "
            f"目前預測勝率較高"
        )

    else:
        print(
            f"\n{team2['team']} "
            f"目前預測勝率較高"
        )

    print("\n主要原因：")

    reasons = explain_result(team1, team2)

    for reason in reasons:
        print("-", reason)


def main():
    try:
        teams = load_teams()

        print("世界盃球隊比較程式")

        print("\n目前可選球隊：")

        for team_name in teams["team"]:
            print("-", team_name)

        team1_name = input(
            "\n請輸入第一支球隊："
        )

        team2_name = input(
            "請輸入第二支球隊："
        )

        team1 = find_team(
            teams,
            team1_name
        )

        team2 = find_team(
            teams,
            team2_name
        )

        if team1 is None:
            print(
                f"\n找不到球隊："
                f"{team1_name}"
            )
            return

        if team2 is None:
            print(
                f"\n找不到球隊："
                f"{team2_name}"
            )
            return

        if (
            team1["team_lower"]
            == team2["team_lower"]
        ):
            print(
                "\n不能選擇相同的球隊"
            )
            return

        compare_teams(team1, team2)

    except FileNotFoundError:
        print(
            "找不到 "
            "data/worldcup_team_stats.csv"
        )

    except Exception as error:
        print(
            "程式發生錯誤：",
            error
        )


if __name__ == "__main__":
    main()