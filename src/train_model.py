import os

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)


INPUT_FILE = "data/training_data.csv"
MODEL_FILE = "models/worldcup_model.joblib"


FEATURE_COLUMNS = [
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "home_loss_rate",
    "away_loss_rate",
    "home_avg_goals_for",
    "away_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_against",
    "home_avg_goal_difference",
    "away_avg_goal_difference",
    "diff_win_rate",
    "diff_avg_goals_for",
    "diff_avg_goals_against",
    "diff_goal_difference",

    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_matches",
    "h2h_win_rate_difference",
]


def main():
    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    if len(data) < 100:
        print("訓練資料太少")
        return

    split_index = int(
        len(data) * 0.8
    )

    train_data = data.iloc[
        :split_index
    ]

    test_data = data.iloc[
        split_index:
    ]

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "target"
    ]

    X_test = test_data[
        FEATURE_COLUMNS
    ]

    y_test = test_data[
        "target"
    ]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("開始訓練模型...")

    model.fit(
        X_train,
        y_train,
    )

    predicted_classes = model.predict(
        X_test
    )

    predicted_probabilities = (
        model.predict_proba(X_test)
    )

    accuracy = accuracy_score(
        y_test,
        predicted_classes,
    )

    loss = log_loss(
        y_test,
        predicted_probabilities,
        labels=[0, 1, 2],
    )

    print("\n訓練完成")
    print(
        "訓練資料筆數：",
        len(train_data),
    )

    print(
        "測試資料筆數：",
        len(test_data),
    )

    print(
        "準確率：",
        round(accuracy * 100, 2),
        "%",
    )

    print(
        "Log Loss：",
        round(loss, 4),
    )

    print("\n分類結果：")

    print(
        classification_report(
            y_test,
            predicted_classes,
            labels=[0, 1, 2],
            target_names=[
                "客隊勝",
                "平手",
                "主隊勝",
            ],
            zero_division=0,
        )
    )

    importance_data = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance":
                model.feature_importances_,
        }
    )

    importance_data = (
        importance_data.sort_values(
            "importance",
            ascending=False,
        )
    )

    print("\n前 10 個重要特徵：")
    print(
        importance_data.head(10)
    )

    os.makedirs(
        "models",
        exist_ok=True,
    )

    bundle = {
        "model": model,
        "feature_columns":
            FEATURE_COLUMNS,
        "feature_importance":
            importance_data,
    }

    joblib.dump(
    bundle,
    MODEL_FILE,
    compress=3,
)

    print(
        "\n模型已儲存到：",
        MODEL_FILE,
    )


if __name__ == "__main__":
    main()