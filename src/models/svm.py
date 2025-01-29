import pandas as pd
from sklearn.pipeline import Pipeline


def train_svm(input, output):
    global results_df

    df_X = pd.read_csv(input)
    df_Y = pd.read_csv(output)

    df_X.drop(columns=["Pair"], inplace=True, errors="ignore")
    df_X = df_X.dropna()

    X = df_X.values
    y = df_Y.iloc[:, 0].values

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("svc", SVC(probability=True, random_state=SEED)),
        ]
    )

    param_grid = {
        "select__k": [5, 40, "all"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma": ["scale", 0.1, 1, 10],
    }

    scoring = {"f1": make_scorer(f1_score)}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=5,
    )

    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    results_df = pd.DataFrame(grid_search.cv_results_)

    print("Best Parameters:", grid_search.best_params_)
    print("Best F1 Score:", grid_search.best_score_)

    return best_model, best_params, best_score


def multi_datasets(datasets):
    summary_list = []

    for input, output in datasets:
        try:
            best_model, best_params, best_score = train_svm(input, output)
            columns = {
                "Dataset": input,
                "K Value": best_params.get("select__k", None),
                "C": best_params.get("svc__C", None),
                "Gamma": best_params.get("svc__gamma", None),
                "Kernel": best_params.get("svc__kernel", None),
                "Best F1 Score": best_score,
            }
            summary_list.append(columns)

        except Exception as e:
            print(f"Check the {input}: {e}")

    summary_df = pd.DataFrame(summary_list)

    def number(series):
        return series.str.extract(r"(\d+)").fillna(0).astype(int)[0]

    summary_df = summary_df.sort_values(by="Dataset", key=number)

    return summary_df
