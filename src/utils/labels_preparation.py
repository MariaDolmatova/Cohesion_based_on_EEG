import pandas as pd


def process_labels(input, output):
    """
    Turn average cohesion score into a binary score
    """
    df = pd.read_csv(input)
    cohesion_score = df["Average cohesion score"].values  # convert to ndarray
    cohesion_pair = cohesion_score.reshape(-1, 2)
    pair_mean = cohesion_pair.mean(axis=1)
    cohesion_binary = (pair_mean > 4.5).astype(int)
    out_cohesion = pd.DataFrame({"pair": range(1, 44), "Labels_nonbinary": pair_mean, "Labels": cohesion_binary})

    out_cohesion_binary = pd.DataFrame({"Labels": cohesion_binary})

    out_cohesion_binary.to_csv(output, index=False)
    return out_cohesion
