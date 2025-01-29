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
    out_cohesion = pd.DataFrame({"pair": range(1, 44), "Labels_nonbi": pair_mean, "Labels": cohesion_binary})

    out_cohesion_binary = pd.DataFrame({"Labels": cohesion_binary})

    out_cohesion_binary.to_csv(output, index=False)
    return out_cohesion


def reshape_input_eeg(input_file: str, output_file, has_part=True) -> None:
    """
    Reshape table
    """
    df = pd.read_csv(input_file, header=None)

    if has_part:  # "part" means we dissect the data by time, this is how we name it in the CSV file
        df.columns = ["PairPart", "Band"] + [f"Electrode{i}" for i in range(1, 9)]  # Give new columns name

        pair_part = df["PairPart"].str.extract(r"(pair\s*\d+)\s+part\s*(\d+)", expand=True)
        df["Pair"] = pair_part[0]
        df["Part"] = pair_part[1].astype(int)
        df.drop("PairPart", axis=1, inplace=True)

    else:
        df.columns = ["Band"] + [f"Electrode{i}" for i in range(1, 9)]

        num_rows = len(df)
        bands_per_pair = 5
        num_pairs = num_rows // bands_per_pair
        if num_rows % bands_per_pair != 0:
            raise ValueError(
                "check the rows!"
            )  # Add Pair column for the rows since there is no Pair column in this format
        df["Pair"] = ["pair " + str(i // bands_per_pair + 1) for i in range(len(df))]

    id_columns = ["Pair", "Band"]
    if has_part:
        id_columns.append("Part")

    melted = df.melt(
        id_vars=id_columns,
        value_vars=[f"Electrode{i}" for i in range(1, 9)],
        var_name="Electrode",
        value_name="Correlation",
    )

    # Pivot the table
    if has_part:
        pivoted = melted.pivot_table(
            index="Pair", columns=["Band", "Part", "Electrode"], values="Correlation", sort=False
        )

        pivoted.columns = [f"{band}_T{part}_{electrode}" for (band, part, electrode) in pivoted.columns]
    else:
        pivoted = melted.pivot_table(index="Pair", columns=["Band", "Electrode"], values="Correlation", sort=False)

        pivoted.columns = [f"{band}_{electrode}" for (band, electrode) in pivoted.columns]

    pivoted.reset_index(inplace=True)
    pivoted.drop(pivoted[pivoted["Pair"] == "pair 1"].index, inplace=True)
    df_cleaned = df.dropna()

    # Save the result
    pivoted.to_csv(output_file, index=False)
    print(f"The input csv processing is done as {output_file}")
