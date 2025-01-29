from src.utils.preprocess import process_labels, reshape_input_eeg
from src.models.cnn import CNN_big, train_model_early_stopping
from src.models.pca import pca
from src.models.svm import 


out_cohesion = process_labels("Averaged Cohesion scores.csv", "labels.csv")


fig = px.scatter(
    out_cohesion,
    x="pair",
    y="Labels",
    color="Binary Labels",
    labels={"X": "X-Axis Label", "Y": "Y-Axis Label"},
    title="Anerage score distribution per pair and binary selection threshold",
)


fig.add_shape(
    type="line",
    x0=0,
    x1=44,
    y0=4.5,
    y1=4.5,
    line=dict(color="blue", width=2, dash="dash"),
)


fig.add_annotation(
    x=22, y=4.7, text="Threshold: 4.5", showarrow=False, font=dict(size=12, color="black"), align="center"
)

################################

data = pd.read_csv("labels.csv") ??????????????
label_counts = data.value_counts()

if "Labels" in data.columns:
    label_counts = data["Labels"].value_counts()

    # Create a Plotly pie chart
    fig = px.pie(
        names=label_counts.index,
        values=label_counts.values,
        title="Distribution of binary Labels for questionnaire results. The threshold is 4.5 (for ranks 1-6)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.show()
else:
    print("Column 'label' not found in the CSV file. Please check the file structure.")

################################

reshape_input_eeg("correlations_array.csv", "reshaped_correlations.csv", has_part=False)
reshape_input_eeg("correlations_array5.csv", "reshaped_correlations5.csv", has_part=True)
reshape_input_eeg("correlations_array10.csv", "reshaped_correlations10.csv", has_part=True)
reshape_input_eeg("correlations_array60.csv", "reshaped_correlations60.csv", has_part=True)
reshape_input_eeg("correlations_array120.csv", "reshaped_correlations120.csv", has_part=True)

################################

best_model, best_params, best_score = train_svm("reshaped_correlations120.csv", "labels.csv")

################################

results_df.head()

heatmap_data = results_df.pivot_table(index="param_svc__kernel", columns="param_svc__gamma", values="mean_test_f1")

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
plt.title("F1 Score Heatmap")
plt.xlabel("Gamma")
plt.ylabel("Kernel type")
plt.show()

################################

if "mean_test_f1" in results_df.columns:
    plot_data = pd.DataFrame({"Index": range(len(results_df["mean_test_f1"])), "F1": results_df["mean_test_f1"]})

    # Find the best F1 score
    best_f1 = plot_data["F1"].max()

    fig = px.line(
        plot_data,
        x="Index",
        y="F1",
        title="F1 Score Evolution During Grid Search",
        labels={"Index": "Parameter Combination Index", "F1": "F1 Score"},
        markers=True,
    )

    fig.add_shape(
        type="line",
        x0=plot_data["Index"].min(),
        x1=plot_data["Index"].max(),
        y0=best_f1,
        y1=best_f1,
        line=dict(color="red", width=2, dash="dash"),
        name="Best F1 Score",
    )

    fig.add_annotation(
        x=plot_data["Index"].max() - 1,
        y=best_f1 + 0.02,
        text=f"Best F1 Score: {best_f1:.3f}",
        showarrow=False,
        font=dict(size=12, color="red"),
        align="right",
    )

    fig.update_layout(margin=dict(r=120), title=dict(x=0.5))

    fig.show()

################################

datasets = [
    ("reshaped_correlations.csv", "labels.csv"),
    ("reshaped_correlations10.csv", "labels.csv"),
    ("reshaped_correlations120.csv", "labels.csv"),
    ("reshaped_correlations5.csv", "labels.csv"),
    ("reshaped_correlations60.csv", "labels.csv"),
]

results_df = multi_datasets(datasets)
results_df

################################

dataset_scores = results_df.groupby("Dataset", as_index=False)["Best F1 Score"].mean()

fig = px.bar(
    dataset_scores,
    x="Best F1 Score",
    y="Dataset",
    orientation="h",
    color="Dataset",
    text="Best F1 Score",
    title="Best F1 Score by Dataset",
)

fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.update_layout(
    xaxis=dict(title="Best F1 Score", range=[0.7, dataset_scores["Best F1 Score"].max() + 0.02]),
    yaxis_title="Dataset",
    showlegend=False,
    height=1.2 * len(results_df["Dataset"].unique()) * 100,
)

fig.show()

################################

pca("reshaped_correlations.csv", 15)

################################

epochs_range = list(range(1, 51))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df_Y = pd.read_csv("labels.csv")
df_X = pd.read_csv("reshaped_correlations120.csv")

# Drop unnecessary columns
df_X.drop(columns=["Pair"], inplace=True, errors="ignore")
df_X = df_X.dropna()

# Data normalization using StandardScaler
scaler = StandardScaler()
df_X = pd.DataFrame(scaler.fit_transform(df_X), columns=df_X.columns)

# Convert to numpy arrays
X_np = df_X.values.astype(np.float32)
y_np = df_Y.values.astype(np.int64)

X_np = X_np.reshape(43, 5, 120, 8)  # (43 paires, 5 bands, 120 timeslots, 8 electrodes)
X_np = X_np.reshape(43, 5 * 8, 120)

# Convert to tensors
X_tensor = torch.tensor(X_np)
y_tensor = torch.tensor(y_np)

################################

# Parameters and cross-validation
# Standard for small dataset

batch_size = 16
lr = 0.001
epochs = 50
patience = 20
min_delta = 0.0
criterion = nn.BCELoss()

dataset = TensorDataset(X_tensor, y_tensor)
fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):  # for also getting the index
    print(f"Fold {fold + 1}")

    model = CNN_big()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False
    )  # For saving the time and calculation, we don't shuffle the validation set

    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Fold {fold + 1} Loss", f"Fold {fold + 1} Accuracy"])

    # Plot Losses
    fig.add_trace(go.Scatter(x=epochs_range, y=train_losses, mode="lines+markers", name="Train Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_range, y=val_losses, mode="lines+markers", name="Val Loss"), row=1, col=1)

    # Plot Accuracies
    fig.add_trace(go.Scatter(x=epochs_range, y=train_accs, mode="lines+markers", name="Train Acc"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs_range, y=val_accs, mode="lines+markers", name="Val Acc"), row=1, col=2)

    # Update layout for better visuals
    fig.update_layout(
        height=500,
        width=1000,  # Adjust figure size
        title_text=f"Results for Fold {fold + 1}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis2_title="Epoch",
        yaxis2_title="Accuracy",
        legend_title="Legend",
        showlegend=True,
    )

    # Show the plot
    fig.show()

    fold_val_losses.append(val_losses[-1])
    fold_val_accs.append(val_accs[-1])
    fold_val_f1s.append(val_f1s[-1])

mean_loss = np.mean(fold_val_losses)  # For calculating the average of all folds
mean_acc = np.mean(fold_val_accs)
mean_f1 = np.mean(fold_val_f1s)

print(f"Average loss: {mean_loss:.4f}")
print(f"Average accuracy: {mean_acc:.4f}")
print(f"Average F1 score: {mean_f1:.4f}")
