import pandas as pd

def percent_non_english_best_model(csv_path="experiments/output/v2/results.csv"):
    """
    Calculates the percentage of grammatical phenomena where the highest performing model was NOT 'english'.
    Also prints a table with, for every task, the accuracy of english and the accuracy/model of the best model.
    """
    df = pd.read_csv(csv_path)
    # Get the row with the highest accuracy for each grammatical phenomenon
    idx = df.groupby("grammatical phenomenon")["accuracy"].idxmax()
    best_per_phenomenon = df.loc[idx]

    # Get the english model for each phenomenon
    english_per_phenomenon = (
        df[df["model name"] == "english"]
        .set_index("grammatical phenomenon")
        .loc[best_per_phenomenon["grammatical phenomenon"]]
    )

    # Print the table
    print(f"{'Phenomenon':35} | {'English Acc':>11} | {'Best Model':>15} | {'Best Acc':>8}")
    print("-" * 80)
    for _, row in best_per_phenomenon.iterrows():
        phenomenon = row["grammatical phenomenon"]
        best_model = row["model name"]
        best_acc = row["accuracy"]
        english_acc = english_per_phenomenon.loc[phenomenon, "accuracy"]
        print(f"{phenomenon:35} | {english_acc:11.3f} | {best_model:15} | {best_acc:8.3f}")

    # Count how many times the best model is not 'english'
    non_english_count = (best_per_phenomenon["model name"] != "english").sum()
    total = len(best_per_phenomenon)
    percent = (non_english_count / total) * 100 if total > 0 else 0
    print(f"\nOut of {total} phenomena, {non_english_count} ({percent:.1f}%) had a best model that was NOT 'english'.")
    return percent

def main():
    percent_non_english_best_model()

if __name__ == "__main__":
    main()