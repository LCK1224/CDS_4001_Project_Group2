from Linear_Regression import DataProcessor


def main():
    processor = DataProcessor("merge.csv")
    col_range = list(range(5, 11)) + list(range(12, 22))
    # list(range(5,11)) + list(range(12,22))
    processor.fill_missing_values(
        "SO2", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values(
        "NOX", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values(
        "NO2", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values(
        "CO", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values(
        "RSP", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values(
        "O3", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values(
        "FSP", col_range, non_zero=True, round_result=True, round_digit=0)
    processor.save_to_csv("cleaned_dataset.csv")


if __name__ == "__main__":
    main()
