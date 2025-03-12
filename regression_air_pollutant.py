from Linear_Regression import DataProcessor


def main():
    processor = DataProcessor("merge.csv")
    col_range = list(range(5, 11)) + list(range(12, 22))
    # list(range(5,11)) + list(range(12,22))
    pollutant_lst = ["SO2", "NOX", "NO2", "CO", "RSP", "O3", "FSP"]
    for p in pollutant_lst:
        processor.fill_missing_values(
            p, col_range, non_zero=True, round_result=True, round_digit=0)
    processor.save_to_csv("cleaned_dataset.csv")


if __name__ == "__main__":
    main()
