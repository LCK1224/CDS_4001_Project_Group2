from Linear_Regression import DataProcessor


def main():
    processor = DataProcessor("merge.csv")
    processor.fill_missing_values("SO2", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values("NOX", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values("NO2", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values("CO", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values("RSP", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values("O3", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.fill_missing_values("FSP", list(
        range(5, 22)), non_zero=True, round_result=True, round_digit=0)
    processor.save_to_csv("cleaned_dataset.csv")


if __name__ == "__main__":
    main()
