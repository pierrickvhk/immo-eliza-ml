from data_utils import load_raw_data, basic_clean, save_clean_data

RAW_PATH = "../data/raw/immo_eliza_raw.csv"
CLEAN_PATH = "../data/processed/immo_eliza_clean.csv"

if __name__ == "__main__":
    df_raw = load_raw_data(RAW_PATH)
    df_clean = basic_clean(df_raw)
    save_clean_data(df_clean, CLEAN_PATH)
    print(f"Cleaned data saved to {CLEAN_PATH}")