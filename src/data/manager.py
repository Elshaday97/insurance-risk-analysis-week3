import pandas as pd
from pathlib import Path
from scripts.constants import (
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    CLEANED_DATA_FILE_NAME,
    RAW_DATA_FILE_NAME,
    RAW_CSV_DATA_FILE_NAME,
    NON_NEGATIVE_COLS,
    Columns,
)
from scripts import parsers


class DataManager:
    def __init__(self):
        self.raw_data_path = Path(RAW_DATA_DIR)
        self.raw_data_file_name = Path(RAW_DATA_FILE_NAME)
        self.raw_csv_data_file_name = Path(RAW_CSV_DATA_FILE_NAME)
        self.cleaned_data_path = Path(CLEANED_DATA_DIR)
        self.cleaned_data_file_name = Path(CLEANED_DATA_FILE_NAME)
        self.df: pd.DataFrame | None = None

    def load_csv(self, load_clean=False) -> pd.DataFrame:
        try:
            path = (
                self.cleaned_data_path / self.cleaned_data_file_name
                if load_clean
                else self.raw_data_path / self.raw_data_file_name
            )
            sep = None if load_clean else "|"
            self.df = (
                pd.read_csv(path, index_col=0)
                if load_clean
                else pd.read_csv(path, sep=sep, low_memory=False)
            )

            print("Basic Data Info:\n")
            self.df.info()

            if not load_clean:
                self.save_to_csv(
                    df=self.df,
                    file_name=self.raw_csv_data_file_name,
                    file_path=self.raw_data_path,
                )
            print("Data loaded!")
        except ValueError as e:
            print(f"Unable to load data: {e}")
            raise e

        return self.df

    def _clean_data(self) -> pd.DataFrame:
        """
        Data Cleaning Pipeline
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Call load_csv() first.")
        self.df = (
            self.df.pipe(self._handle_missing)
            .pipe(self._convert_dtypes)
            .pipe(self._detect_outliers)
            .pipe(self._remove_duplicates)
            .pipe(self._ensure_non_negative)
        )
        return self.df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        print("Handling Missing Values...")
        # [DROP] Vehicle Identification Fields that are important, drop if any null values in these columns
        drop_cols = [
            Columns.mmcode.value,
            Columns.make.value,
            Columns.Model.value,
            Columns.SumInsured.value,
            Columns.TotalPremium.value,
            Columns.NewVehicle.value,
        ]
        df_clean.dropna(subset=drop_cols, inplace=True)

        # [MODE] Low cardinality fields - Bank, AccountType, Marital Status, Gender, Vehicle Type, bodytype
        mode_cols = [
            Columns.Bank.value,
            Columns.AccountType.value,
            Columns.MaritalStatus.value,
            Columns.Gender.value,
            Columns.VehicleType.value,
            Columns.bodytype.value,
        ]
        for mode_col in mode_cols:
            if mode_col in self.df.columns:
                df_clean[mode_col] = df_clean[mode_col].fillna(
                    df_clean[mode_col].mode()[0]
                )

        # [MEDIAN] CustomValueEstimate (Vehicle market value), Cylinder, cubiccapacity, kilowatts, Number of doors
        median_cols = [
            Columns.CustomValueEstimate.value,
            Columns.Cylinders.value,
            Columns.cubiccapacity.value,
            Columns.kilowatts.value,
            Columns.NumberOfDoors.value,
        ]
        for median_col in median_cols:
            if median_col in df_clean.columns:
                df_clean[median_col] = df_clean[median_col].fillna(
                    df_clean[median_col].median()
                )

        # [LEAVE NULL] VehicleIntroDate
        df_clean[Columns.VehicleIntroDate.value] = df_clean[
            Columns.VehicleIntroDate.value
        ].where(df_clean[Columns.VehicleIntroDate.value].notna(), None)

        # [FILL WITH 0] Capital Outstanding, Total Claim
        fill_zero_cols = [Columns.CapitalOutstanding.value, Columns.TotalClaims.value]
        for fill_zero_col in fill_zero_cols:
            if fill_zero_col in df_clean.columns:
                df_clean[fill_zero_col] = df_clean[fill_zero_col].fillna(0)

        # [FILL WITH FALSE] WrittenOff, Rebuilt, Converted, CrossBorder
        fill_false_cols = [
            Columns.WrittenOff.value,
            Columns.Rebuilt.value,
            Columns.Converted.value,
            Columns.CrossBorder.value,
        ]
        for fill_false_col in fill_false_cols:
            if fill_false_col in df_clean.columns:
                df_clean[fill_false_col] = df_clean[fill_false_col].fillna(False)

        # [Drop Col] NumberOfVehiclesInFleet
        df_clean.drop(columns=[Columns.NumberOfVehiclesInFleet.value], inplace=True)

        # [FILL WITH None] Location related
        fill_null_cols = [
            Columns.Province.value,
            Columns.PostalCode.value,
            Columns.MainCrestaZone.value,
            Columns.SubCrestaZone.value,
        ]
        for fill_null_col in fill_null_cols:
            if fill_null_col in df_clean.columns:
                df_clean[fill_null_col] = df_clean[fill_null_col].fillna("Unknown")

        print("Missing values handled!")
        return df_clean

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Converting data types...")
        df = parsers.parse_date_columns(df)
        df = parsers.parse_yes_no_cols(df)
        df = parsers.parse_numeric_cols(df)
        print("Data types converted successfully!")
        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Detecting outliers...")
        df = parsers.flag_outliers(df)
        print("Outliers detected and flagged!")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Removing Duplicates...")
        df_clean = df.drop_duplicates()
        print("Duplicates removed...")
        return df_clean

    def _ensure_non_negative(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Dropping rows with negative values...")
        df_clean = df.copy()
        mask = pd.Series(False, index=df_clean.index)
        for non_negative_col in NON_NEGATIVE_COLS:
            if non_negative_col in df_clean.columns:
                mask |= df_clean[non_negative_col] < 0

        df_clean = df_clean[~mask].copy()
        print(f"Dropped {mask.sum()} rows with negative values!")
        return df_clean

    def save_to_csv(
        self,
        df: pd.DataFrame = None,
        file_path: str = "",
        file_name: str = "",
    ):
        try:
            if df is None:
                df = self.df
            path = (
                file_path / file_name
                if file_name and file_path
                else self.cleaned_data_path / self.cleaned_data_file_name
            )
            df.to_csv(path)
            print(f"File saved to {path} successfully!")
        except ValueError as e:
            print(f"Unable to save data: {e}")
            raise e

    def get_data(self):
        return self.df

    def load_data(self) -> pd.DataFrame:
        self.load_csv()
        self._clean_data()
        self.save_to_csv()
        clean_df = self.get_data()
        return clean_df
