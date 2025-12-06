from enum import Enum

RAW_DATA_DIR = "../data/raw"
RAW_DATA_FILE_NAME = "MachineLearningRating_v3 2.txt"
RAW_CSV_DATA_FILE_NAME = "raw.csv"


CLEANED_DATA_DIR = "../data/processed"
CLEANED_DATA_FILE_NAME = "cleaned_data.csv"


class Columns(Enum):
    UnderwrittenCoverID = "UnderwrittenCoverID"
    PolicyID = "PolicyID"
    TransactionMonth = "TransactionMonth"
    IsVATRegistered = "IsVATRegistered"
    Citizenship = "Citizenship"
    LegalType = "LegalType"
    Title = "Title"
    Language = "Language"
    Bank = "Bank"
    AccountType = "AccountType"
    MaritalStatus = "MaritalStatus"
    Gender = "Gender"
    Country = "Country"
    Province = "Province"
    PostalCode = "PostalCode"
    MainCrestaZone = "MainCrestaZone"
    SubCrestaZone = "SubCrestaZone"
    ItemType = "ItemType"
    mmcode = "mmcode"
    VehicleType = "VehicleType"
    RegistrationYear = "RegistrationYear"
    make = "make"
    Model = "Model"
    Cylinders = "Cylinders"
    cubiccapacity = "cubiccapacity"
    kilowatts = "kilowatts"
    bodytype = "bodytype"
    NumberOfDoors = "NumberOfDoors"
    VehicleIntroDate = "VehicleIntroDate"
    CustomValueEstimate = "CustomValueEstimate"
    AlarmImmobiliser = "AlarmImmobiliser"
    TrackingDevice = "TrackingDevice"
    CapitalOutstanding = "CapitalOutstanding"
    NewVehicle = "NewVehicle"
    WrittenOff = "WrittenOff"
    Rebuilt = "Rebuilt"
    Converted = "Converted"
    CrossBorder = "CrossBorder"
    NumberOfVehiclesInFleet = "NumberOfVehiclesInFleet"
    SumInsured = "SumInsured"
    TermFrequency = "TermFrequency"
    CalculatedPremiumPerTerm = "CalculatedPremiumPerTerm"
    ExcessSelected = "ExcessSelected"
    CoverCategory = "CoverCategory"
    CoverType = "CoverType"
    CoverGroup = "CoverGroup"
    Section = "Section"
    Product = "Product"
    StatutoryClass = "StatutoryClass"
    StatutoryRiskType = "StatutoryRiskType"
    TotalPremium = "TotalPremium"
    TotalClaims = "TotalClaims"


DATE_COLS = [Columns.TransactionMonth.value, Columns.VehicleIntroDate.value]
BOOL_COLS = [
    Columns.AlarmImmobiliser.value,
    Columns.TrackingDevice.value,
    Columns.WrittenOff.value,
    Columns.Converted.value,
    Columns.Rebuilt.value,
    Columns.CrossBorder.value,
]

NUMERIC_COLS = [
    Columns.mmcode.value,
    Columns.RegistrationYear.value,
    Columns.Cylinders.value,
    Columns.cubiccapacity.value,
    Columns.kilowatts.value,
    Columns.NumberOfDoors.value,
    Columns.CustomValueEstimate.value,
    Columns.CapitalOutstanding.value,
    Columns.SumInsured.value,
    Columns.CalculatedPremiumPerTerm.value,
    Columns.TotalPremium.value,
    Columns.TotalClaims.value,
]

OUTLIER_COLS = [
    Columns.TotalPremium.value,
    Columns.TotalClaims.value,
    Columns.CustomValueEstimate.value,
]

threashold = 1.5


NON_NEGATIVE_COLS = [
    Columns.TotalPremium.value,
    Columns.CalculatedPremiumPerTerm.value,
    Columns.SumInsured.value,
    Columns.TotalClaims.value,
    Columns.CapitalOutstanding.value,
    Columns.CustomValueEstimate.value,
]

CATEGORY_COLS = [
    Columns.Bank.value,
    Columns.AccountType.value,
    Columns.Province.value,
    Columns.MainCrestaZone.value,
    Columns.SubCrestaZone.value,
    Columns.ItemType.value,
    Columns.VehicleType.value,
    Columns.NewVehicle.value,
    Columns.TermFrequency.value,
    Columns.CoverCategory.value,
    Columns.CoverType.value,
    Columns.CoverGroup.value,
    Columns.Section.value,
    Columns.Product.value,
    Columns.StatutoryClass.value,
    Columns.StatutoryRiskType.value,
]
