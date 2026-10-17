from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "cattle_data_train.csv"
TEST_CSV = ROOT / "cattle_data_test.csv"
SAMPLE_SUB = ROOT / "sample_submission.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "Milk_Yield_L"
ID_COL = "Cattle_ID"
FARM_COL = "Farm_ID"
SEED = 42
N_FOLDS = 5
METRIC = "RMSE"

# Duplicates / derivable columns (see report: Rumination corr with Previous_Week_Avg_Yield = 1.0,
# Feed_Quantity_lb is kg * 2.2046, Lactation_Stage is a function of Days_in_Milk).
DROP_COLS = ["Feed_Quantity_lb", "Rumination_Time_hrs", "Lactation_Stage"]
CAT_COLS = ["Breed", "Climate_Zone", "Management_System", "Feed_Type"]
WINSOR_COLS = ["Age_Months", "Weight_kg", "Water_Intake_L", "Walking_Distance_km",
               "Ambient_Temperature_C", "Previous_Week_Avg_Yield", "Feed_Quantity_kg"]
