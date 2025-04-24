from preprocessing import load_and_preprocess_data
from base_model import build_model
from sklearn.model_selection import train_test_split

# 데이터 불러오기 및 전처리
df, encoders = load_and_preprocess_data(
    "C:/Users/adminastor/proj_main_6/data/raw/final_merge_outer.csv"
)

feature_cols = [
    "user_id",
    "item_id",
    "GENDER",
    "AGE_GRP",
    "TRAVEL_COMPANIONS_NUM",
    "TRAVEL_MOTIVES",
    "TRAVEL_STYLES",
    "VISIT_CHC_REASON_CD",
    "TRAVEL_PURPOSE",
]
X = df[feature_cols]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

vocab_sizes = {
    "user_id": df["user_id"].nunique(),
    "item_id": df["item_id"].nunique(),
    "TRAVEL_MOTIVES": df["TRAVEL_MOTIVES"].nunique(),
    "TRAVEL_STYLES": df["TRAVEL_STYLES"].nunique(),
    "TRAVEL_PURPOSE": df["TRAVEL_PURPOSE"].nunique(),
}

model = build_model(vocab_sizes)

train_inputs = {name: X_train[name].values for name in X.columns}
val_inputs = {name: X_val[name].values for name in X.columns}

model.fit(
    train_inputs, y_train, epochs=10, batch_size=64, validation_data=(val_inputs, y_val)
)
