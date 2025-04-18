import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # 제거할 칼럼
    df = df.drop(columns=['VISIT_AREA_NM', 'TRAVEL_ID'])

    # 성별 인코딩
    df['GENDER'] = df['GENDER'].map({'남': 0, '여': 1})

    # 결측치 제거
    df = df.dropna()

    # 인코더 정의
    encoders = {
        'user_id': LabelEncoder(),
        'item_id': LabelEncoder(),
        'TRAVEL_MOTIVES': LabelEncoder(),
        'TRAVEL_STYLES': LabelEncoder(),
        'TRAVEL_PURPOSE': LabelEncoder()
    }

    df['user_id'] = encoders['user_id'].fit_transform(df['TRAVELER_ID'])
    df['item_id'] = encoders['item_id'].fit_transform(df['VISIT_AREA_TYPE_CD'])
    df['TRAVEL_MOTIVES'] = encoders['TRAVEL_MOTIVES'].fit_transform(df['TRAVEL_MOTIVES'])
    df['TRAVEL_STYLES'] = encoders['TRAVEL_STYLES'].fit_transform(df['TRAVEL_STYLES'])
    df['TRAVEL_PURPOSE'] = encoders['TRAVEL_PURPOSE'].fit_transform(df['TRAVEL_PURPOSE'])

    # 라벨 생성
    df['label'] = (df['DGSTFN'] >= 4).astype(int)

    return df, encoders
