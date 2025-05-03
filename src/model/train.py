#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
여행 추천 시스템 모델 학습 스크립트
감정 및 여행 동기를 결합한 하이브리드 추천 시스템
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
import pickle
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='여행 추천 시스템 모델 학습')
    parser.add_argument('--data_path', type=str, default='../../data/train/final_merge_outer.csv',
                        help='학습 데이터 경로')
    parser.add_argument('--emotion_data_path', type=str, default='data/emotion_mapping.csv',
                        help='감정-여행동기 매핑 데이터 경로')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='모델 저장 디렉토리')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='임베딩 차원')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 에포크')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='테스트 데이터 비율')
    return parser.parse_args()

# 감정-여행동기 매핑 테이블 생성
def create_emotion_mapping(emotion_data_path=None):
    """
    폴 에크만의 6가지 기본 감정과 여행 동기를 매핑하는 테이블 생성
    
    기본 감정 6가지:
    1. 기쁨(Joy)
    2. 슬픔(Sadness)
    3. 분노(Anger)
    4. 공포(Fear)
    5. 혐오(Disgust)
    6. 놀람(Surprise)
    
    이 함수는 감정과 여행 동기(TMT) 간의 관계를 정의합니다.
    """
    # 데이터 파일이 있으면 로드, 없으면 기본 매핑 생성
    if emotion_data_path and os.path.exists(emotion_data_path):
        logger.info(f"감정 매핑 데이터 로드: {emotion_data_path}")
        return pd.read_csv(emotion_data_path)
    
    # 기본 매핑 정의 (감정 ID와 여행 동기 ID 간의 연관성)
    emotion_mapping = {
        # 감정 ID: [관련 여행 동기 ID와 가중치]
        # 기쁨(1)
        1: [
            {"motive_id": 3, "weight": 0.8},  # 여행 동반자와의 친밀감
            {"motive_id": 7, "weight": 0.7},  # 새로운 경험 추구
            {"motive_id": 9, "weight": 0.9},  # 특별한 목적(칠순여행, 신혼여행 등)
        ],
        # 슬픔(2)
        2: [
            {"motive_id": 2, "weight": 0.9},  # 쉴 수 있는 기회, 정신적 휴식
            {"motive_id": 4, "weight": 0.7},  # 자아 찾기, 자신을 되돌아볼 기회
            {"motive_id": 1, "weight": 0.8},  # 일상에서의 탈출
        ],
        # 분노(3)
        3: [
            {"motive_id": 1, "weight": 0.9},  # 일상적 환경에서의 탈출
            {"motive_id": 6, "weight": 0.7},  # 운동, 건강 증진 및 충전
            {"motive_id": 2, "weight": 0.8},  # 쉴 수 있는 기회
        ],
        # 공포(4)
        4: [
            {"motive_id": 2, "weight": 0.9},  # 쉴 수 있는 기회, 정신적 휴식
            {"motive_id": 3, "weight": 0.8},  # 여행 동반자와의 친밀감
            {"motive_id": 6, "weight": 0.6},  # 운동, 건강 증진
        ],
        # 혐오(5)
        5: [
            {"motive_id": 1, "weight": 0.9},  # 일상에서의 탈출
            {"motive_id": 7, "weight": 0.7},  # 새로운 경험 추구
            {"motive_id": 8, "weight": 0.6},  # 교육적 동기
        ],
        # 놀람(6)
        6: [
            {"motive_id": 7, "weight": 0.9},  # 새로운 경험 추구
            {"motive_id": 8, "weight": 0.7},  # 교육적 동기
            {"motive_id": 5, "weight": 0.5},  # SNS 사진 등록 등 과시
        ]
    }
    
    # 데이터프레임으로 변환
    mapping_data = []
    for emotion_id, motives in emotion_mapping.items():
        for motive in motives:
            mapping_data.append({
                "emotion_id": emotion_id,
                "motive_id": motive["motive_id"],
                "weight": motive["weight"]
            })
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # 감정 및 동기 이름 추가
    emotion_names = {
        1: "기쁨(Joy)",
        2: "슬픔(Sadness)",
        3: "분노(Anger)",
        4: "공포(Fear)",
        5: "혐오(Disgust)",
        6: "놀람(Surprise)"
    }
    
    motive_names = {
        1: "일상적인 환경 및 역할에서의 탈출, 지루함 탈피",
        2: "쉴 수 있는 기회, 육체 피로 해결 및 정신적인 휴식",
        3: "여행 동반자와의 친밀감 및 유대감 증진",
        4: "진정한 자아 찾기 또는 자신을 되돌아볼 기회 찾기",
        5: "SNS 사진 등록 등 과시",
        6: "운동, 건강 증진 및 충전",
        7: "새로운 경험 추구",
        8: "역사 탐방, 문화적 경험 등 교육적 동기",
        9: "특별한 목적(칠순여행, 신혼여행, 수학여행, 인센티브여행)",
        10: "기타"
    }
    
    mapping_df["emotion_name"] = mapping_df["emotion_id"].map(emotion_names)
    mapping_df["motive_name"] = mapping_df["motive_id"].map(motive_names)
    
    return mapping_df

# 데이터 전처리 함수
def preprocess_data(df, test_size=0.2):
    """
    데이터 전처리를 수행합니다.
    1. 다중 선택 컬럼 원-핫 인코딩
    2. 레이블 인코딩
    3. 스케일링
    4. 학습/테스트 분할
    """
    # 다중 선택 컬럼 처리 (Multi-hot 변환)
    def multi_label_binarize(df, column, separator=';', prefix=''):
        all_labels = set()
        for entry in df[column].dropna():
            all_labels.update(entry.split(separator))
        all_labels = sorted(all_labels)

        for label in all_labels:
            df[f"{prefix}{label}"] = df[column].apply(
                lambda x: 1 if pd.notna(x) and label in x.split(separator) else 0
            )
        return df, all_labels

    # 여행 동기, 스타일, 목적 다중 선택 처리
    motives_labels = []
    styles_labels = []
    purposes_labels = []
    
    for col, prefix, labels_list in zip(
        ['TRAVEL_MOTIVES', 'TRAVEL_STYLES', 'TRAVEL_PURPOSE'],
        ['motive_', 'style_', 'purpose_'],
        [motives_labels, styles_labels, purposes_labels]
    ):
        if col in df.columns:
            df, labels = multi_label_binarize(df, col, separator=';', prefix=prefix)
            labels_list.extend(labels)
    
    # 사용자 및 아이템 인코딩
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    if 'TRAVELER_ID' in df.columns:
        df['user'] = user_encoder.fit_transform(df['TRAVELER_ID'])
    
    if 'VISIT_AREA_TYPE_CD' in df.columns:
        df['item'] = item_encoder.fit_transform(df['VISIT_AREA_TYPE_CD'])
    
    # 범주형 변수 처리
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].map({'남': 0, '여': 1})
    
    # 필요한 컬럼만 선택
    feature_cols = ['user', 'GENDER', 'AGE_GRP', 'TRAVEL_COMPANIONS_NUM', 'VISIT_CHC_REASON_CD']
    
    # 동기, 스타일, 목적 관련 컬럼 추가
    motive_cols = [col for col in df.columns if col.startswith('motive_')]
    style_cols = [col for col in df.columns if col.startswith('style_')]
    purpose_cols = [col for col in df.columns if col.startswith('purpose_')]
    
    feature_cols.extend(motive_cols)
    feature_cols.extend(style_cols)
    feature_cols.extend(purpose_cols)
    feature_cols.append('item')
    
    # 대상 컬럼이 존재하는지 확인하고 필터링
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 필요한 데이터 필터링
    if 'DGSTFN' in df.columns:  # 만족도 컬럼이 있는 경우
        df = df[feature_cols + ['DGSTFN']].dropna(subset=['DGSTFN'])
        X = df[feature_cols].copy()
        y = df['DGSTFN'].values
    else:  # 만족도 컬럼이 없는 경우 (예측용 데이터인 경우)
        df = df[feature_cols].copy()
        X = df.copy()
        y = None
    
    # 누락된 값 처리
    X = X.fillna(0)
    
    # 숫자형 변수 스케일링
    numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns
                    if col not in ['user', 'item']]
    if numeric_cols:
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # 범주형 변수 처리
    categorical_cols = [col for col in X.columns if col not in numeric_cols and col != 'user' and col != 'item']
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(X[categorical_cols])
        
        # 원-핫 인코딩 결과를 데이터프레임으로 변환
        encoded_df = pd.DataFrame(encoded_features, 
                                  columns=[f"{col}_{i}" for col, i in 
                                         zip(np.repeat(categorical_cols, encoder.n_features_in_), 
                                             range(encoded_features.shape[1]))])
        
        # 원래 데이터프레임에서 범주형 변수 제거 후 인코딩 결과 합치기
        X = X.drop(categorical_cols, axis=1)
        X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)
    
    # 학습/테스트 데이터 분할
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test, user_encoder, item_encoder
    else:
        return X, None, None, None, user_encoder, item_encoder

# 감정-여행 동기 결합 모델 정의
class EmotionAwareRecommender(tf.keras.Model):
    """
    감정 정보를 활용한 하이브리드 추천 모델
    
    - 사용자 임베딩
    - 아이템 임베딩
    - 감정 임베딩
    - 다중 타워 구조(MLP)
    - 어텐션 메커니즘
    """
    def __init__(self, num_users, num_items, num_emotions=6, embedding_dim=32):
        super().__init__()
        # 기본 임베딩 레이어
        self.user_embedding = layers.Embedding(num_users, embedding_dim, name="user_embedding")
        self.item_embedding = layers.Embedding(num_items, embedding_dim, name="item_embedding")
        
        self.emotion_embedding = layers.Embedding(num_emotions+1, embedding_dim, name="emotion_embedding")  # +1 for padding
        
        # 사용자 타워
        self.user_dense1 = layers.Dense(64, activation='relu', name="user_dense1")
        self.user_dense2 = layers.Dense(32, activation='relu', name="user_dense2")
        
        # 아이템 타워
        self.item_dense1 = layers.Dense(64, activation='relu', name="item_dense1")
        self.item_dense2 = layers.Dense(32, activation='relu', name="item_dense2")
        
        # 감정 타워
        self.emotion_dense1 = layers.Dense(64, activation='relu', name="emotion_dense1")
        self.emotion_dense2 = layers.Dense(32, activation='relu', name="emotion_dense2")
        
        # 특성 처리 레이어
        self.feature_dense = layers.Dense(128, activation='relu', name="feature_dense")
        
        # 어텐션 메커니즘을 위한 레이어
        self.attention_dense = layers.Dense(32, activation='tanh', name="attention_dense")
        self.attention_weight = layers.Dense(1, activation='softmax', name="attention_weight")
        
        # 컨텍스트 벡터와 특성 결합
        self.concat_dense = layers.Dense(64, activation='relu', name="concat_dense")
        
        # 최종 출력 레이어
        self.output_dense = layers.Dense(1, name="output")
        
    def call(self, inputs):
        user_input, item_input, emotion_input, feature_input = inputs
        
        # 임베딩
        user_vec = self.user_embedding(user_input)
        item_vec = self.item_embedding(item_input)
        emotion_vec = self.emotion_embedding(emotion_input)
        
        # 1차원 벡터로 변환
        user_vec = tf.squeeze(user_vec, axis=1)
        item_vec = tf.squeeze(item_vec, axis=1)
        emotion_vec = tf.squeeze(emotion_vec, axis=1)
        
        # 각 타워 통과
        user_tower = self.user_dense2(self.user_dense1(user_vec))
        item_tower = self.item_dense2(self.item_dense1(item_vec))
        emotion_tower = self.emotion_dense2(self.emotion_dense1(emotion_vec))
        
        
        
        # 어텐션 점수 계산
        user_attention = self.attention_weight(self.attention_dense(user_tower))
        item_attention = self.attention_weight(self.attention_dense(item_tower))
        emotion_attention = self.attention_weight(self.attention_dense(emotion_tower))
        

        
        # 정규화
        attention_concat = tf.concat([user_attention, item_attention, emotion_attention], axis=1)
        attention_weights = tf.nn.softmax(attention_concat, axis=1)
        
        # 가중치 적용
        weighted_user = user_tower * tf.expand_dims(attention_weights[:, int(0)], axis=1)
        weighted_item = item_tower * tf.expand_dims(attention_weights[:, int(1)], axis=1)
        weighted_emotion = emotion_tower * tf.expand_dims(attention_weights[:, int(2)], axis=1)
        
        # 컨텍스트 벡터 생성
        context_vector = weighted_user + weighted_item + weighted_emotion
        
        # 특성 처리
        feature_vector = self.feature_dense(feature_input)
        
        # 컨텍스트 벡터와 특성 결합
        concat_vector = tf.concat([context_vector, feature_vector], axis=1)
        hidden = self.concat_dense(concat_vector)
        
        # 최종 출력
        output = self.output_dense(hidden)
        
        return output

# 모델 학습 함수
def train_model(X_train, X_test, y_train, y_test, args):
    """
    모델을 학습시키고 평가합니다.
    """
    # 하이퍼파라미터 설정
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    # 모델 입력 준비
    num_users = X_train['user'].max() + 1
    num_items = X_train['item'].max() + 1
    num_emotions = 6  # 폴 에크만의 6가지 기본 감정
    
    # 입력 데이터 분리
    user_input_train = X_train['user'].values.reshape(-1, 1)
    item_input_train = X_train['item'].values.reshape(-1, 1)
    
    # 임시 감정 데이터 생성 (실제로는 감정 정보가 필요함)
    # 학습 시에는 기본 감정으로 초기화 (1: 기쁨)
    emotion_input_train = np.ones_like(user_input_train)
    
    # 나머지 특성
    feature_cols = [col for col in X_train.columns if col not in ['user', 'item']]
    feature_input_train = X_train[feature_cols].values
    
    # 테스트 데이터도 동일하게 준비
    user_input_test = X_test['user'].values.reshape(-1, 1)
    item_input_test = X_test['item'].values.reshape(-1, 1)
    emotion_input_test = np.ones_like(user_input_test)
    feature_input_test = X_test[feature_cols].values
    
    # 모델 인스턴스 생성
    model = EmotionAwareRecommender(num_users, num_items, num_emotions, embedding_dim)
    
    # 모델 컴파일
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # 모델 빌드를 위한 샘플 데이터 전달
    sample_user = tf.constant(user_input_train[:1], dtype=tf.int32)
    sample_item = tf.constant(item_input_train[:1], dtype=tf.int32)
    sample_emotion = tf.constant(emotion_input_train[:1], dtype=tf.int32)
    sample_feature = tf.constant(feature_input_train[:1], dtype=tf.float32)
    
    model([sample_user, sample_item, sample_emotion, sample_feature])
    
    # 모델 학습
    history = model.fit(
        [user_input_train, item_input_train, emotion_input_train, feature_input_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            [user_input_test, item_input_test, emotion_input_test, feature_input_test],
            y_test
        ),
        verbose=1
    )
    
    # 모델 평가
    predictions = model.predict(
        [user_input_test, item_input_test, emotion_input_test, feature_input_test]
    )
    pred_flat = predictions.flatten()
    
    # 평가 지표 계산
    mse = mean_squared_error(y_test, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred_flat)
    r2 = r2_score(y_test, pred_flat)
    
    # 허용 오차 기반 정확도
    tolerance = 0.5
    within_tolerance = np.abs(pred_flat - y_test) <= tolerance
    tolerance_accuracy = np.mean(within_tolerance)
    
    # 평가 결과 출력
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"허용 오차({tolerance}) 내 정확도: {tolerance_accuracy:.4f}")
    
    # 학습 이력 및 평가 결과
    results = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'tolerance_accuracy': float(tolerance_accuracy),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }
    }
    
    return model, results

# 모델 저장 함수
def save_model(model, user_encoder, item_encoder, results, args):
    """
    학습된 모델과 관련 인코더, 결과를 저장합니다.
    """
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 현재 시간을 이용한 모델 버전 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"model_v{timestamp}"
    model_dir = os.path.join(args.output_dir, model_version)
    os.makedirs(model_dir, exist_ok=True)
    
    # TensorFlow 모델 저장
    model_path = os.path.join(model_dir, "emotion_recommender.keras")
    model.save(model_path)
    logger.info(f"모델 저장 완료: {model_path}")
    
    # 인코더 저장
    encoder_path = os.path.join(model_dir, "encoders.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump({
            'user_encoder': user_encoder,
            'item_encoder': item_encoder
        }, f)
    logger.info(f"인코더 저장 완료: {encoder_path}")
    
    # 평가 결과 저장
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"평가 결과 저장 완료: {results_path}")
    
    # 최신 모델 정보 갱신
    latest_info = {
        'latest_model': model_version,
        'timestamp': timestamp,
        'metrics': {
            'rmse': results['rmse'],
            'mae': results['mae']
        }
    }
    
    latest_path = os.path.join(args.output_dir, "latest.json")
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(latest_info, f, ensure_ascii=False, indent=2)
    logger.info(f"최신 모델 정보 갱신 완료: {latest_path}")
    
    return model_version

def main():
    # 인자 파싱
    args = parse_args()
    logger.info("감정 기반 여행 추천 시스템 모델 학습 시작")
    
    # 감정-여행동기 매핑 테이블 생성
    emotion_mapping = create_emotion_mapping(args.emotion_data_path)
    logger.info(f"감정-여행동기 매핑 테이블 생성 완료 (크기: {len(emotion_mapping)})")
    
    # 매핑 테이블 저장
    os.makedirs(args.output_dir, exist_ok=True)
    mapping_path = os.path.join(args.output_dir, "emotion_mapping.csv")
    emotion_mapping.to_csv(mapping_path, index=False, encoding='utf-8')
    logger.info(f"감정-여행동기 매핑 테이블 저장 완료: {mapping_path}")
    
    # 데이터 로드
    try:
        df = pd.read_csv(args.data_path)
        logger.info(f"데이터 로드 완료 (크기: {df.shape})")
    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}")
        return
    
    # 데이터 전처리
    try:
        X_train, X_test, y_train, y_test, user_encoder, item_encoder = preprocess_data(df, args.test_size)
        logger.info(f"데이터 전처리 완료 (학습 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape})")
    except Exception as e:
        logger.error(f"데이터 전처리 실패: {str(e)}")
        return
    
    # 모델 학습
    try:
        model, results = train_model(X_train, X_test, y_train, y_test, args)
        logger.info("모델 학습 완료")
    except Exception as e:
        logger.error(f"모델 학습 실패: {str(e)}")
        return
    
    # 모델 저장
    try:
        model_version = save_model(model, user_encoder, item_encoder, results, args)
        logger.info(f"모델 저장 완료 (버전: {model_version})")
    except Exception as e:
        logger.error(f"모델 저장 실패: {str(e)}")
        return
    
    logger.info("감정 기반 여행 추천 시스템 모델 학습 완료")

if __name__ == "__main__":
    main()