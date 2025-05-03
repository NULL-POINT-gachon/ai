#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
여행 추천 시스템 추론 스크립트
Node.js 백엔드에서 차일드 프로세스로 호출되어 사용됨

사용 예시:
  도시 추천: 
    python ai_recommendation.py --mode city --trip_duration 3 --companions_count 2 --emotion_ids 2 --top_n 3 --alpha 0.7
  
  상세 여행지 추천:
    python ai_recommendation.py --mode detail --city "서울" --activity_type "실내" --activity_ids 1,3,5 --emotion_ids 2 --preferred_transport "대중교통" --companions_count 2 --activity_level 10 --top_n 3 --alpha 0.7
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_recommendation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='여행 추천 시스템 추론')
    parser.add_argument('--mode', type=str, required=True, choices=['city', 'detail'],
                        help='추천 모드 (city: 도시 추천, detail: 상세 여행지 추천)')
    
    # 공통 인자
    parser.add_argument('--emotion_ids', type=str, required=True,
                        help='감정 ID (쉼표로 구분)')
    parser.add_argument('--companions_count', type=int, required=True,
                        help='여행 동반자 수')
    parser.add_argument('--top_n', type=int, default=3,
                        help='추천 결과 개수')
    parser.add_argument('--recommendation_type', type=str, default='both',
                        help='추천 유형 (content, collaborative, both)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='하이브리드 추천에서 협업 필터링 가중치')
    
    # 도시 추천 모드 인자
    parser.add_argument('--trip_duration', type=int,
                        help='여행 기간 (일)')
    
    # 상세 여행지 추천 모드 인자
    parser.add_argument('--city', type=str,
                        help='도시명')
    parser.add_argument('--activity_type', type=str,
                        help='활동 유형 (실내, 실외)')
    parser.add_argument('--activity_ids', type=str,
                        help='활동 ID (쉼표로 구분)')
    parser.add_argument('--preferred_transport', type=str,
                        help='선호 교통수단')
    parser.add_argument('--activity_level', type=int,
                        help='활동 수준 (1-10)')
    
    # 모델 관련 인자
    parser.add_argument('--models_dir', type=str, default='../model/models',
                        help='모델 디렉토리')
    parser.add_argument('--city_data', type=str, default='../../data/reommencder/city/city_data.csv',
                        help='도시 데이터 경로')
    parser.add_argument('--place_data', type=str, default='../../data/reommencder/place/place_data.csv',
                        help='장소 데이터 경로')
    
    return parser.parse_args()

# 감정 매핑 로드
def load_emotion_mapping(models_dir):
    """감정-여행동기 매핑 테이블을 로드합니다."""
    mapping_path = os.path.join(models_dir, "emotion_mapping.csv")
    try:
        mapping_df = pd.read_csv(mapping_path)
        return mapping_df
    except Exception as e:
        logger.error(f"감정 매핑 로드 실패: {str(e)}")
        # 기본 매핑 생성
        default_mapping = pd.DataFrame({
            'emotion_id': [1, 2, 3, 4, 5, 6],
            'motive_id': [7, 2, 1, 2, 1, 7],
            'weight': [0.8, 0.9, 0.9, 0.9, 0.9, 0.9],
            'emotion_name': ['기쁨', '슬픔', '분노', '공포', '혐오', '놀람'],
            'motive_name': ['새로운 경험 추구', '정신적 휴식', '일상 탈출', '정신적 휴식', '일상 탈출', '새로운 경험 추구']
        })
        return default_mapping

# 최신 모델 정보 로드
def load_latest_model_info(models_dir):
    """최신 모델 정보를 로드합니다."""
    latest_path = os.path.join(models_dir, "latest.json")
    try:
        with open(latest_path, 'r', encoding='utf-8') as f:
            latest_info = json.load(f)
        return latest_info
    except Exception as e:
        logger.error(f"최신 모델 정보 로드 실패: {str(e)}")
        return {'latest_model': ''}

# 모델 및 인코더 로드
def load_model_and_encoders(models_dir, model_version=None):
    """모델과 인코더를 로드합니다."""
    if model_version is None:
        # 최신 모델 버전 사용
        latest_info = load_latest_model_info(models_dir)
        model_version = latest_info.get('latest_model')
        if not model_version:
            logger.error("유효한 모델 버전을 찾을 수 없습니다.")
            return None, None
    
    model_dir = os.path.join(models_dir, model_version)
    
    # 모델 로드
    model_path = os.path.join(model_dir, "emotion_recommender")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"모델 로드 완료: {model_path}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return None, None
    
    # 인코더 로드
    encoder_path = os.path.join(model_dir, "encoders.pkl")
    try:
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        logger.info(f"인코더 로드 완료: {encoder_path}")
    except Exception as e:
        logger.error(f"인코더 로드 실패: {str(e)}")
        return model, None
    
    return model, encoders

# 도시 데이터 로드
def load_city_data(city_data_path):
    """도시 데이터를 로드합니다."""
    try:
        city_df = pd.read_csv(city_data_path)
        return city_df
    except Exception as e:
        logger.error(f"도시 데이터 로드 실패: {str(e)}")
        # 기본 도시 데이터 생성
        default_cities = pd.DataFrame({
            'city_id': ['CTY_SEOUL', 'CTY_BUSAN', 'CTY_JEJU', 'CTY_GANGNEUNG', 'CTY_GYEONGJU'],
            'item_name': ['서울', '부산', '제주', '강릉', '경주'],
            'region': ['수도권', '경상권', '제주권', '강원권', '경상권'],
            'features': [
                '현대적;쇼핑;역사적;문화예술',
                '해변;항구;산;음식',
                '자연;해변;화산;휴양',
                '바다;계절성;전통시장;커피',
                '역사적;불교;전통;문화유산'
            ],
            'avg_stay_duration': [3, 3, 4, 2, 2],
            'season_preference': ['All', 'Summer', 'All', 'Winter', 'Spring'],
            'motive_match': [
                '1;3;5;7;8',
                '1;2;6;7',
                '1;2;6;7;9',
                '1;2;7',
                '1;4;8'
            ]
        })
        return default_cities

# 장소 데이터 로드
def load_place_data(place_data_path):
    """장소 데이터를 로드합니다."""
    try:
        place_df = pd.read_csv(place_data_path)
        return place_df
    except Exception as e:
        logger.error(f"장소 데이터 로드 실패: {str(e)}")
        # 기본 장소 데이터 생성
        default_places = pd.DataFrame({
            'place_id': ['PLC001', 'PLC002', 'PLC003', 'PLC004', 'PLC005'],
            '여행지명': ['경복궁', '남산서울타워', '해운대 해수욕장', '제주 성산일출봉', '불국사'],
            '여행지설명': [
                '조선 시대의 대표적인 궁궐로 웅장한 건축미를 자랑합니다.',
                '서울의 상징적인 랜드마크로 도시 전경을 한눈에 볼 수 있습니다.',
                '부산의 대표적인 해변으로 아름다운 해안선을 자랑합니다.',
                '제주도의 유네스코 세계자연유산으로 장엄한 일출로 유명합니다.',
                '경주의 대표적인 불교 사찰로 통일신라시대의 불교 문화를 엿볼 수 있습니다.'
            ],
            'city': ['서울', '서울', '부산', '제주', '경주'],
            '분류': ['역사', '랜드마크', '자연', '자연', '역사'],
            'activity_type': ['실외', '실외', '실외', '실외', '실외'],
            'activity_ids': ['1;8', '5;7', '2;6', '7;8', '4;8'],
            'activity_level': [3, 4, 5, 6, 3],
            'season_preference': ['All', 'All', 'Summer', 'All', 'Spring'],
            'emotion_match': ['1;6', '1;6', '1;2', '1;6', '2;4']
        })
        return default_places

# 감정 ID를 여행 동기 ID로 변환
def emotion_to_motive(emotion_ids, emotion_mapping):
    """
    감정 ID를 여행 동기 ID로 변환합니다.
    """
    if isinstance(emotion_ids, str):
        emotion_ids = [int(id) for id in emotion_ids.split(',')]
    
    motive_weights = {}
    
    for emotion_id in emotion_ids:
        # 해당 감정에 맞는 동기들 추출
        motives = emotion_mapping[emotion_mapping['emotion_id'] == emotion_id]
        
        for _, motive in motives.iterrows():
            motive_id = motive['motive_id']
            weight = motive['weight']
            
            # 가중치 누적 (같은 동기가 여러 감정에서 나타날 수 있음)
            if motive_id in motive_weights:
                motive_weights[motive_id] += weight
            else:
                motive_weights[motive_id] = weight
    
    # 가중치가 높은 순으로 정렬
    sorted_motives = sorted(motive_weights.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 동기 ID 반환
    top_motives = [motive[0] for motive in sorted_motives]
    
    return top_motives

# 도시 추천 기능
def recommend_cities(args, emotion_mapping, city_df):
    """
    감정 ID, 여행 기간, 동반자 수 등을 고려하여 도시를 추천합니다.
    """
    # 감정 ID를 여행 동기로 변환
    motive_ids = emotion_to_motive(args.emotion_ids, emotion_mapping)
    
    # 도시별 점수 계산
    city_scores = []
    
    for _, city in city_df.iterrows():
        score = 0
        
        # 1. 동기 매칭 점수
        city_motives = [int(m) for m in str(city['motive_match']).split(';') if m.isdigit()]
        motive_match_score = sum(5.0 / (idx + 1) for idx, m_id in enumerate(motive_ids) if m_id in city_motives)
        score += motive_match_score * 0.4  # 40% 가중치
        
        # 2. 여행 기간 적합성
        if args.trip_duration:
            duration_diff = abs(city['avg_stay_duration'] - args.trip_duration)
            duration_score = max(0, 5 - duration_diff) / 5  # 0~1 정규화
            score += duration_score * 0.3  # 30% 가중치
        
        # 3. 동반자 수 적합성 (예: 가족 여행은 4명 이상)
        companion_score = 0
        if args.companions_count == 1:  # 혼자 여행
            if '혼자' in str(city['features']).lower():
                companion_score = 1
        elif args.companions_count == 2:  # 커플/친구
            if '커플' in str(city['features']).lower() or '친구' in str(city['features']).lower():
                companion_score = 1
        elif args.companions_count >= 3:  # 가족/단체
            if '가족' in str(city['features']).lower() or '단체' in str(city['features']).lower():
                companion_score = 1
        
        score += companion_score * 0.3  # 30% 가중치
        
        # 현재 시즌 적합성 보너스 (선택적)
        current_month = datetime.now().month
        season = ''
        if 3 <= current_month <= 5:
            season = 'Spring'
        elif 6 <= current_month <= 8:
            season = 'Summer'
        elif 9 <= current_month <= 11:
            season = 'Fall'
        else:
            season = 'Winter'
        
        if season in str(city['season_preference']) or 'All' in str(city['season_preference']):
            score += 0.2  # 시즌 매칭 보너스
        
        city_scores.append({
            'city_id': city['city_id'],
            'item_name': city['item_name'],
            'score': score,
            'features': str(city['features']).split(';') if 'features' in city else []
        })
    
    # 점수 기준 정렬
    sorted_cities = sorted(city_scores, key=lambda x: x['score'], reverse=True)
    
    # 상위 N개 도시 선택
    top_cities = sorted_cities[:args.top_n]
    
    # 관련 활동 추가 (각 도시의 특성에서 추출)
    for city in top_cities:
        activities = []
        features = city.pop('features', [])
        
        # 특성을 기반으로 관련 활동 생성
        for feature in features:
            if feature == '해변':
                activities.append('해변 산책')
                activities.append('수영')
            elif feature == '산':
                activities.append('등산')
                activities.append('트레킹')
            elif feature == '쇼핑':
                activities.append('쇼핑')
                activities.append('로컬 마켓 탐방')
            elif feature == '역사적':
                activities.append('역사 유적 탐방')
                activities.append('박물관 방문')
            elif feature == '문화예술':
                activities.append('미술관 관람')
                activities.append('공연 관람')
            elif feature == '자연':
                activities.append('자연 경관 감상')
                activities.append('피크닉')
            elif feature == '음식':
                activities.append('맛집 탐방')
                activities.append('요리 체험')
        
        # 중복 제거 및 상위 3개 활동 선택
        unique_activities = list(set(activities))
        city['related_activities'] = unique_activities[:3]
    
    # 최종 결과 생성
    result = {
        'user_id': f"u{hash(args.emotion_ids + str(args.companions_count)) % 1000}",  # 임시 사용자 ID
        'recommendations': top_cities
    }
    
    return result

# 상세 여행지 추천 기능
def recommend_places(args, emotion_mapping, place_df):
    """
    도시, 활동 유형, 활동 ID, 감정 ID 등을 고려하여 상세 여행지를 추천합니다.
    """
    # 해당 도시의 여행지만 필터링
    city_places = place_df[place_df['city'] == args.city].copy()
    
    if len(city_places) == 0:
        logger.warning(f"'{args.city}'에 해당하는 여행지가 없습니다.")
        return {'places': []}
    
    # 감정 ID를 여행 동기로 변환
    motive_ids = emotion_to_motive(args.emotion_ids, emotion_mapping)
    motive_ids_str = [str(id) for id in motive_ids]
    
    # 활동 ID 분리
    activity_ids = [int(id) for id in args.activity_ids.split(',')] if args.activity_ids else []
    
    # 여행지별 점수 계산
    place_scores = []
    
    for _, place in city_places.iterrows():
        score = 0
        
        # 1. 활동 유형 매칭 (실내/실외)
        if args.activity_type and place['activity_type'] == args.activity_type:
            score += 1.0
        
        # 2. 활동 ID 매칭
        place_activities = [act for act in str(place['activity_ids']).split(';') if act.isdigit()]
        activity_match_count = sum(1 for act in activity_ids if str(act) in place_activities)
        if activity_ids:
            activity_match_score = activity_match_count / len(activity_ids)
            score += activity_match_score * 2.0
        
        # 3. 감정 매칭
        place_emotions = [emo for emo in str(place['emotion_match']).split(';') if emo.isdigit()]
        emotion_match = args.emotion_ids in place_emotions
        if emotion_match:
            score += 1.5
        
        # 4. 활동 수준 적합성
        if args.activity_level:
            level_diff = abs(place['activity_level'] - args.activity_level)
            level_score = max(0, 10 - level_diff) / 10  # 0~1 정규화
            score += level_score * 1.0
        
        # 5. 선호 교통수단 적합성 (가정)
        if args.preferred_transport:
            # 이 부분은 실제 데이터에 따라 조정 필요
            transport_score = 0
            if args.preferred_transport == '대중교통' and '대중교통' in str(place.get('access_method', '')):
                transport_score = 1
            elif args.preferred_transport == '자가용' and '주차장' in str(place.get('facilities', '')):
                transport_score = 1
            score += transport_score * 0.5
        
        # 6. 동반자 수에 따른 적합성 (가정)
        if '가족' in str(place.get('suitable_for', '')) and args.companions_count >= 3:
            score += 0.5
        elif '커플' in str(place.get('suitable_for', '')) and args.companions_count == 2:
            score += 0.5
        elif '혼자' in str(place.get('suitable_for', '')) and args.companions_count == 1:
            score += 0.5
        
        place_scores.append({
            '여행지명': place['여행지명'],
            '여행지설명': place['여행지설명'],
            '분류': place['분류'],
            'score': score
        })
    
    # 점수 기준 정렬
    sorted_places = sorted(place_scores, key=lambda x: x['score'], reverse=True)
    
    # 상위 N개 여행지 선택
    top_places = sorted_places[:args.top_n]
    
    # 점수 필드 제거 (응답에는 불필요)
    for place in top_places:
        place.pop('score', None)
    
    # 최종 결과 생성
    result = {
        'places': top_places
    }
    
    return result

def main():
    # 인자 파싱
    args = parse_args()
    
    # 감정 매핑 로드
    emotion_mapping = load_emotion_mapping(args.models_dir)
    
    if args.mode == 'city':
        # 도시 데이터 로드
        city_df = load_city_data(args.city_data)
        
        # 도시 추천
        if not args.trip_duration:
            logger.error("도시 추천 모드에서는 여행 기간(--trip_duration)이 필요합니다.")
            sys.exit(1)
        
        result = recommend_cities(args, emotion_mapping, city_df)
    
    elif args.mode == 'detail':
        # 상세 여행지 데이터 로드
        place_df = load_place_data(args.place_data)
        
        # 필수 인자 확인
        if not args.city:
            logger.error("상세 여행지 추천 모드에서는 도시명(--city)이 필요합니다.")
            sys.exit(1)
        if not args.activity_type:
            logger.error("상세 여행지 추천 모드에서는 활동 유형(--activity_type)이 필요합니다.")
            sys.exit(1)
        if not args.activity_ids:
            logger.error("상세 여행지 추천 모드에서는 활동 ID(--activity_ids)가 필요합니다.")
            sys.exit(1)
        
        # 상세 여행지 추천
        result = recommend_places(args, emotion_mapping, place_df)
    
    else:
        logger.error(f"알 수 없는 모드: {args.mode}")
        sys.exit(1)
    
    # 결과 출력 (JSON 형식)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()