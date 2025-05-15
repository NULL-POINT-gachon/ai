#!/bin/bash

# 필요한 디렉토리 생성
mkdir -p data models

# 모델 학습 실행
echo "========== 모델 학습 시작 =========="
python train.py --data_path data/final_merge_outer.csv --emotion_data_path data/emotion_mapping.csv --output_dir models --epochs 5

python train.py --data_path ../../data/train/final_df.csv --emotion_data_path ../../data/emotion_mapping.csv --output_dir ./models --epochs 5

# 도시 추천 테스트
echo -e "\n\n========== 도시 추천 테스트 =========="
python ./src/recommender/ai_recommendation.py --mode city --trip_duration 3 --companions_count 2 --emotion_ids 2 --top_n 3 --alpha 0.7
        --models_dir ./src/model/models --city_data ./data/recommender/city/city_data_clean.csv --place_data ./data/recommender/places/place_data_clean.csv

# 상세 여행지 추천 테스트
echo -e "\n\n========== 상세 여행지 추천 테스트 =========="
python ./src/recommender/ai_recommendation.py --mode detail --city "서울특별시" --activity_type "실내" --activity_ids 1,3,5 --emotion_ids 3 --preferred_transport "대중교통" --companions_count 5 --activity_level 1 --top_n 3 --alpha 0.7
        --models_dir ./src/model/models --city_data ./data/recommender/city/city_data_clean.csv --place_data ./data/recommender/places/place_data_clean.csv
echo -e "\n\n모든 테스트가 완료되었습니다."
