#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
여행 추천 시스템 추론 스크립트
Node.js 백엔드에서 차일드 프로세스로 호출됨

사용 예시
---------
도시 추천:
  python ai_recommendation.py \
    --mode city \
    --trip_duration 3 \
    --companions_count 2 \
    --moods "설렘,모험" \
    --top_n 3

상세 여행지 추천:
  python ai_recommendation.py \
    --mode detail \
    --city "서울" \
    --activity_type "실내" \
    --activity_ids 1,3,5 \
    --moods "힐링" \
    --companions_count 2 \
    --activity_level 10 \
    --top_n 3
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
import pickle
import io

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity  # 향후 CF 사용 대비

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# ────────────────────── 로깅 ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_recommendation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# ────────────────────── 무드 → 감정 매핑 ──────────────────────
MOOD_TO_EMOTIONS = {
    "설렘":   [(1, 0.6), (6, 0.4)],   # 기쁨 + 놀람
    "힐링":   [(2, 1.0)],             # 슬픔(휴식욕구)
    "감성":   [(2, 0.7), (4, 0.3)],
    "여유":   [(2, 1.0)],
    "활력":   [(1, 0.6), (3, 0.4)],
    "모험":   [(6, 0.5), (4, 0.5)],
    "로맨틱": [(1, 1.0)],
    "재충전": [(2, 0.6), (3, 0.4)],
}

# ────────────────────── CLI ──────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="여행 추천 시스템 추론")
    p.add_argument("--mode", required=True, choices=["city", "detail"])
    # 공통
    p.add_argument("--moods", type=str, help="무드 이름(쉼표 구분) 예: 설렘,모험")
    p.add_argument("--emotion_ids", type=str, help="감정 ID(쉼표 구분) 직접 지정")
    p.add_argument("--companions_count", type=int, required=True)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--recommendation_type", type=str, default="both")
    p.add_argument("--alpha", type=float, default=0.7)
    # city 모드
    p.add_argument("--trip_duration", type=int, help="여행 기간(일)")
    # detail 모드
    p.add_argument("--city", type=str)
    p.add_argument("--activity_type", type=str)
    p.add_argument("--activity_ids", type=str)
    p.add_argument("--preferred_transport", type=str)
    p.add_argument("--activity_level", type=int)
    p.add_argument("--place_name", type=str,
                   help="사용자가 반드시 포함하길 원하는 장소명")
    # 파일 경로
    p.add_argument("--models_dir", type=str,
                   default="C:/Users/adminastor/total1/ai/src/model")
    p.add_argument("--city_data", type=str,
                   default="C:/Users/adminastor/total1/ai/data/recommender/city/city_data_clean.csv")
    p.add_argument("--place_data", type=str,
                   default="C:/Users/adminastor/total1/ai/data/recommender/places/place_data_refined.csv")
    return p.parse_args()

# ────────────────────── 무드/감정 해석 ──────────────────────
def get_weighted_emotions(args) -> dict:
    """moods / emotion_ids 를 {emotion_id: weight} dict 로 반환"""
    weighted = {}

    # (A) 무드에서 가져오기
    if args.moods:
        for mood in [m.strip() for m in args.moods.split(",") if m.strip()]:
            for eid, w in MOOD_TO_EMOTIONS.get(mood, []):
                weighted[eid] = weighted.get(eid, 0) + w

    # (B) 사용자가 명시한 emotion_ids (가중치 1.0으로 보장)
    if args.emotion_ids:
        for eid in [int(e) for e in args.emotion_ids.split(",") if e]:
            weighted[eid] = max(weighted.get(eid, 0), 1.0)

    # 예외: 아무것도 없으면 기본값 = 기쁨
    if not weighted:
        weighted[1] = 1.0

    # 정규화
    s = sum(weighted.values())
    for k in weighted:
        weighted[k] /= s
    return weighted  # e.g. {1:0.5, 6:0.3, 4:0.2}

# ────────────────────── 데이터/모델 로드 ──────────────────────
def load_emotion_mapping(models_dir):
    path = os.path.join(models_dir, "emotion_mapping.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback 기본 매핑
    logger.warning("emotion_mapping.csv 없어서 기본 매핑 사용")
    return pd.DataFrame({
        "emotion_id": [1,2,3,4,5,6],
        "motive_id":  [7,2,1,2,1,7],
        "weight":     [0.8,0.9,0.9,0.9,0.9,0.9],
    })

def emotion_to_motive(weighted_emotions:dict, mapping_df:pd.DataFrame) -> dict:
    """{emotion: w} → {motive: agg_w}"""
    motive_w = {}
    for eid, e_w in weighted_emotions.items():
        rows = mapping_df[mapping_df["emotion_id"] == eid]
        for _, r in rows.iterrows():
            mid, m_w = int(r["motive_id"]), r["weight"]
            motive_w[mid] = motive_w.get(mid, 0) + e_w * m_w
    return motive_w

def load_city_data(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def load_place_data(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

# ────────────────────── 추천 로직 ──────────────────────
def recommend_cities(args, mapping_df, city_df):
    if city_df.empty:
        logger.error("도시 데이터가 비어 있습니다.")
        return {"recommendations": []}

    w_emotions = get_weighted_emotions(args)
    motive_w   = emotion_to_motive(w_emotions, mapping_df)

    results = []
    for _, city in city_df.iterrows():
        score = 0.0

        # 1) 여행 동기 매칭
        city_motives = [int(m) for m in str(city["motive_match"]).split(";") if m.isdigit()]
        score += sum(motive_w.get(mid, 0) for mid in city_motives) * 0.4

        # 2) 여행 기간
        if args.trip_duration and "avg_stay_duration" in city:
            diff = abs(city["avg_stay_duration"] - args.trip_duration)
            score += max(0, 5 - diff) / 5 * 0.3

        # 3) 동반자 수 적합성 (예시)
        feat = str(city.get("features", "")).lower()
        cscore = 0
        if args.companions_count == 1 and "혼자" in feat:
            cscore = 1
        elif args.companions_count == 2 and ("커플" in feat or "친구" in feat):
            cscore = 1
        elif args.companions_count >= 3 and ("가족" in feat or "단체" in feat):
            cscore = 1
        score += cscore * 0.3

        # 4) 시즌 보너스
        month = datetime.now().month
        season = ("Spring","Summer","Fall","Winter")[(month%12)//3]
        if season in str(city.get("season_preference","")) or "All" in str(city.get("season_preference","")):
            score += 0.2

        results.append({
            "city_id":  city["city_id"],
            "item_name":city["item_name"],
            "score":    score,
            "features": str(city.get("features","")).split(";")
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:args.top_n]

    # 간단한 활동 제안 예시
    for c in top:
        acts = []
        for f in c.pop("features", []):
            if f == "해변":    acts += ["해변 산책","수영"]
            elif f == "산":   acts += ["등산","트레킹"]
            elif f == "쇼핑": acts += ["쇼핑","로컬 마켓 탐방"]
            elif f == "역사적":acts += ["유적 탐방","박물관 방문"]
            elif f == "문화예술": acts += ["미술관", "공연 관람"]
        c["related_activities"] = list(dict.fromkeys(acts))[:3]  # 중복 제거
        c.pop("score", None)

    return {
        "user_id": f"u{abs(hash(str(w_emotions)+str(args.companions_count)))%1000}",
        "recommendations": top,
    }

def recommend_places(args, mapping_df, place_df):
    if place_df.empty:
        logger.error("장소 데이터가 비어 있습니다.")
        return {"places": []}
    
    # 🔧 도시 이름 정규화 (공백 제거 및 소문자 처리)
    args.city = args.city.strip().lower()
    place_df["city"] = place_df["city"].astype(str).str.strip().str.lower()

    subset = place_df[place_df["city"].str.contains(args.city, na=False)]

    if subset.empty:
        logger.warning(f"{args.city}에 해당하는 여행지가 없습니다.")
        return {"places": []}

    w_emotions = get_weighted_emotions(args)
    scores = []

    activity_ids = (
        [int(i) for i in args.activity_ids.split(",") if i.strip()]
        if args.activity_ids else []
    )

    for _, p in subset.iterrows():
        s = 0.0

        # 1) 실내/실외
        if args.activity_type and p["activity_type"] == args.activity_type:
            s += 1.0

        # 2) 활동 ID
        p_acts = [int(a) for a in str(p["activity_ids"]).split(";") if a.isdigit()]
        if activity_ids:
            match = sum(1 for a in activity_ids if a in p_acts)/len(activity_ids)
            s += match * 2.0

        # 3) 감정 매칭
        p_emotions = [int(e) for e in str(p["emotion_match"]).split(";") if e.isdigit()]
        s += sum(w_emotions.get(e,0) for e in p_emotions) * 1.5

        # 4) 활동 레벨
        if args.activity_level:
            diff = abs(p["activity_level"] - args.activity_level)
            s += max(0, 10-diff)/10 * 1.0

        # 5) 기타(교통/동반자) 필요 시 추가 …

        scores.append({"여행지명":p["여행지명"], "분류":p["분류"], "score":s})

    scores.sort(key=lambda x: x["score"], reverse=True)
    top = scores[:args.top_n]
    for t in top:
        t.pop("score", None)
    return {"places": top}


def recommend_final_places(args, mapping_df, place_df):
    """
    - recommend_places() 1차 결과 + place_name(필수 포함) 유지
    - 목표 개수 = args.top_n  (프런트에서 2×activityLevel 로 계산해 보냄)
    - 부족분은 동일 도시에서 스코어를 다시 계산해 채움
    - 반환: {"places": [...]}
    """
    args.city = args.city.strip().lower()
    place_df["city"] = place_df["city"].astype(str).str.strip().str.lower()
    # 1차 추천
    base = recommend_places(args, mapping_df, place_df)["places"]
    final_places = base[:]                 # 깊은 복사
    used = {p["여행지명"] for p in final_places}

    # ── (1) place_name 강제 포함 ─────────────────────────────
    if args.place_name and args.place_name not in used:
        row = place_df[(place_df["city"] == args.city) &
                       (place_df["여행지명"] == args.place_name)]
        if not row.empty:
            r = row.iloc[0]
            final_places.insert(0, {"여행지명": r["여행지명"], "분류": r["분류"]})
        else:
            # 데이터에 없더라도 이름만 넣어 준다
            final_places.insert(0, {"여행지명": args.place_name, "분류": "Unknown"})
        used.add(args.place_name)

    # ── (2) 목표 개수 맞추기 ────────────────────────────────
    target = args.top_n                  # 이미 2×activityLevel 로 넘어옴
    if len(final_places) < target:
        # 가중치 계산을 위해 공통 도구 재사용
        w_emotions  = get_weighted_emotions(args)
        req_act_ids = [int(i) for i in args.activity_ids.split(",")] if args.activity_ids else []

        extras = []
        for _, row in place_df[place_df["city"] == args.city].iterrows():
            if row["여행지명"] in used:
                continue

            score = 0.0
            if args.activity_type and row["activity_type"] == args.activity_type:
                score += 1.0

            row_acts = [int(a) for a in str(row["activity_ids"]).split(";") if a.isdigit()]
            if req_act_ids:
                match = sum(1 for a in req_act_ids if a in row_acts) / len(req_act_ids)
                score += match * 2.0

            row_emotions = [int(e) for e in str(row["emotion_match"]).split(";") if e.isdigit()]
            score += sum(w_emotions.get(e, 0) for e in row_emotions) * 1.5

            if args.activity_level is not None:
                diff = abs(row["activity_level"] - args.activity_level)
                score += max(0, 10 - diff) / 10

            extras.append({"여행지명": row["여행지명"], "분류": row["분류"], "score": score})

        extras.sort(key=lambda x: x["score"], reverse=True)
        need = target - len(final_places)

        for d in extras:
            if d["여행지명"] not in used:
                final_places.append({k: v for k, v in d.items() if k != "score"})
                used.add(d["여행지명"])
            if len(final_places) >= target:
                break

    # ── (3) 초과 시 자르기 ──────────────────────────────────
    final_places = final_places[:target]
    return {"places": final_places}

# ────────────────────── 메인 ──────────────────────
def main():
    args = parse_args()
    mapping_df = load_emotion_mapping(args.models_dir)

    if args.city:
        city_aliases = {
            "서울특별시": "서울",
            "부산광역시": "부산",
            "대구광역시": "대구",
            "인천광역시": "인천",
            "광주광역시": "광주",
            "대전광역시": "대전",
            "울산광역시": "울산",
            "세종특별자치시": "세종특별자치시",
            "경기도": "경기",
            "강원도": "강원",
            "충청북도": "충북",
            "충청남도": "충남",
            "전라북도": "전북",
            "전라남도": "전남",
            "경상북도": "경북",
            "경상남도": "경남",
            "제주특별자치도": "제주",
        }
        args.city = city_aliases.get(args.city.strip(), args.city.strip())

    if args.mode == "city":
        if not args.trip_duration:
            logger.error("--trip_duration 이 필요합니다.")
            sys.exit(1)
        city_df = load_city_data(args.city_data)
        result = recommend_cities(args, mapping_df, city_df)
        
    elif args.mode == "detail":
        for req, flag in [("city", args.city),
                      ("activity_type", args.activity_type),
                      ("activity_ids", args.activity_ids)]:
            if not flag:
                logger.error(f"--{req} 인자가 필요합니다.")
                sys.exit(1)
        place_df = load_place_data(args.place_data)
        result = recommend_final_places(args, mapping_df, place_df)
    else:
        logger.error("알 수 없는 모드")
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
