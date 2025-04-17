"""
추천 관련 더미 서비스 구현
"""
import random
from datetime import datetime

class DummyRecommendationService:
    """추천 관련 더미 서비스 클래스"""
    
    def __init__(self):
        """더미 추천 서비스 초기화"""
        # 더미 장소 데이터
        self.places = [
            {"id": "place_1", "name": "제주 성산일출봉", "type": "자연", "activities": ["자연 감상", "사진 촬영", "트레킹"], "transport": ["렌터카", "대중교통"]},
            {"id": "place_2", "name": "경주 불국사", "type": "역사", "activities": ["문화 체험", "사진 촬영", "역사 탐방"], "transport": ["대중교통"]},
            {"id": "place_3", "name": "서울 남산타워", "type": "랜드마크", "activities": ["야경 감상", "데이트", "산책"], "transport": ["대중교통", "도보"]},
            {"id": "place_4", "name": "부산 해운대", "type": "해변", "activities": ["해수욕", "휴식", "맛집 탐방"], "transport": ["대중교통", "렌터카"]},
            {"id": "place_5", "name": "전주 한옥마을", "type": "전통", "activities": ["문화 체험", "맛집 탐방", "전통 의상 체험"], "transport": ["대중교통", "자전거"]},
            {"id": "place_6", "name": "강원 설악산", "type": "산", "activities": ["등산", "트레킹", "자연 감상"], "transport": ["대중교통", "렌터카"]},
            {"id": "place_7", "name": "여수 오동도", "type": "섬", "activities": ["산책", "자연 감상", "사진 촬영"], "transport": ["대중교통", "렌터카"]},
            {"id": "place_8", "name": "인천 월미도", "type": "놀이공원", "activities": ["놀이기구", "맛집 탐방", "데이트"], "transport": ["대중교통"]},
            {"id": "place_9", "name": "속초 중앙시장", "type": "시장", "activities": ["쇼핑", "맛집 탐방", "문화 체험"], "transport": ["대중교통", "도보"]},
            {"id": "place_10", "name": "안동 하회마을", "type": "전통", "activities": ["문화 체험", "역사 탐방", "사진 촬영"], "transport": ["대중교통", "렌터카"]}
        ]
        
        # 더미 도시 데이터
        self.cities = [
            {"id": "city_1", "name": "서울", "activities": ["쇼핑", "맛집 탐방", "문화 체험", "역사 탐방"]},
            {"id": "city_2", "name": "부산", "activities": ["해수욕", "맛집 탐방", "야경 감상", "쇼핑"]},
            {"id": "city_3", "name": "제주", "activities": ["자연 감상", "트레킹", "해수욕", "맛집 탐방"]},
            {"id": "city_4", "name": "경주", "activities": ["역사 탐방", "문화 체험", "사진 촬영", "자전거"]},
            {"id": "city_5", "name": "전주", "activities": ["맛집 탐방", "문화 체험", "전통 의상 체험", "역사 탐방"]},
            {"id": "city_6", "name": "강릉", "activities": ["해수욕", "맛집 탐방", "카페 투어", "자연 감상"]},
            {"id": "city_7", "name": "여수", "activities": ["야경 감상", "맛집 탐방", "해수욕", "크루즈"]},
            {"id": "city_8", "name": "속초", "activities": ["맛집 탐방", "해수욕", "온천", "시장"]},
            {"id": "city_9", "name": "안동", "activities": ["문화 체험", "역사 탐방", "맛집 탐방", "사진 촬영"]},
            {"id": "city_10", "name": "춘천", "activities": ["자연 감상", "수상 스포츠", "맛집 탐방", "데이트"]}
        ]
    
    def get_recommendations(self, user_id=None, user_profile=None, top_n=5, recommendation_type="both", alpha=0.5):
        """
        여행지 추천 더미 구현
        
        Args:
            user_id: 사용자 ID
            user_profile: 사용자 프로필 객체
            top_n: 추천 개수
            recommendation_type: 추천 유형 (place, travel, both)
            alpha: 추천 알고리즘 가중치
            
        Returns:
            추천 결과 정보
        """
        # 추천 유형에 따라 결과 필터링
        places_to_recommend = self.places.copy()
        
        # 사용자 프로필이 있는 경우, 선호도에 따른 가중치 적용
        if user_profile and user_profile.preferred_activities:
            # 선호 활동이 많이 포함된 장소를 우선 추천
            places_to_recommend.sort(key=lambda x: self._calculate_activity_match_score(x, user_profile.preferred_activities), reverse=True)
        else:
            # 프로필이 없으면 랜덤 셔플
            random.shuffle(places_to_recommend)
        
        # top_n 개수만큼 추출
        top_places = places_to_recommend[:top_n]
        
        # 추천 장소를 응답 형식으로 변환
        recommendations = []
        for idx, place in enumerate(top_places):
            # 더미 추천 점수
            score = round(0.95 - (idx * 0.05), 2)
            
            # 더미 특성 점수
            feature_score = round(random.uniform(0.85, 0.98), 2)
            content_score = round(random.uniform(0.80, 0.95), 2)
            
            # 세부 정보
            details = {
                'avg_satisfaction': round(random.uniform(4.0, 4.9), 1),
                'avg_residence_time': random.randint(60, 180),
                'related_activities': place['activities'],
                'related_transport': place['transport']
            }
            
            recommendations.append({
                'item_id': place['id'],
                'item_name': place['name'],
                'type': 'place',
                'score': score,
                'source': 'hybrid',
                'feature_score': feature_score,
                'content_score': content_score,
                'details': details
            })
        
        # 더미 사용자 선호도 요약
        user_preference_summary = None
        if user_profile:
            user_preference_summary = {
                'top_activities': user_profile.preferred_activities[:3] if user_profile.preferred_activities else ["여행", "관광", "휴식"],
                'preferred_transport': user_profile.preferred_transport[:2] if user_profile.preferred_transport else ["대중교통"]
            }
        
        # 더미 응답 반환
        return {
            'user_id': user_id or "anonymous",
            'recommendations': recommendations,
            'user_preference_summary': user_preference_summary
        }
    
    def get_city_recommendations(self, user_id=None, user_profile=None, top_n=5, alpha=0.5):
        """
        도시 추천 더미 구현
        
        Args:
            user_id: 사용자 ID
            user_profile: 사용자 프로필 객체
            top_n: 추천 개수
            alpha: 추천 알고리즘 가중치
            
        Returns:
            도시 추천 결과 정보
        """
        # 추천할 도시 목록
        cities_to_recommend = self.cities.copy()
        
        # 사용자 프로필이 있는 경우, 선호도에 따른 가중치 적용
        if user_profile and user_profile.preferred_activities:
            # 선호 활동이 많이 포함된 도시를 우선 추천
            cities_to_recommend.sort(key=lambda x: self._calculate_activity_match_score(x, user_profile.preferred_activities), reverse=True)
        else:
            # 프로필이 없으면 랜덤 셔플
            random.shuffle(cities_to_recommend)
        
        # top_n 개수만큼 추출
        top_cities = cities_to_recommend[:top_n]
        
        # 추천 도시를 응답 형식으로 변환
        recommendations = []
        for city in top_cities:
            recommendations.append({
                'city_id': city['id'],
                'item_name': city['name'],
                'related_activities': city['activities']
            })
        
        # 더미 응답 반환
        return {
            'user_id': user_id or "anonymous",
            'recommendations': recommendations
        }
    
    def _calculate_activity_match_score(self, item, preferred_activities):
        """
        선호 활동 매칭 점수 계산
        
        Args:
            item: 장소 또는 도시 데이터
            preferred_activities: 선호 활동 목록
            
        Returns:
            매칭 점수
        """
        if not preferred_activities:
            return 0
            
        # 활동 목록
        activities = item.get('activities', [])
        
        # 매칭되는 활동 수
        matches = sum(1 for activity in activities if activity in preferred_activities)
        
        # 매칭 점수 (활동이 많을수록 정확도가 떨어지는 것을 보정)
        return matches / max(1, len(activities)) * (matches / max(1, len(preferred_activities)))