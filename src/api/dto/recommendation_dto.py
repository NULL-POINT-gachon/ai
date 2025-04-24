"""
추천 관련 DTO 클래스 정의
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class UserProfileDTO:
    """사용자 프로필 DTO"""

    travel_style: Optional[str] = None
    travel_motive: Optional[str] = None
    age_group: Optional[str] = None
    gender: Optional[str] = None
    preferred_activities: List[str] = field(default_factory=list)
    preferred_transport: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfileDTO":
        """딕셔너리에서 DTO 객체 생성"""
        if data is None:
            return cls()
        return cls(
            travel_style=data.get("travel_style"),
            travel_motive=data.get("travel_motive"),
            age_group=data.get("age_group"),
            gender=data.get("gender"),
            preferred_activities=data.get("preferred_activities", []),
            preferred_transport=data.get("preferred_transport", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            "travel_style": self.travel_style,
            "travel_motive": self.travel_motive,
            "age_group": self.age_group,
            "gender": self.gender,
            "preferred_activities": self.preferred_activities,
            "preferred_transport": self.preferred_transport,
        }


@dataclass
class RecommendationRequestDTO:
    """추천 요청 DTO"""

    user_id: Optional[str] = None
    user_profile: Optional[UserProfileDTO] = None
    top_n: int = 5
    recommendation_type: str = "both"
    alpha: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationRequestDTO":
        """딕셔너리에서 DTO 객체 생성"""
        user_profile = (
            UserProfileDTO.from_dict(data.get("user_profile"))
            if data.get("user_profile")
            else None
        )
        return cls(
            user_id=data.get("user_id"),
            user_profile=user_profile,
            top_n=data.get("top_n", 5),
            recommendation_type=data.get("recommendation_type", "both"),
            alpha=data.get("alpha", 0.5),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        result = {
            "user_id": self.user_id,
            "top_n": self.top_n,
            "recommendation_type": self.recommendation_type,
            "alpha": self.alpha,
        }
        if self.user_profile:
            result["user_profile"] = self.user_profile.to_dict()
        return result


@dataclass
class RecommendationItemDTO:
    """추천 항목 DTO"""

    item_id: str
    item_name: str
    type: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    related_activities: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationItemDTO":
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            item_id=data.get("item_id"),
            item_name=data.get("item_name"),
            type=data.get("type"),
            score=data.get("score"),
            source=data.get("source"),
            details=data.get("details"),
            related_activities=data.get("related_activities", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "type": self.type,
            "score": self.score,
            "source": self.source,
            "details": self.details,
            "related_activities": self.related_activities,
        }


@dataclass
class RecommendationResponseDTO:
    """추천 응답 DTO"""

    user_id: str
    recommendations: List[RecommendationItemDTO] = field(default_factory=list)
    user_preference_summary: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationResponseDTO":
        """딕셔너리에서 DTO 객체 생성"""
        recommendations = []
        for item_data in data.get("recommendations", []):
            recommendations.append(RecommendationItemDTO.from_dict(item_data))

        return cls(
            user_id=data.get("user_id"),
            recommendations=recommendations,
            user_preference_summary=data.get("user_preference_summary"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            "user_id": self.user_id,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "user_preference_summary": self.user_preference_summary,
        }


@dataclass
class CityRecommendationResponseDTO:
    """도시 추천 응답 DTO"""

    user_id: str
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CityRecommendationResponseDTO":
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            user_id=data.get("user_id"), recommendations=data.get("recommendations", [])
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {"user_id": self.user_id, "recommendations": self.recommendations}
