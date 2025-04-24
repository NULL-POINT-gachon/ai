"""
추천 관련 API 엔드포인트 정의
"""

from flask_restx import Namespace, Resource, fields

# 네임스페이스 생성
recommendation_ns = Namespace("recommendations", description="여행지 추천 API")
city_recommendation_ns = Namespace("recommendation/city", description="도시 추천 API")


def register_endpoints(api, recommendation_service):
    """
    추천 관련 엔드포인트 등록

    Args:
        api: flask_restx API 객체
        recommendation_service: 추천 서비스 객체
    """
    api.add_namespace(recommendation_ns)
    api.add_namespace(city_recommendation_ns)

    # 공통 모델 정의
    user_profile = api.model(
        "UserProfile",
        {
            "travel_style": fields.String(description="여행 스타일 코드"),
            "travel_motive": fields.String(description="여행 동기 코드"),
            "age_group": fields.String(description="연령대"),
            "gender": fields.String(description="성별"),
            "preferred_activities": fields.List(
                fields.String, description="선호 활동 코드 목록"
            ),
            "preferred_transport": fields.List(
                fields.String, description="선호 이동수단 목록"
            ),
        },
    )

    # 도시 추천 모델 정의
    city_request = api.model(
        "CityRecommendationRequest",
        {
            "user_id": fields.String(description="사용자 ID (기존 사용자의 경우)"),
            "user_profile": fields.Nested(user_profile, description="프로필"),
            "top_n": fields.Integer(description="추천 개수", default=5),
            "recommendation_type": fields.String(
                description="추천 유형(place, travel, both)", default="both"
            ),
            "alpha": fields.Float(description="추천 알고리즘 가중치(0-1)", default=0.7),
        },
    )

    city_item = api.model(
        "CityRecommendationItem",
        {
            "city_id": fields.String(description="추천 항목 ID"),
            "item_name": fields.String(description="추천 항목 이름"),
            "related_activities": fields.List(fields.String, description="연관 항목"),
        },
    )

    city_response = api.model(
        "CityRecommendationResponse",
        {
            "user_id": fields.String(description="사용자 ID"),
            "recommendations": fields.List(
                fields.Nested(city_item), description="추천 목록"
            ),
        },
    )

    # 일반 추천 모델 정의
    recommendation_request = api.model(
        "RecommendationRequest",
        {
            "user_id": fields.String(description="사용자 ID (기존 사용자의 경우)"),
            "user_profile": fields.Nested(user_profile, description="프로필"),
            "top_n": fields.Integer(description="추천 개수", default=5),
            "recommendation_type": fields.String(
                description="추천 유형(place, travel, both)", default="both"
            ),
            "alpha": fields.Float(description="추천 알고리즘 가중치(0-1)", default=0.7),
        },
    )

    recommendation_details = api.model(
        "RecommendationDetails",
        {
            "avg_satisfaction": fields.Float(description="평균 만족도"),
            "avg_residence_time": fields.Integer(description="평균 체류 시간"),
            "related_activities": fields.List(fields.String, description="연관 활동"),
            "related_transport": fields.List(
                fields.String, description="연관 이동수단"
            ),
        },
    )

    recommendation_item = api.model(
        "RecommendationItem",
        {
            "item_id": fields.String(description="추천 항목 ID"),
            "item_name": fields.String(description="추천 항목 이름"),
            "type": fields.String(description="항목 유형(place, travel)"),
            "score": fields.Float(description="추천 점수"),
            "source": fields.String(description="추천 소스(hybrid, feature, content)"),
            "feature_score": fields.Float(description="특성 기반 점수"),
            "content_score": fields.Float(description="콘텐츠 기반 점수"),
            "details": fields.Nested(
                recommendation_details, description="추천 상세 정보"
            ),
        },
    )

    recommendation_response = api.model(
        "RecommendationResponse",
        {
            "user_id": fields.String(description="사용자 ID"),
            "recommendations": fields.List(
                fields.Nested(recommendation_item), description="추천 목록"
            ),
        },
    )

    @city_recommendation_ns.route("")
    class CityRecommendationResource(Resource):
        """도시 추천 리소스"""

        @city_recommendation_ns.doc(description="도시 추천 기능")
        @city_recommendation_ns.expect(city_request)
        @city_recommendation_ns.marshal_with(city_response)
        def get(self):
            """
            GET 방식 도시 추천 엔드포인트

            Returns:
                도시 추천 목록
            """
            return recommendation_service.get_city_recommendations(
                city_recommendation_ns.payload
            )

        @city_recommendation_ns.doc(description="도시 추천 기능")
        @city_recommendation_ns.expect(city_request)
        @city_recommendation_ns.marshal_with(city_response)
        def post(self):
            """
            POST 방식 도시 추천 엔드포인트

            Returns:
                도시 추천 목록
            """
            return recommendation_service.get_city_recommendations(
                city_recommendation_ns.payload
            )

    @recommendation_ns.route("")
    class RecommendationResource(Resource):
        """여행지 추천 리소스"""

        @recommendation_ns.doc(description="여행지 추천 기능")
        @recommendation_ns.expect(recommendation_request)
        @recommendation_ns.marshal_with(recommendation_response)
        def get(self):
            """
            GET 방식 여행지 추천 엔드포인트

            Returns:
                여행지 추천 목록
            """
            return recommendation_service.get_recommendations(recommendation_ns.payload)

        @recommendation_ns.doc(description="여행지 추천 기능")
        @recommendation_ns.expect(recommendation_request)
        @recommendation_ns.marshal_with(recommendation_response)
        def post(self):
            """
            POST 방식 여행지 추천 엔드포인트

            Returns:
                여행지 추천 목록
            """
            return recommendation_service.get_recommendations(recommendation_ns.payload)

    return api
