"""
모델 관리 및 장애 관련 API 엔드포인트 정의
"""

from flask_restx import Namespace, Resource, fields
from flask import request

# 네임스페이스 생성
model_ns = Namespace("recommendation/models", description="AI 여행지 추천 관리 API")
incident_ns = Namespace(
    "recommendation/incidents", description="AI 여행지 추천 장애기록 API"
)


def register_model_endpoints(api, model_service):
    """
    모델 관리 관련 엔드포인트 등록

    Args:
        api: flask_restx API 객체
        model_service: 모델 서비스 객체
    """
    api.add_namespace(model_ns)

    # 모델 정의
    model_metrics = model_ns.model(
        "ModelMetrics",
        {
            "rmse": fields.Float(description="Root Mean Square Error"),
            "mae": fields.Float(description="Mean Absolute Error"),
        },
    )

    model_info = model_ns.model(
        "ModelInfo",
        {
            "model_id": fields.String(description="모델 ID"),
            "model_name": fields.String(description="모델 이름"),
            "model_type": fields.String(description="모델 유형"),
            "created_at": fields.String(description="생성 시간"),
            "status": fields.String(description="모델 상태"),
            "metrics": fields.Nested(model_metrics, description="모델 성능 지표"),
            "is_deployed": fields.Boolean(description="배포 여부"),
        },
    )

    models_response = model_ns.model(
        "ModelsResponse",
        {
            "total": fields.Integer(description="전체 모델 수"),
            "page": fields.Integer(description="현재 페이지 번호"),
            "limit": fields.Integer(description="페이지당 결과 수"),
            "models": fields.List(fields.Nested(model_info), description="모델 목록"),
        },
    )

    parameters_update = model_ns.model(
        "ParametersUpdate",
        {"parameters": fields.Raw(description="수정할 파라미터", required=True)},
    )

    parameters_response = model_ns.model(
        "ParametersResponse",
        {
            "status": fields.String(description="작업 상태"),
            "model_id": fields.String(description="수정된 모델 ID"),
            "updated_parameters": fields.List(
                fields.String, description="수정된 파라미터 목록"
            ),
        },
    )

    @model_ns.route("")
    class ModelsResource(Resource):
        """모델 관리 리소스"""

        @model_ns.doc(
            description="여행지 추천 모듈 버전관리 기능",
            params={
                "status": "모델 상태 필터링(optional)",
                "model_type": "모델 유형 필터링(optional)",
                "page": "페이지 번호(optional)",
                "limit": "페이지당 결과 수(optional)",
            },
        )
        @model_ns.marshal_with(models_response)
        def get(self):
            """
            모델 목록 조회 엔드포인트

            Returns:
                모델 목록 정보
            """
            filters = {
                "status": request.args.get("status"),
                "model_type": request.args.get("model_type"),
                "page": request.args.get("page", 1, type=int),
                "limit": request.args.get("limit", 10, type=int),
            }
            return model_service.list_models(filters)

    @model_ns.route("/<string:model_id>/parameters")
    @model_ns.param("model_id", "모델 ID")
    class ModelParametersResource(Resource):
        """모델 파라미터 관리 리소스"""

        @model_ns.doc(description="여행지 추천 모듈 파라미터 수정 기능")
        @model_ns.expect(parameters_update)
        @model_ns.marshal_with(parameters_response)
        def put(self, model_id):
            """
            모델 파라미터 수정 엔드포인트

            Args:
                model_id: 모델 ID

            Returns:
                파라미터 수정 결과
            """
            return model_service.update_model_parameters(model_id, model_ns.payload)

    return api


def register_incident_endpoints(api, incident_service):
    """
    장애 관련 엔드포인트 등록

    Args:
        api: flask_restx API 객체
        incident_service: 장애 서비스 객체
    """
    api.add_namespace(incident_ns)

    # 모델 정의
    error_details = incident_ns.model(
        "ErrorDetails",
        {
            "error_code": fields.String(description="오류 코드", required=True),
            "error_message": fields.String(description="오류 메시지", required=True),
            "stack_trace": fields.String(description="스택 트레이스"),
        },
    )

    incident_request = incident_ns.model(
        "IncidentRequest",
        {
            "severity": fields.String(
                description="심각도(high, medium, low)", required=True
            ),
            "component": fields.String(description="장애 발생 컴포넌트", required=True),
            "description": fields.String(description="장애 설명", required=True),
            "error_details": fields.Nested(
                error_details, description="오류 세부 정보", required=True
            ),
        },
    )

    incident_response = incident_ns.model(
        "IncidentResponse",
        {
            "incident_id": fields.String(description="장애 ID"),
            "timestamp": fields.String(description="발생 시간"),
            "status": fields.String(description="장애 상태"),
        },
    )

    @incident_ns.route("")
    class IncidentResource(Resource):
        """장애 기록 리소스"""

        @incident_ns.doc(description="여행지 추천 모듈 장애기록 기능")
        @incident_ns.expect(incident_request)
        @incident_ns.marshal_with(incident_response)
        def post(self):
            """
            장애 기록 엔드포인트

            Returns:
                장애 기록 결과
            """
            return incident_service.report_incident(incident_ns.payload)

    return api
