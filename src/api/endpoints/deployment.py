"""
배포 관련 API 엔드포인트 정의
"""

from flask_restx import Namespace, Resource, fields

# 네임스페이스 생성
deployment_ns = Namespace(
    "recommendation/deployment", description="AI 여행지 추천 배포 API"
)


def register_endpoints(api, deployment_service):
    """
    배포 관련 엔드포인트 등록

    Args:
        api: flask_restx API 객체
        deployment_service: 배포 서비스 객체
    """
    api.add_namespace(deployment_ns)

    # 모델 정의
    deployment_config = deployment_ns.model(
        "DeploymentConfig",
        {
            "auto_scaling": fields.Boolean(
                description="자동 스케일링 활성화 여부", required=True
            ),
            "min_instances": fields.Integer(
                description="최소 인스턴스 수", required=True
            ),
            "max_instances": fields.Integer(
                description="최대 인스턴스 수", required=True
            ),
            "enable_caching": fields.Boolean(
                description="캐싱 활성화 여부", required=True
            ),
            "cache_ttl": fields.Integer(
                description="캐시 유효 시간(초)", required=True
            ),
        },
    )

    deployment_request = deployment_ns.model(
        "DeploymentRequest",
        {
            "model_id": fields.String(description="배포할 모델 ID", required=True),
            "environment": fields.String(
                description="배포 환경(staging, production)", required=True
            ),
            "deployment_config": fields.Nested(
                deployment_config, description="배포 설정", required=True
            ),
        },
    )

    deployment_response = deployment_ns.model(
        "DeploymentResponse",
        {
            "deployment_id": fields.String(description="배포 ID"),
            "status": fields.String(description="배포 상태"),
            "model_id": fields.String(description="모델 ID"),
            "environment": fields.String(description="배포 환경"),
            "endpoint": fields.String(description="추론 API 엔드포인트"),
            "estimated_completion": fields.String(description="예상 완료 시간"),
        },
    )

    version_request = deployment_ns.model(
        "VersionRequest",
        {
            "deployment_id": fields.String(
                description="활성화할 배포 ID", required=True
            ),
            "active": fields.Boolean(
                description="활성화 상태 설정(true: 활성화)", required=True
            ),
        },
    )

    version_response = deployment_ns.model(
        "VersionResponse",
        {
            "status": fields.String(description="작업 상태"),
            "active_deployment_id": fields.String(description="현재 활성화된 배포 ID"),
            "previous_deployment_id": fields.String(
                description="이전 활성화된 배포 ID"
            ),
        },
    )

    @deployment_ns.route("")
    class DeploymentResource(Resource):
        """모델 배포 리소스"""

        @deployment_ns.doc(description="여행지 추천 모듈 배포 기능")
        @deployment_ns.expect(deployment_request)
        @deployment_ns.marshal_with(deployment_response)
        def post(self):
            """
            모델 배포 엔드포인트

            Returns:
                배포 상태 정보
            """
            return deployment_service.deploy_model(deployment_ns.payload)

    @deployment_ns.route("/version")
    class VersionResource(Resource):
        """배포 버전 관리 리소스"""

        @deployment_ns.doc(description="여행지 추천 모듈 배포 버전 선택 기능")
        @deployment_ns.expect(version_request)
        @deployment_ns.marshal_with(version_response)
        def put(self):
            """
            배포 버전 활성화 엔드포인트

            Returns:
                활성화 결과 정보
            """
            return deployment_service.update_deployment_version(deployment_ns.payload)

    return api
