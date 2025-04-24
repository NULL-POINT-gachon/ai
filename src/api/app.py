"""
Flask 앱 생성 및 설정
"""

from flask import Flask
from flask_restx import Api, Resource
from src.api.config import active_config
from src.api.services import create_dummy_services
from src.api.utils.error_handler import register_error_handlers


def create_app():
    """
    Flask 앱 생성 및 설정

    Returns:
        설정이 완료된 Flask 앱 객체
    """
    # Flask 앱 생성
    app = Flask(__name__)

    # 앱 설정 적용
    app.config.from_object(active_config)

    # API 접두사 설정
    api_prefix = app.config.get("API_PREFIX", "/api/v1")

    # flask_restx API 생성
    api = Api(
        app,
        version="1.0",
        title="AI 기반 맞춤형 여행 계획 추천 API",
        description="여행지 추천, 학습, 배포 API",
        doc="/swagger/",  # Swagger UI URL
    )

    # 더미 서비스 생성
    services = create_dummy_services()

    # 에러 핸들러 등록
    app = register_error_handlers(app)

    # 모든 엔드포인트 등록
    from src.api.endpoints import register_all_endpoints

    register_all_endpoints(api, services)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
