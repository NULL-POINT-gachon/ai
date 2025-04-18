"""
엔드포인트 패키지 초기화
"""
from src.api.endpoints.health import register_endpoints as register_health_endpoints
from src.api.endpoints.train import register_endpoints as register_train_endpoints
from src.api.endpoints.deployment import register_endpoints as register_deployment_endpoints
from src.api.endpoints.model import register_model_endpoints, register_incident_endpoints
from src.api.endpoints.recommendation import register_endpoints as register_recommendation_endpoints

def register_all_endpoints(api, services):
    """
    모든 API 엔드포인트를 등록
    
    Args:
        api: flask_restx API 객체
        services: 서비스 객체 딕셔너리
    
    Returns:
        flask_restx API 객체
    """
    # 상태 확인 엔드포인트 등록
    register_health_endpoints(api)
    
    # 학습 관련 엔드포인트 등록
    if 'train_service' in services:
        register_train_endpoints(api, services['train_service'])
    
    # 배포 관련 엔드포인트 등록
    if 'deployment_service' in services:
        register_deployment_endpoints(api, services['deployment_service'])
    
    # 모델 관리 관련 엔드포인트 등록
    if 'model_service' in services:
        register_model_endpoints(api, services['model_service'])
    
    # 장애 관련 엔드포인트 등록
    if 'incident_service' in services:
        register_incident_endpoints(api, services['incident_service'])
    
    # 추천 관련 엔드포인트 등록
    if 'recommendation_service' in services:
        register_recommendation_endpoints(api, services['recommendation_service'])
    
    return api