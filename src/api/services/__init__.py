"""
서비스 패키지 초기화
"""

from src.api.services.dummy_train_service import DummyTrainService
from src.api.services.dummy_deployment_service import DummyDeploymentService
from src.api.services.dummy_model_incident_service import (
    DummyModelService,
    DummyIncidentService,
)
from src.api.services.dummy_recommendation_service import DummyRecommendationService


def create_dummy_services():
    """
    더미 서비스 객체들을 생성

    Returns:
        서비스 객체 딕셔너리
    """
    return {
        "train_service": DummyTrainService(),
        "deployment_service": DummyDeploymentService(),
        "model_service": DummyModelService(),
        "incident_service": DummyIncidentService(),
        "recommendation_service": DummyRecommendationService(),
    }


__all__ = [
    "DummyTrainService",
    "DummyDeploymentService",
    "DummyModelService",
    "DummyIncidentService",
    "DummyRecommendationService",
    "create_dummy_services",
]
