"""
DTO 패키지 초기화
"""
from src.api.dto.train_dto import (
    TrainRequestDTO,
    TrainResponseDTO,
    ValidationRequestDTO,
    ValidationResponseDTO
)

from src.api.dto.deployment_dto import (
    DeploymentRequestDTO,
    DeploymentResponseDTO,
    DeploymentVersionRequestDTO,
    DeploymentVersionResponseDTO
)

from src.api.dto.recommendation_dto import (
    UserProfileDTO,
    RecommendationRequestDTO,
    RecommendationItemDTO,
    RecommendationResponseDTO,
    CityRecommendationResponseDTO
)

from src.api.dto.model_incident_dto import (
    ModelListRequestDTO,
    ModelInfoDTO,
    ModelListResponseDTO,
    ModelParameterUpdateRequestDTO,
    ModelParameterUpdateResponseDTO,
    IncidentReportRequestDTO,
    IncidentReportResponseDTO
)

__all__ = [
    # 학습 관련 DTO
    'TrainRequestDTO',
    'TrainResponseDTO',
    'ValidationRequestDTO',
    'ValidationResponseDTO',
    
    # 배포 관련 DTO
    'DeploymentRequestDTO',
    'DeploymentResponseDTO',
    'DeploymentVersionRequestDTO',
    'DeploymentVersionResponseDTO',
    
    # 추천 관련 DTO
    'UserProfileDTO',
    'RecommendationRequestDTO',
    'RecommendationItemDTO',
    'RecommendationResponseDTO',
    'CityRecommendationResponseDTO',
    
    # 모델 관리 및 장애 관련 DTO
    'ModelListRequestDTO',
    'ModelInfoDTO',
    'ModelListResponseDTO',
    'ModelParameterUpdateRequestDTO',
    'ModelParameterUpdateResponseDTO',
    'IncidentReportRequestDTO',
    'IncidentReportResponseDTO'
]