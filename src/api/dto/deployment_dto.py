"""
배포 관련 DTO 클래스 정의
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DeploymentRequestDTO:
    """배포 요청 DTO"""
    model_id: str
    environment: str
    deployment_config: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentRequestDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            model_id=data.get('model_id'),
            environment=data.get('environment'),
            deployment_config=data.get('deployment_config', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'model_id': self.model_id,
            'environment': self.environment,
            'deployment_config': self.deployment_config
        }

@dataclass
class DeploymentResponseDTO:
    """배포 응답 DTO"""
    deployment_id: str
    status: str
    model_id: str
    environment: str
    endpoint: str
    estimated_completion: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentResponseDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            deployment_id=data.get('deployment_id'),
            status=data.get('status'),
            model_id=data.get('model_id'),
            environment=data.get('environment'),
            endpoint=data.get('endpoint'),
            estimated_completion=data.get('estimated_completion')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'deployment_id': self.deployment_id,
            'status': self.status,
            'model_id': self.model_id,
            'environment': self.environment,
            'endpoint': self.endpoint,
            'estimated_completion': self.estimated_completion
        }

@dataclass
class DeploymentVersionRequestDTO:
    """배포 버전 요청 DTO"""
    deployment_id: str
    active: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentVersionRequestDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            deployment_id=data.get('deployment_id'),
            active=data.get('active')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'deployment_id': self.deployment_id,
            'active': self.active
        }

@dataclass
class DeploymentVersionResponseDTO:
    """배포 버전 응답 DTO"""
    status: str
    active_deployment_id: str
    previous_deployment_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentVersionResponseDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            status=data.get('status'),
            active_deployment_id=data.get('active_deployment_id'),
            previous_deployment_id=data.get('previous_deployment_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'status': self.status,
            'active_deployment_id': self.active_deployment_id,
            'previous_deployment_id': self.previous_deployment_id
        }