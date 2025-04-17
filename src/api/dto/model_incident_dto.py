"""
모델 관리 및 장애 기록 관련 DTO 클래스 정의
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class ModelListRequestDTO:
    """모델 목록 요청 DTO"""
    status: Optional[str] = None
    model_type: Optional[str] = None
    page: int = 1
    limit: int = 10
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelListRequestDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            status=data.get('status'),
            model_type=data.get('model_type'),
            page=int(data.get('page', 1)),
            limit=int(data.get('limit', 10))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'status': self.status,
            'model_type': self.model_type,
            'page': self.page,
            'limit': self.limit
        }

@dataclass
class ModelInfoDTO:
    """모델 정보 DTO"""
    model_id: str
    model_name: str
    model_type: str
    created_at: str
    status: str
    metrics: Dict[str, Any]
    is_deployed: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfoDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            model_id=data.get('model_id'),
            model_name=data.get('model_name'),
            model_type=data.get('model_type'),
            created_at=data.get('created_at'),
            status=data.get('status'),
            metrics=data.get('metrics', {}),
            is_deployed=data.get('is_deployed', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'created_at': self.created_at,
            'status': self.status,
            'metrics': self.metrics,
            'is_deployed': self.is_deployed
        }

@dataclass
class ModelListResponseDTO:
    """모델 목록 응답 DTO"""
    total: int
    page: int
    limit: int
    models: List[ModelInfoDTO] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelListResponseDTO':
        """딕셔너리에서 DTO 객체 생성"""
        models = []
        for model_data in data.get('models', []):
            models.append(ModelInfoDTO.from_dict(model_data))
        
        return cls(
            total=data.get('total', 0),
            page=data.get('page', 1),
            limit=data.get('limit', 10),
            models=models
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'total': self.total,
            'page': self.page,
            'limit': self.limit,
            'models': [model.to_dict() for model in self.models]
        }

@dataclass
class ModelParameterUpdateRequestDTO:
    """모델 파라미터 업데이트 요청 DTO"""
    parameters: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParameterUpdateRequestDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            parameters=data.get('parameters', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'parameters': self.parameters
        }

@dataclass
class ModelParameterUpdateResponseDTO:
    """모델 파라미터 업데이트 응답 DTO"""
    status: str
    model_id: str
    updated_parameters: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParameterUpdateResponseDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            status=data.get('status'),
            model_id=data.get('model_id'),
            updated_parameters=data.get('updated_parameters', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'status': self.status,
            'model_id': self.model_id,
            'updated_parameters': self.updated_parameters
        }

@dataclass
class IncidentReportRequestDTO:
    """장애 기록 요청 DTO"""
    severity: str
    component: str
    description: str
    error_details: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IncidentReportRequestDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            severity=data.get('severity'),
            component=data.get('component'),
            description=data.get('description'),
            error_details=data.get('error_details', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'severity': self.severity,
            'component': self.component,
            'description': self.description,
            'error_details': self.error_details
        }

@dataclass
class IncidentReportResponseDTO:
    """장애 기록 응답 DTO"""
    incident_id: str
    timestamp: str
    status: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IncidentReportResponseDTO':
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            incident_id=data.get('incident_id'),
            timestamp=data.get('timestamp'),
            status=data.get('status')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            'incident_id': self.incident_id,
            'timestamp': self.timestamp,
            'status': self.status
        }