"""
학습 관련 DTO 클래스 정의
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class TrainRequestDTO:
    """학습 요청 DTO"""

    model_name: str
    model_type: str
    parameters: Dict[str, Any]
    training_data_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainRequestDTO":
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            model_name=data.get("model_name"),
            model_type=data.get("model_type"),
            parameters=data.get("parameters", {}),
            training_data_config=data.get("training_data_config"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "training_data_config": self.training_data_config,
        }


@dataclass
class TrainResponseDTO:
    """학습 응답 DTO"""

    training_id: str
    status: str
    estimated_time: str
    model_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainResponseDTO":
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            training_id=data.get("training_id"),
            status=data.get("status"),
            estimated_time=data.get("estimated_time"),
            model_id=data.get("model_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            "training_id": self.training_id,
            "status": self.status,
            "estimated_time": self.estimated_time,
            "model_id": self.model_id,
        }


@dataclass
class ValidationRequestDTO:
    """검증 요청 DTO"""

    model_id: str
    test_data_config: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationRequestDTO":
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            model_id=data.get("model_id"),
            test_data_config=data.get("test_data_config", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {"model_id": self.model_id, "test_data_config": self.test_data_config}


@dataclass
class ValidationResponseDTO:
    """검증 응답 DTO"""

    validation_id: str
    model_id: str
    status: str
    metrics: Dict[str, Any]
    validation_plots: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResponseDTO":
        """딕셔너리에서 DTO 객체 생성"""
        return cls(
            validation_id=data.get("validation_id"),
            model_id=data.get("model_id"),
            status=data.get("status"),
            metrics=data.get("metrics", {}),
            validation_plots=data.get("validation_plots", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return {
            "validation_id": self.validation_id,
            "model_id": self.model_id,
            "status": self.status,
            "metrics": self.metrics,
            "validation_plots": self.validation_plots,
        }
