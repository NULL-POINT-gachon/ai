"""
학습 관련 컨트롤러
"""
import logging
from flask import request, jsonify
from src.api.dto.train_dto import (
    TrainRequestDTO, 
    TrainResponseDTO,
    ValidationRequestDTO,
    ValidationResponseDTO
)
from src.api.utils.error_handler import ValidationError, ProcessingError
from src.api.utils.validators import validate_json

# 로깅 설정
logger = logging.getLogger(__name__)

class TrainController:
    """학습 관련 요청을 처리하는 컨트롤러"""
    
    def __init__(self, train_service):
        """
        Args:
            train_service: 학습 서비스 객체
        """
        self.train_service = train_service
        
    @validate_json('model_name', 'model_type', 'parameters')
    def train_model(self):
        """모델 학습 엔드포인트
        
        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            train_request = TrainRequestDTO.from_dict(data)
            
            # 서비스 호출
            result = self.train_service.train_model(
                train_request.model_name,
                train_request.model_type,
                train_request.parameters,
                train_request.training_data_config
            )
            
            # 응답 생성
            response = TrainResponseDTO(
                training_id=result.get('training_id'),
                status=result.get('status'),
                estimated_time=result.get('estimated_time'),
                model_id=result.get('model_id')
            )
            
            return jsonify(response.to_dict()), 200
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 400
            
        except ProcessingError as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            return jsonify({"error": "서버 내부 오류가 발생했습니다.", "status": "error"}), 500
    
    @validate_json('model_id', 'test_data_config')
    def validate_model(self):
        """모델 검증 엔드포인트
        
        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            validation_request = ValidationRequestDTO.from_dict(data)
            
            # 서비스 호출
            result = self.train_service.validate_model(
                validation_request.model_id,
                validation_request.test_data_config
            )
            
            # 응답 생성
            response = ValidationResponseDTO(
                validation_id=result.get('validation_id'),
                model_id=result.get('model_id'),
                status=result.get('status'),
                metrics=result.get('metrics', {}),
                validation_plots=result.get('validation_plots', [])
            )
            
            return jsonify(response.to_dict()), 200
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 400
            
        except ProcessingError as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            return jsonify({"error": "서버 내부 오류가 발생했습니다.", "status": "error"}), 500