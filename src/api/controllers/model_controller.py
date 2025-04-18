"""
모델 관리 관련 컨트롤러
"""
import logging
from flask import request, jsonify
from src.api.dto.model_incident_dto import (
    ModelListRequestDTO,
    ModelListResponseDTO,
    ModelParameterUpdateRequestDTO,
    ModelParameterUpdateResponseDTO,
    IncidentReportRequestDTO,
    IncidentReportResponseDTO
)
from src.api.utils.error_handler import ValidationError, ProcessingError, NotFoundError
from src.api.utils.validators import validate_json

# 로깅 설정
logger = logging.getLogger(__name__)

class ModelController:
    """모델 관리 관련 요청을 처리하는 컨트롤러"""
    
    def __init__(self, model_service):
        """
        Args:
            model_service: 모델 서비스 객체
        """
        self.model_service = model_service
        
    def list_models(self):
        """모델 목록 조회 엔드포인트
        
        Returns:
            JSON 응답
        """
        try:
            # 쿼리 파라미터 파싱
            status = request.args.get('status')
            model_type = request.args.get('model_type')
            page = int(request.args.get('page', 1))
            limit = int(request.args.get('limit', 10))
            
            request_dto = ModelListRequestDTO(
                status=status,
                model_type=model_type,
                page=page,
                limit=limit
            )
            
            # 서비스 호출
            result = self.model_service.list_models(
                request_dto.status,
                request_dto.model_type,
                request_dto.page,
                request_dto.limit
            )
            
            # 응답 생성
            response = ModelListResponseDTO.from_dict(result)
            
            return jsonify(response.to_dict()), 200
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 400
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            return jsonify({"error": "서버 내부 오류가 발생했습니다.", "status": "error"}), 500
    
    @validate_json('parameters')
    def update_model_parameters(self, model_id):
        """모델 파라미터 수정 엔드포인트
        
        Args:
            model_id: 모델 ID
            
        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            param_request = ModelParameterUpdateRequestDTO.from_dict(data)
            
            # 서비스 호출
            result = self.model_service.update_model_parameters(
                model_id,
                param_request.parameters
            )
            
            # 응답 생성
            response = ModelParameterUpdateResponseDTO(
                status=result.get('status'),
                model_id=result.get('model_id'),
                updated_parameters=result.get('updated_parameters', [])
            )
            
            return jsonify(response.to_dict()), 200
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 400
            
        except NotFoundError as e:
            logger.error(f"Not found error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 404
            
        except ProcessingError as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            return jsonify({"error": "서버 내부 오류가 발생했습니다.", "status": "error"}), 500

class IncidentController:
    """장애 기록 관련 요청을 처리하는 컨트롤러"""
    
    def __init__(self, incident_service):
        """
        Args:
            incident_service: 장애 기록 서비스 객체
        """
        self.incident_service = incident_service
        
    @validate_json('severity', 'component', 'description', 'error_details')
    def report_incident(self):
        """장애 기록 엔드포인트
        
        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            incident_request = IncidentReportRequestDTO.from_dict(data)
            
            # 서비스 호출
            result = self.incident_service.report_incident(
                incident_request.severity,
                incident_request.component,
                incident_request.description,
                incident_request.error_details
            )
            
            # 응답 생성
            response = IncidentReportResponseDTO(
                incident_id=result.get('incident_id'),
                timestamp=result.get('timestamp'),
                status=result.get('status')
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