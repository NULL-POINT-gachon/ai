"""
배포 관련 컨트롤러
"""

import logging
from flask import request, jsonify
from src.api.dto.deployment_dto import (
    DeploymentRequestDTO,
    DeploymentResponseDTO,
    DeploymentVersionRequestDTO,
    DeploymentVersionResponseDTO,
)
from src.api.utils.error_handler import ValidationError, ProcessingError
from src.api.utils.validators import validate_json

# 로깅 설정
logger = logging.getLogger(__name__)


class DeploymentController:
    """배포 관련 요청을 처리하는 컨트롤러"""

    def __init__(self, deployment_service):
        """
        Args:
            deployment_service: 배포 서비스 객체
        """
        self.deployment_service = deployment_service

    @validate_json("model_id", "environment", "deployment_config")
    def deploy_model(self):
        """모델 배포 엔드포인트

        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            deploy_request = DeploymentRequestDTO.from_dict(data)

            # 서비스 호출
            result = self.deployment_service.deploy_model(
                deploy_request.model_id,
                deploy_request.environment,
                deploy_request.deployment_config,
            )

            # 응답 생성
            response = DeploymentResponseDTO(
                deployment_id=result.get("deployment_id"),
                status=result.get("status"),
                model_id=result.get("model_id"),
                environment=result.get("environment"),
                endpoint=result.get("endpoint"),
                estimated_completion=result.get("estimated_completion"),
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
            return (
                jsonify({"error": "서버 내부 오류가 발생했습니다.", "status": "error"}),
                500,
            )

    @validate_json("deployment_id", "active")
    def update_deployment_version(self):
        """모델 배포 버전 선택 엔드포인트

        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            version_request = DeploymentVersionRequestDTO.from_dict(data)

            # 서비스 호출
            result = self.deployment_service.update_deployment_version(
                version_request.deployment_id, version_request.active
            )

            # 응답 생성
            response = DeploymentVersionResponseDTO(
                status=result.get("status"),
                active_deployment_id=result.get("active_deployment_id"),
                previous_deployment_id=result.get("previous_deployment_id"),
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
            return (
                jsonify({"error": "서버 내부 오류가 발생했습니다.", "status": "error"}),
                500,
            )
