# 추론 컨트롤러
"""
추천 관련 컨트롤러
"""
import logging
from flask import request, jsonify
from src.api.dto.recommendation_dto import (
    RecommendationRequestDTO,
    RecommendationResponseDTO,
    CityRecommendationResponseDTO
)
from src.api.utils.error_handler import ValidationError, ProcessingError
from src.api.utils.validators import validate_json

# 로깅 설정
logger = logging.getLogger(__name__)

class RecommendationController:
    """추천 관련 요청을 처리하는 컨트롤러"""
    
    def __init__(self, recommendation_service):
        """
        Args:
            recommendation_service: 추천 서비스 객체
        """
        self.recommendation_service = recommendation_service
        
    def get_recommendations(self):
        """여행지 추천 엔드포인트
        
        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱 (GET 요청인 경우 query params, POST 요청인 경우 body)
            if request.method == 'GET':
                data = {
                    'user_id': request.args.get('user_id'),
                    'top_n': request.args.get('top_n', 5),
                    'recommendation_type': request.args.get('recommendation_type', 'both'),
                    'alpha': request.args.get('alpha', 0.5)
                }
                
                # user_profile은 GET에서 처리하지 않음 (복잡해질 수 있음)
            else:  # POST
                data = request.get_json()
                
            rec_request = RecommendationRequestDTO.from_dict(data)
            
            # 서비스 호출
            result = self.recommendation_service.get_recommendations(
                user_id=rec_request.user_id,
                user_profile=rec_request.user_profile,
                top_n=rec_request.top_n,
                recommendation_type=rec_request.recommendation_type,
                alpha=rec_request.alpha
            )
            
            # 응답 생성
            response = RecommendationResponseDTO.from_dict(result)
            
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
    
    def get_city_recommendations(self):
        """도시 추천 엔드포인트
        
        Returns:
            JSON 응답
        """
        try:
            # 요청 데이터 파싱 (GET 요청인 경우 query params, POST 요청인 경우 body)
            if request.method == 'GET':
                data = {
                    'user_id': request.args.get('user_id'),
                    'top_n': request.args.get('top_n', 5),
                    'recommendation_type': 'city',
                    'alpha': request.args.get('alpha', 0.5)
                }
                
                # user_profile은 GET에서 처리하지 않음 (복잡해질 수 있음)
            else:  # POST
                data = request.get_json()
                data['recommendation_type'] = 'city'
                
            rec_request = RecommendationRequestDTO.from_dict(data)
            
            # 서비스 호출
            result = self.recommendation_service.get_city_recommendations(
                user_id=rec_request.user_id,
                user_profile=rec_request.user_profile,
                top_n=rec_request.top_n,
                alpha=rec_request.alpha
            )
            
            # 응답 생성
            response = CityRecommendationResponseDTO.from_dict(result)
            
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