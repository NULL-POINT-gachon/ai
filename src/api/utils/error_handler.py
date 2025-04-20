# 에러 핸들링
"""
에러 처리 유틸리티
API 요청 처리 중 발생하는 예외 처리 로직
"""
from flask import jsonify

class APIError(Exception):
    """API 에러 기본 클래스"""
    status_code = 500
    message = "서버 내부 오류가 발생했습니다."

    def __init__(self, message=None, status_code=None, payload=None):
        Exception.__init__(self)
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['error'] = self.message
        rv['status'] = 'error'
        return rv

class BadRequestError(APIError):
    """잘못된 요청 오류"""
    status_code = 400
    message = "잘못된 요청입니다."

class NotFoundError(APIError):
    """리소스 찾을 수 없음 오류"""
    status_code = 404
    message = "요청한 리소스를 찾을 수 없습니다."

class ValidationError(BadRequestError):
    """데이터 유효성 검증 오류"""
    message = "입력 데이터가 유효하지 않습니다."

class ProcessingError(APIError):
    """처리 중 오류"""
    status_code = 500
    message = "데이터 처리 중 오류가 발생했습니다."

class ModelNotFoundError(NotFoundError):
    """모델을 찾을 수 없음 오류"""
    message = "요청한 모델을 찾을 수 없습니다."

def register_error_handlers(app):
    """Flask 앱에 에러 핸들러 등록"""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(404)
    def handle_not_found(error):
        response = jsonify({
            'status': 'error',
            'error': '요청한 리소스를 찾을 수 없습니다.'
        })
        response.status_code = 404
        return response
    
    @app.errorhandler(500)
    def handle_server_error(error):
        response = jsonify({
            'status': 'error',
            'error': '서버 내부 오류가 발생했습니다.'
        })
        response.status_code = 500
        return response
        
    return app