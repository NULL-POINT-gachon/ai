# 입력 검증
"""
입력 데이터 검증 유틸리티
API 요청 데이터 검증 로직
"""

from functools import wraps
from flask import request, jsonify
from src.api.utils.error_handler import ValidationError


def validate_json(*required_fields):
    """
    JSON 요청 데이터 검증 데코레이터

    Args:
        *required_fields: 필수 필드 목록

    Returns:
        데코레이터 함수
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # JSON 요청 검증
            if not request.is_json:
                raise ValidationError(
                    "Content-Type은 반드시 application/json이어야 합니다."
                )

            data = request.get_json()
            if data is None:
                raise ValidationError("유효한 JSON 데이터가 아닙니다.")

            # 필수 필드 검증
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(
                    f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}"
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def validate_train_request(data):
    """
    학습 요청 데이터 검증

    Args:
        data: 검증할 요청 데이터

    Returns:
        None (유효하지 않을 경우 예외 발생)
    """
    # 모델 타입 검증
    if "model_type" not in data:
        raise ValidationError("model_type은 필수 필드입니다.")

    # 학습 파라미터 검증
    if "parameters" not in data or not isinstance(data["parameters"], dict):
        raise ValidationError("parameters는 필수 필드이며 딕셔너리 형태여야 합니다.")

    # 학습 데이터 검증 (생략 - 실제로는 데이터 형식 검증 로직 필요)
    if "training_data" not in data:
        raise ValidationError("training_data는 필수 필드입니다.")


def validate_predict_request(data):
    """
    추론 요청 데이터 검증

    Args:
        data: 검증할 요청 데이터

    Returns:
        None (유효하지 않을 경우 예외 발생)
    """
    # 모델 ID 검증
    if "model_id" not in data:
        raise ValidationError("model_id는 필수 필드입니다.")

    # 입력 데이터 검증
    if "input_data" not in data:
        raise ValidationError("input_data는 필수 필드입니다.")
