"""
학습 관련 더미 서비스 구현
"""

import time
import uuid
from datetime import datetime, timedelta


class DummyTrainService:
    """학습 관련 더미 서비스 클래스"""

    def __init__(self):
        """더미 학습 서비스 초기화"""
        self.training_jobs = {}
        self.models = {}
        self.validations = {}

    def train_model(
        self, model_name, model_type, parameters, training_data_config=None
    ):
        """
        모델 학습 더미 구현

        Args:
            model_name: 모델 이름
            model_type: 모델 유형
            parameters: 학습 파라미터
            training_data_config: 학습 데이터 설정

        Returns:
            학습 결과 정보
        """
        # 더미 데이터 생성
        training_id = f"train_{str(uuid.uuid4())[:8]}"
        model_id = f"model_{str(uuid.uuid4())[:8]}"

        # 현재 시간
        now = datetime.now()

        # 더미 학습 작업 정보 저장
        self.training_jobs[training_id] = {
            "training_id": training_id,
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "parameters": parameters,
            "training_data_config": training_data_config,
            "status": "queued",
            "created_at": now.isoformat(),
            "estimated_completion": (now + timedelta(minutes=45)).isoformat(),
        }

        # 더미 모델 정보 저장
        self.models[model_id] = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "parameters": parameters,
            "status": "training",
            "created_at": now.isoformat(),
            "metrics": {},
        }

        # 더미 응답 반환
        return {
            "training_id": training_id,
            "status": "queued",
            "estimated_time": "45 minutes",
            "model_id": model_id,
        }

    def validate_model(self, model_id, test_data_config):
        """
        모델 검증 더미 구현

        Args:
            model_id: 모델 ID
            test_data_config: 테스트 데이터 설정

        Returns:
            검증 결과 정보
        """
        # 더미 데이터 생성
        validation_id = f"val_{str(uuid.uuid4())[:8]}"

        # 검증 메트릭 생성
        metrics = {"rmse": 0.342, "mae": 0.267, "hit_ratio@10": 0.65, "ndcg@10": 0.72}

        # 검증 플롯 경로 생성
        validation_plots = [
            f"/api/plots/{model_id}_prediction.png",
            f"/api/plots/{model_id}_loss.png",
        ]

        # 더미 검증 정보 저장
        self.validations[validation_id] = {
            "validation_id": validation_id,
            "model_id": model_id,
            "test_data_config": test_data_config,
            "status": "completed",
            "metrics": metrics,
            "validation_plots": validation_plots,
            "created_at": datetime.now().isoformat(),
        }

        # 모델이 존재하면 메트릭 업데이트
        if model_id in self.models:
            self.models[model_id]["metrics"] = metrics
            self.models[model_id]["status"] = "trained"

        # 더미 응답 반환
        return {
            "validation_id": validation_id,
            "model_id": model_id,
            "status": "completed",
            "metrics": metrics,
            "validation_plots": validation_plots,
        }
