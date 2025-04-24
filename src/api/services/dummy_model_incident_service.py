"""
모델 관리 및 장애 기록 관련 더미 서비스 구현
"""

import uuid
import random
from datetime import datetime, timedelta
import math


class DummyModelService:
    """모델 관리 관련 더미 서비스 클래스"""

    def __init__(self):
        """더미 모델 서비스 초기화"""
        self.models = self._generate_dummy_models(15)  # 더미 모델 15개 생성

    def _generate_dummy_models(self, count):
        """
        더미 모델 데이터 생성

        Args:
            count: 생성할 모델 수

        Returns:
            더미 모델 딕셔너리
        """
        models = {}
        model_types = [
            "neural_collaborative_filtering",
            "content_based",
            "feature_based",
            "hybrid",
        ]
        status_options = ["trained", "training", "failed"]

        for i in range(count):
            model_id = f"model_{str(uuid.uuid4())[:8]}"
            model_type = random.choice(model_types)

            # 생성 시간 (최근 30일 내)
            days_ago = random.randint(0, 30)
            created_at = (datetime.now() - timedelta(days=days_ago)).isoformat()

            # 랜덤 메트릭
            metrics = {
                "rmse": round(random.uniform(0.2, 0.5), 3),
                "mae": round(random.uniform(0.15, 0.4), 3),
                "hit_ratio@10": round(random.uniform(0.5, 0.8), 2),
                "ndcg@10": round(random.uniform(0.6, 0.85), 2),
            }

            models[model_id] = {
                "model_id": model_id,
                "model_name": f"TravelRec_{model_type.split('_')[0].title()}_v{i + 1}",
                "model_type": model_type,
                "created_at": created_at,
                "status": random.choice(status_options) if days_ago > 0 else "training",
                "metrics": metrics if days_ago > 0 else {},
                "is_deployed": random.choice([True, False]) if days_ago > 0 else False,
            }

        return models

    def list_models(self, status=None, model_type=None, page=1, limit=10):
        """
        모델 목록 조회 더미 구현

        Args:
            status: 모델 상태 필터
            model_type: 모델 유형 필터
            page: 페이지 번호
            limit: 페이지당 결과 수

        Returns:
            모델 목록 정보
        """
        # 필터링된 모델 리스트
        filtered_models = list(self.models.values())

        # 상태 필터 적용
        if status:
            filtered_models = [m for m in filtered_models if m["status"] == status]

        # 모델 유형 필터 적용
        if model_type:
            filtered_models = [
                m for m in filtered_models if m["model_type"] == model_type
            ]

        # 총 모델 수
        total = len(filtered_models)

        # 페이지네이션
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total)

        # 페이지에 해당하는 모델 추출
        paged_models = filtered_models[start_idx:end_idx]

        # 더미 응답 반환
        return {"total": total, "page": page, "limit": limit, "models": paged_models}

    def update_model_parameters(self, model_id, parameters):
        """
        모델 파라미터 업데이트 더미 구현

        Args:
            model_id: 모델 ID
            parameters: 수정할 파라미터

        Returns:
            파라미터 업데이트 결과 정보
        """
        # 모델이 존재하는지 확인
        if model_id not in self.models:
            return {"status": "error", "error": f"Model with ID {model_id} not found"}

        # 모델에 파라미터 필드가 없으면 추가
        if "parameters" not in self.models[model_id]:
            self.models[model_id]["parameters"] = {}

        # 업데이트된 파라미터 목록
        updated_parameters = []

        # 파라미터 업데이트
        for param_name, param_value in parameters.items():
            self.models[model_id]["parameters"][param_name] = param_value
            updated_parameters.append(param_name)

        # 더미 응답 반환
        return {
            "status": "success",
            "model_id": model_id,
            "updated_parameters": updated_parameters,
        }


class DummyIncidentService:
    """장애 기록 관련 더미 서비스 클래스"""

    def __init__(self):
        """더미 장애 서비스 초기화"""
        self.incidents = {}

    def report_incident(self, severity, component, description, error_details):
        """
        장애 기록 더미 구현

        Args:
            severity: 심각도
            component: 장애 발생 컴포넌트
            description: 장애 설명
            error_details: 오류 세부 정보

        Returns:
            장애 기록 결과 정보
        """
        # 더미 데이터 생성
        incident_id = f"inc_{str(uuid.uuid4())[:8]}"

        # 현재 시간
        now = datetime.now()

        # 더미 장애 정보 저장
        self.incidents[incident_id] = {
            "incident_id": incident_id,
            "severity": severity,
            "component": component,
            "description": description,
            "error_details": error_details,
            "status": "active",
            "timestamp": now.isoformat(),
            "resolved_at": None,
        }

        # 더미 응답 반환
        return {
            "incident_id": incident_id,
            "timestamp": now.isoformat(),
            "status": "active",
        }
