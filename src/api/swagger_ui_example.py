"""
flask_restx 애플리케이션 실행 예제
"""

from flask import Flask
from app import create_app


# 여기서는 서비스 객체들을 모의(mock) 구현하여 제공합니다.
# 실제 환경에서는 실제 서비스 구현체를 사용해야 합니다.
class MockTrainService:
    def train_model(self, data):
        return {
            "training_id": "train_12345",
            "status": "queued",
            "estimated_time": "45 minutes",
            "model_id": "model_678",
        }

    def validate_model(self, data):
        return {
            "validation_id": "val_456",
            "model_id": "model_678",
            "status": "completed",
            "metrics": {
                "rmse": 0.342,
                "mae": 0.267,
                "hit_ratio@10": 0.65,
                "ndcg@10": 0.72,
            },
            "validation_plots": [
                "/api/plots/model_678_prediction.png",
                "/api/plots/model_678_loss.png",
            ],
        }


class MockDeploymentService:
    def deploy_model(self, data):
        return {
            "deployment_id": "deploy_789",
            "status": "deploying",
            "model_id": "model_678",
            "environment": "production",
            "endpoint": "/api/recommendation/inference/deploy_789",
            "estimated_completion": "2023-06-15T14:30:45Z",
        }

    def update_deployment_version(self, data):
        return {
            "status": "success",
            "active_deployment_id": "deploy_789",
            "previous_deployment_id": "deploy_456",
        }


class MockModelService:
    def list_models(self, filters):
        return {
            "total": 15,
            "page": filters["page"],
            "limit": filters["limit"],
            "models": [
                {
                    "model_id": "model_678",
                    "model_name": "TravelRec_NCF_v1",
                    "model_type": "neural_collaborative_filtering",
                    "created_at": "2023-06-14T10:30:45Z",
                    "status": "trained",
                    "metrics": {"rmse": 0.342, "mae": 0.267},
                    "is_deployed": True,
                }
            ],
        }

    def update_model_parameters(self, model_id, data):
        return {
            "status": "success",
            "model_id": model_id,
            "updated_parameters": ["embedding_size", "learning_rate"],
        }


class MockIncidentService:
    def report_incident(self, data):
        return {
            "incident_id": "inc_456",
            "timestamp": "2023-06-15T15:30:45Z",
            "status": "active",
        }


class MockRecommendationService:
    def get_recommendations(self, data):
        return {
            "user_id": data.get("user_id", "anonymous"),
            "recommendations": [
                {
                    "item_id": "place_456",
                    "item_name": "제주 성산일출봉",
                    "type": "place",
                    "score": 0.95,
                    "source": "hybrid",
                    "feature_score": 0.97,
                    "content_score": 0.90,
                    "details": {
                        "avg_satisfaction": 4.8,
                        "avg_residence_time": 120,
                        "related_activities": ["자연 감상", "사진 촬영", "트레킹"],
                        "related_transport": ["렌터카", "대중교통"],
                    },
                }
            ],
        }

    def get_city_recommendations(self, data):
        return {
            "user_id": data.get("user_id", "anonymous"),
            "recommendations": [
                {
                    "city_id": "1",
                    "item_name": "제주도",
                    "related_activities": ["자연 감상", "사진 촬영", "트레킹"],
                },
                {
                    "city_id": "0",
                    "item_name": "서울",
                    "related_activities": ["자연 감상", "사진 촬영", "트레킹"],
                },
            ],
        }


if __name__ == "__main__":
    # 서비스 모의 객체 생성
    services = {
        "train_service": MockTrainService(),
        "deployment_service": MockDeploymentService(),
        "model_service": MockModelService(),
        "incident_service": MockIncidentService(),
        "recommendation_service": MockRecommendationService(),
    }

    # 애플리케이션 생성
    app = create_app(services)

    # 애플리케이션 실행
    app.run(debug=True, host="0.0.0.0", port=5000)
