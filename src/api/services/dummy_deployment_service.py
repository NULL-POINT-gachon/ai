"""
배포 관련 더미 서비스 구현
"""
import uuid
from datetime import datetime, timedelta

class DummyDeploymentService:
    """배포 관련 더미 서비스 클래스"""
    
    def __init__(self):
        """더미 배포 서비스 초기화"""
        self.deployments = {}
        self.active_deployments = {}
    
    def deploy_model(self, model_id, environment, deployment_config):
        """
        모델 배포 더미 구현
        
        Args:
            model_id: 모델 ID
            environment: 배포 환경
            deployment_config: 배포 설정
            
        Returns:
            배포 결과 정보
        """
        # 더미 데이터 생성
        deployment_id = f"deploy_{str(uuid.uuid4())[:8]}"
        
        # 현재 시간
        now = datetime.now()
        
        # 엔드포인트 생성
        endpoint = f"/api/recommendation/inference/{deployment_id}"
        
        # 더미 배포 정보 저장
        self.deployments[deployment_id] = {
            'deployment_id': deployment_id,
            'model_id': model_id,
            'environment': environment,
            'deployment_config': deployment_config,
            'status': 'deploying',
            'created_at': now.isoformat(),
            'estimated_completion': (now + timedelta(minutes=15)).isoformat(),
            'endpoint': endpoint
        }
        
        # 더미 응답 반환
        return {
            'deployment_id': deployment_id,
            'status': 'deploying',
            'model_id': model_id,
            'environment': environment,
            'endpoint': endpoint,
            'estimated_completion': (now + timedelta(minutes=15)).isoformat()
        }
    
    def update_deployment_version(self, deployment_id, active):
        """
        배포 버전 업데이트 더미 구현
        
        Args:
            deployment_id: 배포 ID
            active: 활성화 여부
            
        Returns:
            버전 업데이트 결과 정보
        """
        # 이전 활성 배포 ID 저장
        previous_deployment_id = None
        if deployment_id in self.deployments:
            environment = self.deployments[deployment_id]['environment']
            if environment in self.active_deployments:
                previous_deployment_id = self.active_deployments[environment]
        
            # 활성화 상태 업데이트
            if active:
                self.active_deployments[environment] = deployment_id
                self.deployments[deployment_id]['status'] = 'active'
                
                # 이전 배포가 있고, 현재와 다른 경우 상태 업데이트
                if previous_deployment_id and previous_deployment_id != deployment_id:
                    if previous_deployment_id in self.deployments:
                        self.deployments[previous_deployment_id]['status'] = 'inactive'
            else:
                if environment in self.active_deployments and self.active_deployments[environment] == deployment_id:
                    del self.active_deployments[environment]
                
                self.deployments[deployment_id]['status'] = 'inactive'
        
        # 더미 응답 반환
        return {
            'status': 'success',
            'active_deployment_id': deployment_id if active else None,
            'previous_deployment_id': previous_deployment_id
        }
    