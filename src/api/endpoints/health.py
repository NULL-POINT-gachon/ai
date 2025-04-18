"""
서버 상태 확인 API 엔드포인트 정의
"""
from flask_restx import Namespace, Resource
import platform
from datetime import datetime

# 네임스페이스 생성
health_ns = Namespace('health', description='서버 상태 확인 API')

def register_endpoints(api):
    """
    상태 확인 관련 엔드포인트 등록
    
    Args:
        api: flask_restx API 객체
    """
    api.add_namespace(health_ns)
    
    @health_ns.route('')
    class HealthResource(Resource):
        """상태 확인 리소스"""
        
        @health_ns.doc(description='서버 상태 확인')
        def get(self):
            """
            서버 상태 확인 엔드포인트
            
            Returns:
                상태 정보 JSON 응답
            """
            health_info = {
                'status': 'up',
                'timestamp': datetime.now().isoformat(),
                'server_info': {
                    'platform': platform.platform(),
                    'python_version': platform.python_version()
                }
            }
            return health_info
    
    return api