# 앱 설정
"""
애플리케이션 설정 모듈
환경변수 및 설정값 관리
"""
import os
from pathlib import Path

# 기본 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 환경 설정
class Config:
    """기본 설정 클래스"""
    # Flask 설정
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    TESTING = os.getenv('TESTING', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # 서버 설정
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 3001))
    
    # 모델 관련 설정
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'data', 'training')
    
    # API 설정
    API_PREFIX = '/api/v1'

class DevelopmentConfig(Config):
    """개발 환경 설정 클래스"""
    DEBUG = True

class TestingConfig(Config):
    """테스트 환경 설정 클래스"""
    DEBUG = True
    TESTING = True

class ProductionConfig(Config):
    """운영 환경 설정 클래스"""
    DEBUG = False
    TESTING = False

# 환경에 따른 설정 선택
config_by_name = {
    'dev': DevelopmentConfig,
    'test': TestingConfig,
    'prod': ProductionConfig
}

# 기본 설정은 개발 환경
active_config = config_by_name[os.getenv('FLASK_ENV', 'dev')]