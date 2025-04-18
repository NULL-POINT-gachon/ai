#!/usr/bin/env python
"""
AI 학습 서버 실행 스크립트
"""
import os
import sys

# src 디렉토리의 상위 디렉토리를 모듈 경로에 추가
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 먼저 flask_restx가 설치되어 있는지 확인
try:
    import flask_restx
    print(f"flask_restx version: {flask_restx.__version__}")
except ImportError:
    print("Error: flask_restx is not installed. Please install it using:")
    print("pip install flask-restx")
    sys.exit(1)

# 일반적인 import 방식 시도
try:
    from src.api.app import create_app, active_config
    
    # 앱 실행
    if __name__ == '__main__':
        app = create_app()
        app.run(
            host=active_config.HOST, 
            port=active_config.PORT, 
            debug=active_config.DEBUG
        )
except ImportError as e:
    print(f"ImportError: {e}")
    print("Trying dynamic module loading instead...")
    
    # 모듈 로드 함수 정의
    import importlib.util
    def load_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # 이 줄 추가
        spec.loader.exec_module(module)
        return module

    # app.py 파일 경로
    app_path = os.path.join(project_root, 'src', 'api', 'app.py')
    
    # 모듈 동적 로드
    app_module = load_module('app_module', app_path)
    
    # 앱 실행
    if __name__ == '__main__':
        app = app_module.create_app()
        app.run(
            host=app_module.active_config.HOST, 
            port=app_module.active_config.PORT, 
            debug=app_module.active_config.DEBUG
        )