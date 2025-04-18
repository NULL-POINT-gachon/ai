"""
학습 관련 API 엔드포인트 정의
"""
from flask_restx import Namespace, Resource, fields

# 네임스페이스 생성
train_ns = Namespace('recommendation/training', description='AI 여행지 추천 학습 API')

def register_endpoints(api, train_service):
    """
    학습 관련 엔드포인트 등록
    
    Args:
        api: flask_restx API 객체
        train_service: 학습 서비스 객체
    """
    api.add_namespace(train_ns)
    
    # 모델 정의
    training_data_config = train_ns.model('TrainingDataConfig', {
        'use_validation': fields.Boolean(description='검증 데이터 사용 여부', default=True),
        'validation_split': fields.Float(description='검증 데이터 비율', default=0.2),
        'random_seed': fields.Integer(description='랜덤 시드', default=42)
    })
    
    parameters_model = train_ns.model('Parameters', {
        'embedding_size': fields.Integer(description='임베딩 차원 수', required=True),
        'learning_rate': fields.Float(description='학습률', required=True),
        'batch_size': fields.Integer(description='배치 크기', required=True),
        'epochs': fields.Integer(description='학습 에포크 수', required=True),
        'hidden_layers': fields.List(fields.Integer, description='은닉층 구성')
    })
    
    training_request = train_ns.model('TrainingRequest', {
        'model_name': fields.String(description='모델 이름', required=True),
        'model_type': fields.String(description='모델 유형(ncf, feature_based, content_based, hybrid)', required=True),
        'parameters': fields.Nested(parameters_model, description='모델 학습 파라미터', required=True),
        'training_data_config': fields.Nested(training_data_config, description='학습 데이터 설정')
    })
    
    training_response = train_ns.model('TrainingResponse', {
        'training_id': fields.String(description='학습 ID'),
        'status': fields.String(description='학습 상태(queued, running, completed, failed)'),
        'estimated_time': fields.String(description='예상 완료 시간'),
        'model_id': fields.String(description='생성된 모델 ID')
    })
    
    test_data_config = train_ns.model('TestDataConfig', {
        'test_size': fields.Float(description='테스트 데이터 비율', required=True),
        'metrics': fields.List(fields.String, description='검증할 메트릭 목록', required=True)
    })
    
    validation_request = train_ns.model('ValidationRequest', {
        'model_id': fields.String(description='검증할 모델 ID', required=True),
        'test_data_config': fields.Nested(test_data_config, description='테스트 데이터 설정', required=True)
    })
    
    validation_metrics = train_ns.model('ValidationMetrics', {
        'rmse': fields.Float(description='Root Mean Square Error'),
        'mae': fields.Float(description='Mean Absolute Error'),
        'hit_ratio@10': fields.Float(description='추천 적중률'),
        'ndcg@10': fields.Float(description='Normalized Discounted Cumulative Gain')
    })
    
    validation_response = train_ns.model('ValidationResponse', {
        'validation_id': fields.String(description='검증 ID'),
        'model_id': fields.String(description='모델 ID'),
        'status': fields.String(description='검증 상태'),
        'metrics': fields.Nested(validation_metrics, description='검증 메트릭 결과'),
        'validation_plots': fields.List(fields.String, description='검증 결과 시각화 URL')
    })

    @train_ns.route('')
    class TrainingResource(Resource):
        """모델 학습 리소스"""
        
        @train_ns.doc(description='여행지 추천 모듈 학습 기능')
        @train_ns.expect(training_request)
        @train_ns.marshal_with(training_response)
        def post(self):
            """
            모델 학습 엔드포인트
            
            Returns:
                학습 상태 정보
            """
            return train_service.train_model(train_ns.payload)
    
    @train_ns.route('/validation')
    class ValidationResource(Resource):
        """모델 검증 리소스"""
        
        @train_ns.doc(description='여행지 추천 모듈 검증 기능')
        @train_ns.expect(validation_request)
        @train_ns.marshal_with(validation_response)
        def post(self):
            """
            모델 검증 엔드포인트
            
            Returns:
                검증 결과 정보
            """
            return train_service.validate_model(train_ns.payload)
    
    return api