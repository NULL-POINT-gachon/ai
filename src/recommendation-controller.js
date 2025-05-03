const { spawn } = require('child_process');
const express = require('express');
const router = express.Router();

/**
 * 도시 추천 API
 * 감정 기반 도시 추천을 제공합니다.
 */
router.post('/recommendation/city', async (req, res) => {
  try {
    const {
      trip_duration,       // 여행 기간 (일)
      companions_count,    // 여행 동반자 수
      emotion_ids,         // 감정 ID 배열
      top_n = 3,           // 추천 개수 (기본값: 3)
      recommendation_type = 'both', // 추천 유형 (기본값: both)
      alpha = 0.7          // 하이브리드 추천 가중치 (기본값: 0.7)
    } = req.body;

    // 필수 파라미터 체크
    if (!trip_duration || !companions_count || !emotion_ids) {
      return res.status(400).json({
        success: false,
        message: '필수 파라미터가 누락되었습니다.'
      });
    }

    // Python 스크립트 실행
    const pythonProcess = spawn('python', [
      'ai_recommendation.py',
      '--mode', 'city',
      '--trip_duration', trip_duration.toString(),
      '--companions_count', companions_count.toString(),
      '--emotion_ids', emotion_ids.join(','),
      '--top_n', top_n.toString(),
      '--recommendation_type', recommendation_type,
      '--alpha', alpha.toString()
    ]);

    // 결과 데이터 수집
    let result = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });

    // 프로세스 종료 시 처리
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python 프로세스 오류 (코드: ${code}): ${errorData}`);
        return res.status(500).json({
          success: false,
          message: '도시 추천 처리 중 오류가 발생했습니다.',
          error: errorData
        });
      }

      try {
        // Python 스크립트 출력 결과를 JSON으로 파싱
        const recommendations = JSON.parse(result);
        
        return res.status(200).json(recommendations);
      } catch (error) {
        console.error('JSON 파싱 오류:', error);
        return res.status(500).json({
          success: false,
          message: '결과 파싱 중 오류가 발생했습니다.',
          error: error.message
        });
      }
    });
  } catch (error) {
    console.error('도시 추천 API 오류:', error);
    return res.status(500).json({
      success: false,
      message: '서버 오류가 발생했습니다.',
      error: error.message
    });
  }
});

/**
 * 상세 여행지 추천 API
 * 도시 내 세부 여행지 추천을 제공합니다.
 */
router.post('/recommendations', async (req, res) => {
  try {
    const {
      city,                // 도시명
      activity_type,       // 활동 유형 (실내, 실외)
      activity_ids,        // 활동 ID 배열
      emotion_ids,         // 감정 ID 배열
      preferred_transport, // 선호 교통수단
      companions_count,    // 여행 동반자 수
      activity_level,      // 활동 수준 (1-10)
      top_n = 3,           // 추천 개수 (기본값: 3)
      recommendation_type = 'both', // 추천 유형 (기본값: both)
      alpha = 0.7          // 하이브리드 추천 가중치 (기본값: 0.7)
    } = req.body;

    // 필수 파라미터 체크
    if (!city || !activity_type || !activity_ids || !emotion_ids || !preferred_transport || 
        !companions_count || !activity_level) {
      return res.status(400).json({
        success: false,
        message: '필수 파라미터가 누락되었습니다.'
      });
    }

    // Python 스크립트 실행
    const pythonProcess = spawn('python', [
      'ai_recommendation.py',
      '--mode', 'detail',
      '--city', city,
      '--activity_type', activity_type,
      '--activity_ids', activity_ids.join(','),
      '--emotion_ids', emotion_ids.join(','),
      '--preferred_transport', preferred_transport,
      '--companions_count', companions_count.toString(),
      '--activity_level', activity_level.toString(),
      '--top_n', top_n.toString(),
      '--recommendation_type', recommendation_type,
      '--alpha', alpha.toString()
    ]);

    // 결과 데이터 수집
    let result = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });

    // 프로세스 종료 시 처리
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python 프로세스 오류 (코드: ${code}): ${errorData}`);
        return res.status(500).json({
          success: false,
          message: '여행지 추천 처리 중 오류가 발생했습니다.',
          error: errorData
        });
      }

      try {
        // Python 스크립트 출력 결과를 JSON으로 파싱
        const recommendations = JSON.parse(result);
        
        return res.status(200).json(recommendations);
      } catch (error) {
        console.error('JSON 파싱 오류:', error);
        return res.status(500).json({
          success: false,
          message: '결과 파싱 중 오류가 발생했습니다.',
          error: error.message
        });
      }
    });
  } catch (error) {
    console.error('여행지 추천 API 오류:', error);
    return res.status(500).json({
      success: false,
      message: '서버 오류가 발생했습니다.',
      error: error.message
    });
  }
});

module.exports = router;