# =====================================================
# 🔧 Project Variables (필요하면 수정)
# =====================================================
SRC_DIR    ?= src
TEST_DIR   ?= tests
FLASK_APP  ?= src.api.app        # src/api/app.py → app = Flask(__name__)
FLASK_ENV  ?= development
PORT       ?= 5000

# =====================================================
# 💻 Development Commands
# =====================================================

## 로컬 Flask 개발 서버 실행
run:
	flask --app $(FLASK_APP) --debug run --port $(PORT)

## Ruff 린트 검사
lint:
	ruff check $(SRC_DIR)

## Black + Ruff 포맷
format:
	black $(SRC_DIR)
	ruff format $(SRC_DIR)

## Mypy 타입 검사
typecheck:
	mypy $(SRC_DIR)

## PyTest + 커버리지
test:
	PYTHONPATH=. pytest -v --cov=$(SRC_DIR) -o log_cli_level=INFO $(TEST_DIR)/

## 포맷 ➜ 린트 ➜ 타입체크 일괄 실행
check: format lint typecheck

# =====================================================
# ⚙️  Dev Utilities
# =====================================================

## .env 파일 템플릿 복사
env:
	cp -n .env.example .env

# =====================================================
# 🧹  House-keeping
# =====================================================

clean:
	find $(SRC_DIR) -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage

.PHONY: run lint format typecheck test check env clean
