#!/usr/bin/env bash
set -euo pipefail

# Ensure we're in repo root
cd "$(dirname "$0")/.."

echo "==> Checking JSON schema files are valid JSON"
# Find schema files and jq-validate them
if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found, please install jq (or ensure CI image has it)" >&2
  exit 1
fi

SCHEMA_FILES=$(git ls-files "backend/api/schemas/*.json" || true)
if [ -n "$SCHEMA_FILES" ]; then
  for f in $SCHEMA_FILES; do
    echo "Validating JSON: $f"
    jq -e . "$f" >/dev/null
  done
fi

echo "==> Running Python tests (pytest)"
PYTEST_OPTS=${PYTEST_OPTS:-"-q"}
# Force strict mode so schema violations return 400 in tests
export CHECK_API_STRICT_MODE=1
# Limit to schema/validation-related tests to avoid importing heavy modules in full repo tests
pytest $PYTEST_OPTS \
  tests/test_model_deployment_validation.py \
  tests/test_training_execution_validation.py \
  tests/test_pipeline_validation.py \
  tests/test_three_stage_validation.py \
  tests/test_training_metrics_constraints.py \
  tests/test_model_service_constraints.py \
  tests/test_model_deploy_constraints.py \
  tests/test_pipeline_start_constraints.py \
  tests/test_training_create_session_constraints.py \
  tests/test_pipeline_steps_constraints.py \
  tests/test_three_stage_constraints.py \
  tests/test_three_stage_model_name_constraints.py \
  tests/test_pipeline_steps_require_type.py \
  tests/test_metrics_loss_constraints.py \
  tests/test_model_service_ports_unique.py \
  tests/e2e/test_embeddings_e2e_schema.py \
  tests/e2e/test_intelligent_decision_e2e_schema.py \
  tests/e2e/test_intelligent_adaptive_e2e_schema.py

echo "All checks passed."
