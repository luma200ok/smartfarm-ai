#!/usr/bin/env bash
# 모델 .pt 를 OCI 서버로 전송 + 서비스 재시작.
# CD(deploy.sh)는 git 추적 파일만 반영하고 .pt 는 .gitignore 제외라 안 옮긴다 → 모델은 이 스크립트로 수동 배포.
# 사용: bash deploy/push_models.sh [ssh_host]   (기본 host=oci-arm1)
set -euo pipefail

HOST="${1:-oci-arm1}"
DEST="/opt/smartfarm_ai/models/"

cd "$(dirname "$0")/.."   # 레포 루트로 이동(어디서 실행해도 models/ 경로 일치)

echo "▶ 모델 전송 → ${HOST}:${DEST}"
rsync -avz \
  models/tomato_resnet18.pt \
  models/tomato_yolov8n.pt \
  models/tomato_part.pt \
  "${HOST}:${DEST}"

echo "▶ 서비스 재시작"
ssh "${HOST}" 'sudo systemctl restart smartfarm-ai'

echo "✅ 모델 전송 + 재시작 완료"
