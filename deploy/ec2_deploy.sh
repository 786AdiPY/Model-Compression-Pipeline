#!/usr/bin/env bash
# Deploy xgb-churn-serve to EC2 (Amazon Linux 2 / Ubuntu).
# Usage: ./ec2_deploy.sh <EC2_HOST> <KEY_FILE> [DOCKER_IMAGE]
set -euo pipefail

EC2_HOST="${1:?Usage: $0 <host> <key_file> [image]}"
KEY_FILE="${2:?Usage: $0 <host> <key_file> [image]}"
IMAGE="${3:-xgb-churn-serve:latest}"
REMOTE_ARTIFACTS="/opt/xgb-churn/artifacts"
REMOTE_DATA="/opt/xgb-churn/data"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$EC2_HOST"
SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no"

echo "==> Ensuring Docker installed on remote..."
$SSH "which docker || (sudo yum update -y && sudo yum install -y docker && sudo systemctl start docker && sudo usermod -aG docker ec2-user)"

echo "==> Creating artifact/data dirs on remote..."
$SSH "mkdir -p $REMOTE_ARTIFACTS $REMOTE_DATA"

echo "==> Syncing artifacts to remote..."
$SCP -r artifacts/* "ec2-user@$EC2_HOST:$REMOTE_ARTIFACTS/"
$SCP -r data/test.csv "ec2-user@$EC2_HOST:$REMOTE_DATA/"

echo "==> Pulling image on remote..."
$SSH "docker pull $IMAGE || echo 'Pull failed — building locally on remote'"

echo "==> (Re)starting container..."
$SSH "
  docker stop xgb-churn-serve 2>/dev/null || true
  docker rm   xgb-churn-serve 2>/dev/null || true
  docker run -d \
    --name xgb-churn-serve \
    --restart unless-stopped \
    -p 8000:8000 \
    -v $REMOTE_ARTIFACTS:/artifacts:ro \
    -v $REMOTE_DATA:/data:ro \
    $IMAGE
"

echo "==> Waiting for health check..."
sleep 5
STATUS=$($SSH "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health")
if [ "$STATUS" = "200" ]; then
  echo "==> Deploy OK  (EC2: $EC2_HOST:8000)"
else
  echo "==> Health check returned HTTP $STATUS — check docker logs"
  $SSH "docker logs xgb-churn-serve --tail 50"
  exit 1
fi
