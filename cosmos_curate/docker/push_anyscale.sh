TAG=${1:-1}
REGISTRY=${2:-aws}
IMAGE=anyscale-cosmos-curate
SRC=${IMAGE}:${TAG}

if [ "$REGISTRY" = "aws" ]; then
  AWS_ACCOUNT=367974485317
  AWS_REGION=us-west-2
  AWS_REPO=wagner-west-2
  DST_BASE=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}
  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
else
  PROJECT_ID=troubleshootingorg-gcp-pub
  REGION=us-central1
  REPO=wagner-docker
  DST_BASE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}
fi

docker tag ${SRC} ${DST_BASE}:${TAG}
docker push ${DST_BASE}:${TAG}
docker tag ${SRC} ${DST_BASE}:latest
docker push ${DST_BASE}:latest
