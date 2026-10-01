TAG=${1:-1}
REPO_ROOT=$HOME/git/cosmos-curate
IMAGE=anyscale-cosmos-curate

docker build \
  --ulimit nofile=65536 \
  --progress=auto \
  --network=host \
  -f anyscale-cosmos-curate.Dockerfile \
  -t ${IMAGE}:$TAG \
  -t ${IMAGE}:latest \
  $REPO_ROOT
