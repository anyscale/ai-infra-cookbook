TAG=1
REPO_ROOT=$HOME/git/cosmos-curate
docker build \
  --ulimit nofile=65536 \
  --network=host \
  -f cosmos-curate.Dockerfile \
  -t cosmos-curate:$TAG \
  -t cosmos-curate:latest \
  $REPO_ROOT
