# drop cuml from default envs built
CWD=$(pwd)
REPO_ROOT=$HOME/git/cosmos-curate
cd $REPO_ROOT
cosmos-curate image build \
  --curator-path "${REPO_ROOT}" \
  --image-name cosmos-curate \
  --image-tag 1 \
  --dry-run \
  --envs legacy-transformers,transformers,unified \
  --dockerfile-output-path "${CWD}/cosmos-curate.Dockerfile"
