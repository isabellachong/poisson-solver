# Transfer the project to and from Vanda

Run transfer commands in your laptop's Terminal, outside an SSH session.
Change these variables for your account and directories:

```bash
export VANDA_USER="YOUR_USERID"
export VANDA_HOST="vanda.nus.edu.sg"
export LOCAL_PROJECT_DIR="$HOME/Desktop/dsa5208-project3"
export VANDA_PROJECT_DIR="/home/svu/$VANDA_USER/dsa5208-project3"
export RESULTS_DIR="results/performance"
```

## Upload source and documentation

```bash
ls "$LOCAL_PROJECT_DIR/CMakeLists.txt" "$LOCAL_PROJECT_DIR/scripts/benchmark.py"
ssh "$VANDA_USER@$VANDA_HOST" "mkdir -p '$VANDA_PROJECT_DIR'"
rsync -av --progress \
  --exclude='/build*/' --exclude='/.venv*/' --exclude='/results/' \
  --exclude='/outputs/' --exclude='/work/' --exclude='/tmp/' \
  --exclude='/jobs/' --exclude='*.log' --exclude='/*.o[0-9]*' \
  --exclude='/*.e[0-9]*' --exclude='/.git/' \
  --exclude='__pycache__/' --exclude='.DS_Store' \
  "$LOCAL_PROJECT_DIR/" \
  "$VANDA_USER@$VANDA_HOST:$VANDA_PROJECT_DIR/"
```

This copies code, documentation, tests and application/analysis scripts. The
trailing slash copies the directory contents, avoiding a nested project folder.
It excludes local builds, environments, results and scheduler artifacts. Existing
remote files are not deleted. Repeating the command updates matching files.

## Build and run remotely

```bash
ssh "$VANDA_USER@$VANDA_HOST"
```

In the remote shell, set the project directory again (laptop variables are not
automatically exported through SSH):

```bash
export VANDA_PROJECT_DIR="$HOME/dsa5208-project3"
cd "$VANDA_PROJECT_DIR"
ls CMakeLists.txt src executables tests scripts docs
```

Use your chosen remote path if it differs. Obtain suitable CPU/GPU resources
through the cluster's normal allocation procedure. Then follow the direct build,
run and test commands in [the README](../README.md) and the direct measurement
commands in [EXPERIMENTS.md](EXPERIMENTS.md). No project submission scripts are
needed. Build Linux binaries on Vanda rather than copying laptop binaries.

## Download measured results and figures

Back in the laptop Terminal, use the same variables as above. Set `RESULTS_DIR`
to the actual study directory on Vanda; renamed local folders do not rename the
remote data. Download the complete selected study and accuracy outputs:

```bash
mkdir -p "$LOCAL_PROJECT_DIR/$RESULTS_DIR" "$LOCAL_PROJECT_DIR/results/accuracy"
rsync -av --progress \
  "$VANDA_USER@$VANDA_HOST:$VANDA_PROJECT_DIR/$RESULTS_DIR/" \
  "$LOCAL_PROJECT_DIR/$RESULTS_DIR/"
rsync -av --progress \
  "$VANDA_USER@$VANDA_HOST:$VANDA_PROJECT_DIR/results/accuracy/" \
  "$LOCAL_PROJECT_DIR/results/accuracy/"
```

These commands retain remote data and do not delete local files. Keep measurements
for analysis, and include the required figures/findings in the report. Exclude
scheduler artifacts, diagnostic logs and terminal transcripts from the final
source submission. Changing a documentation path does not move an existing study;
rename only after all processes writing to it finish, and avoid overwriting data.
