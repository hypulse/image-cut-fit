#!/bin/zsh
set -euo pipefail

SCRIPT_PATH="${0:A}"
PROJECT_DIR="${SCRIPT_PATH:h}"
cd "$PROJECT_DIR"

SYSTEM_PYTHON="${PYTHON_BIN:-python3}"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
STREAMLIT_BIN="$PROJECT_DIR/.venv/bin/streamlit"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating virtual environment..."
  "$SYSTEM_PYTHON" -m venv "$PROJECT_DIR/.venv"
fi

if [[ ! -x "$STREAMLIT_BIN" ]]; then
  echo "Installing dependencies..."
  "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
fi

PORT="$(
  "$VENV_PYTHON" -c 'import os, socket
preferred = int(os.environ.get("IMAGE_CUT_FIT_PORT", "8501"))
candidates = [preferred] + [port for port in range(8501, 8521) if port != preferred]
for port in candidates:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            continue
    print(port)
    break
else:
    raise SystemExit("No available port found from 8501 to 8520.")'
)"
URL="http://localhost:$PORT"

echo "Starting Image Cut Fit..."
echo "URL: $URL"
echo "Close this Terminal window or press Ctrl+C to stop the app."

if [[ "${IMAGE_CUT_FIT_NO_OPEN:-0}" != "1" ]]; then
  open "$URL" >/dev/null 2>&1 &
fi

exec "$STREAMLIT_BIN" run "$PROJECT_DIR/app.py" \
  --server.port "$PORT" \
  --server.headless true
