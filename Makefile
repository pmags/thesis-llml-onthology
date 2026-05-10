.PHONY: post-create test-tools init sync lint format type-check gitignore freeze dev-tools ensure-ipykernel

# Post-create command: run tool verification, init project, and sync dependencies
post-create: test-tools init sync ensure-ipykernel

# Initialize Python environment with uv and create pyproject.toml if needed
init:
	@echo "Initializing Python environment..."
	@(uv python install ${PYTHON_VERSION} && \
	if [ ! -f pyproject.toml ]; then \
		if [ "${UV_INIT_BARE}" = "true" ]; then \
			uv init --bare --python ${PYTHON_VERSION}; \
		else \
			uv init --python ${PYTHON_VERSION}; \
		fi; \
	fi) > /tmp/init.log 2>&1
	@echo "✓ Initialization complete (log: /tmp/init.log)"

# Ensure ipykernel is available for notebooks after dependency sync
ensure-ipykernel:
	@if [ "${INSTALL_IPYKERNEL}" = "true" ]; then \
		( \
			if uv run python -c "import ipykernel" >/dev/null 2>&1; then \
				echo "ipykernel available."; \
			else \
				echo "ipykernel is missing after dependency sync."; \
				exit 1; \
			fi \
		) > /tmp/jupyter-kernel.log 2>&1; \
		echo "✓ ipykernel check complete (log: /tmp/jupyter-kernel.log)"; \
	else \
		echo "Skipping ipykernel installation (INSTALL_IPYKERNEL != true)"; \
	fi

# Verify installed tools
test-tools:
	@echo "Running tool verification..."
	@bash .devcontainer/test_tools.sh > /tmp/test-tools.log 2>&1
	@echo "✓ Tool verification complete (log: /tmp/test-tools.log)"

# Sync the full project environment with uv
sync:
	@echo "Syncing dependencies..."
	@echo "Streaming verbose uv output and saving a copy to /tmp/uv-sync.log"
	@bash -o pipefail -c '\
		: > /tmp/uv-sync.log; \
		heartbeat() { \
			while true; do \
				sleep 30; \
				echo "[uv sync] still running ($$(date +%H:%M:%S))" | tee -a /tmp/uv-sync.log; \
			done; \
		}; \
		heartbeat & heartbeat_pid=$$!; \
		trap "kill $$heartbeat_pid 2>/dev/null" EXIT; \
		uv sync --extra dev --verbose 2>&1 | tee -a /tmp/uv-sync.log \
	'
	@echo "✓ Dependency sync complete (log: /tmp/uv-sync.log)"

# Run ruff linter
lint:
	@uv run ruff check .

# Run ruff formatter
format:
	@uv run ruff format .

# Run mypy type checker
type-check:
	@uv run mypy .

# Download Python .gitignore from GitHub
gitignore:
	@if [ -f .gitignore ]; then \
		echo "⚠️  .gitignore already exists, skipping"; \
	else \
		( \
			echo "📥 Downloading Python .gitignore from GitHub..."; \
			curl -fsSL https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore -o .gitignore; \
			echo "✅ .gitignore created"; \
		) > /tmp/gitignore.log 2>&1; \
		echo "✓ .gitignore download complete (log: /tmp/gitignore.log)"; \
	fi

# Verify dev tools managed by pyproject.toml
dev-tools:
	@echo "Checking dev tools..."
	@(uv run ruff --version && uv run mypy --version) > /tmp/dev-tools.log 2>&1
	@echo "✓ Dev tools available (log: /tmp/dev-tools.log)"

# Freeze dependencies to tmp folder
freeze:
	@echo "Freezing dependencies..."
	@echo "# Generated on $$(date)" > /tmp/requirements.txt
	@uv pip freeze >> /tmp/requirements.txt
	@echo "✓ Dependencies frozen (log: /tmp/requirements.txt)"
