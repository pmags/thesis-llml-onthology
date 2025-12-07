# Thesis LLML Ontology

This repository contains a development environment for working with Large Language Models (LLM) and ontology-related projects using Python.

## Development Environment

This project uses a **Dev Container** with **Miniconda** to ensure a consistent development environment across different machines.

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your machine
- [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting Started with Dev Container

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd thesis-llml-onthology
   ```

2. **Open in VS Code:**
   ```bash
   code .
   ```

3. **Reopen in Container:**
   - When prompted, click "Reopen in Container"
   - Or use the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and select "Dev Containers: Reopen in Container"

4. **Wait for the container to build:**
   - The first build may take several minutes as it downloads the base image and installs all dependencies
   - Subsequent builds will be faster due to Docker's caching

### Dev Container Features

The development container includes:

- **Base Image:** Microsoft's official Miniconda Dev Container (Debian-based)
- **Python Environment:** Miniconda with Python 3
- **Pre-installed VS Code Extensions:**
  - Jupyter notebooks support
  - Python language support (Pylance, Pylint)
  - GitHub Copilot
  - Semantic Kernel
  - Markdown and Mermaid diagram support
  - Rainbow CSV for data files
  - Auto-docstring generator
  - And more...

- **Python Formatting:** Black formatter with format-on-save enabled
- **Git LFS:** Pre-configured for large file support

## Conda Environment

The project uses a Conda environment defined in `environment.yml` with the following key packages:

### Core Libraries

- **Data Science:** pandas, numpy, scipy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Development:** ipython, ipykernel, ipywidgets, pytest
- **Database:** pyodbc
- **Configuration:** python-dotenv, configparser
- **Semantic Web:** rdflib (for ontology work)
- **AI/ML:** semantic-kernel (Microsoft's Semantic Kernel)

### Conda Channels

The environment uses the following channels in priority order:
1. `pytorch` - For PyTorch packages
2. `nvidia` - For NVIDIA/CUDA related packages
3. `conda-forge` - Community-maintained packages

### Managing the Conda Environment

#### Inside the Dev Container

The environment is automatically created when the container is built. The base conda environment is updated with the packages from `environment.yml`.

To activate the environment manually:
```bash
conda activate thesis_env
```

#### Outside the Dev Container (Local Development)

If you prefer to work outside the container, you can create the environment locally:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate thesis_env

# Install Jupyter kernel (for notebook support)
python -m ipykernel install --user --name thesis_env
```

#### Updating the Environment

If you add new dependencies to `environment.yml`:

1. **Inside the Dev Container:** Rebuild the container
   - Command Palette → "Dev Containers: Rebuild Container"

2. **Local Development:**
   ```bash
   conda env update -f environment.yml
   ```

#### Exporting the Current Environment

To export your current environment configuration:
```bash
conda env export > environment.yml
```

## Project Structure

```
thesis-llml-onthology/
├── .devcontainer/          # Dev container configuration
│   ├── devcontainer.json   # Container settings and VS Code extensions
│   ├── Dockerfile          # Container image definition
│   └── noop.txt           # Placeholder file for build process
├── environment.yml         # Conda environment specification
└── README.md              # This file
```

## Working with Jupyter Notebooks

The environment comes pre-configured for Jupyter notebook development:

1. Create a new `.ipynb` file in VS Code
2. Select the kernel: `thesis_env` or `base` (both have the same packages in this setup)
3. Start coding!

## Troubleshooting

### Container fails to build

- Ensure Docker is running
- Check that you have sufficient disk space
- Try rebuilding without cache: Command Palette → "Dev Containers: Rebuild Container Without Cache"

### Python packages missing

- Verify the package is listed in `environment.yml`
- Rebuild the container to install new packages

### Conda environment not activated

```bash
conda activate thesis_env
```

## Additional Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Conda Documentation](https://docs.conda.io/)
- [Miniconda Documentation](https://docs.conda.io/en/latest/miniconda.html)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
