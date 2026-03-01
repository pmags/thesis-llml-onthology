FROM mcr.microsoft.com/devcontainers/miniconda:0-3

ARG AZURE_DEVOPS_PAT
ENV AZURE_DEVOPS_PAT=$AZURE_DEVOPS_PAT

ARG DBT_ENV_SECRET_AZURE_DEVOPS_PAT
ENV DBT_ENV_SECRET_AZURE_DEVOPS_PAT=$DBT_ENV_SECRET_AZURE_DEVOPS_PAT

RUN echo "Debug - DBT_ENV_SECRET_AZURE_DEVOPS_PAT value during build: $DBT_ENV_SECRET_AZURE_DEVOPS_PAT"

RUN rm -f /etc/apt/sources.list.d/yarn.list

# Copy the environment.yml file and other required files
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
    

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*