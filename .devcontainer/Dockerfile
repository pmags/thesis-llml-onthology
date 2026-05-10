FROM mcr.microsoft.com/devcontainers/base:ubuntu

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		graphviz \
		unixodbc \
		unixodbc-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
