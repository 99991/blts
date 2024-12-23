FROM ubuntu:24.04

# Install without asking questions
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
ENV TZ="UTC"

RUN apt-get update && \
    apt-get -y install \
        python3 \
        python3-pip \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' testuser

# Switch to testuser
USER testuser

# Create and activate a virtual environment, then install Python packages
RUN python3 -m venv /home/testuser/testenv && \
    /home/testuser/testenv/bin/pip install --no-cache-dir \
        numpy scipy scikit-learn pillow networkx pandas pycryptodome z3 && \
    rm -rf /home/testuser/testenv/share/python-wheels && \
    echo 'export PATH="/home/testuser/testenv/bin:$PATH"' >> ~/.bashrc && \
    echo 'alias p="python3"' >> ~/.bashrc

# Set the working directory
WORKDIR /home/testuser
