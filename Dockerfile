FROM runpod/pytorch:3.10-2.0.1-120-devel
# Create a working directory

RUN pip install -U poetry

COPY poetry.lock .
COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
     && poetry install --no-interaction --no-ansi

RUN pip install torch

RUN rm poetry.lock
RUN rm pyproject.toml

RUN curl -fsSL https://starship.rs/install.sh | sh -s -- -y
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

RUN type -p curl >/dev/null || (apt update && apt install curl -y) && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \
    apt install gh -y

RUN gh auth login

RUN gh repo clone

# EXPOSE 8888 MAYBE RELEVANT

CMD ["bash", "-c", "sleep infinity"]