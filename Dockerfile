FROM runpod/pytorch:3.10-2.0.1-120-devel
# Create a working directory

RUN pip install -U poetry

COPY poetry.lock .
COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
     && poetry install --no-interaction --no-ansi

RUN rm poetry.lock
RUN rm pyproject.toml

RUN curl -fsSL https://starship.rs/install.sh | sh -s -- -y
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc


# EXPOSE 8888 MAYBE RELEVANT

CMD ["bash", "-c", "sleep infinity"]