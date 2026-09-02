# syntax=docker/dockerfile:1

FROM node:22-alpine AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY crossword ./crossword
COPY corpus ./corpus
COPY --from=web /src/web/dist ./web/dist
RUN pip install --no-cache-dir -e '.[web]'
EXPOSE 8000
CMD ["uvicorn", "crossword.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
