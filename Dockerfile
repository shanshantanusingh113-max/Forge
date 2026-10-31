# Run Forge on any host that supports Docker (Railway, Fly.io, a VPS, ...).
#   docker build -t forge .
#   docker run -p 8765:8765 -e FORGE_PASSWORD=change-me -e FORGE_PROVIDER=demo forge
FROM python:3.12-slim
WORKDIR /app
COPY . .
ENV FORGE_HOST=0.0.0.0 \
    FORGE_PORT=8765 \
    PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["python", "server.py", "--no-browser"]
