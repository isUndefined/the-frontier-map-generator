FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY prepare_region_grid.py stitch_region_grid.py build_grid_from_jobs.py \
     terrain_alignment.py tile_grid.py ./
COPY tests/ ./tests/
ENTRYPOINT ["python"]
CMD ["prepare_region_grid.py", "--help"]
