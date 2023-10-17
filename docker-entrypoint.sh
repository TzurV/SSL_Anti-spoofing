#!/bin/bash
gunicorn "app.main:app" --workers=2 --worker-class uvicorn.workers.UvicornWorker --timeout 60 --bind 0.0.0.0:80
