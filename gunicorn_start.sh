#!/bin/bash
exec gunicorn --config /app/gunicorn_config.py main:app