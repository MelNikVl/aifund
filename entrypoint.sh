#!/bin/bash
set -e

echo "Running initial update..."
cd /app && python scripts/update.py

echo "0 0,5,10,15,20 * * * cd /app && python scripts/update.py >> /var/log/aifund.log 2>&1" | crontab -

echo "Starting cron daemon..."
cron -f
