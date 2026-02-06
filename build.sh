#!/usr/bin/env bash
set -e  # Stop on any error

echo "=== Installing system dependencies ==="
apt-get update
apt-get install -y build-essential wget

echo "=== Installing TA-Lib C library ==="
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt
