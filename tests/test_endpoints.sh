#!/bin/bash

echo "Testing /api/stampli..."
curl -N -X POST http://127.0.0.1:8081/api/stampli \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, is the endpoint working?"}
    ]
  }'

echo -e "\n\n--------------------------------\n"

echo "Testing /api/sage (model=stampli)..."
curl -N -X POST http://127.0.0.1:8081/api/sage \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stampli",
    "messages": [
      {"role": "user", "content": "How are you?"}
    ]
  }'
