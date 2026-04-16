#!/usr/bin/env bash
set -euo pipefail

export ANTHROPIC_API_KEY=$(cat ~/.gopal/anthropic_key)
export OPENAI_API_KEY=$(cat ~/.gopal/openai_key)

uv run critique "$@"
