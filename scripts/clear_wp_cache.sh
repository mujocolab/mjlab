#!/bin/bash
# clear_warp_cache.sh — clears Warp's cache directory for the current user

CACHE_DIR="$HOME/.cache/warp"

if [ -d "$CACHE_DIR" ]; then
    echo "Removing Warp cache from: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
    echo "Warp cache cleared."
else
    echo "No Warp cache found at: $CACHE_DIR"
fi
