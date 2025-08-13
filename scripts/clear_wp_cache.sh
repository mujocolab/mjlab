#!/bin/bash
# clear_warp_cache.sh — clears Warp's cache directory for the current user

CACHE_DIR="$HOME/.cache/warp"

if [ -d "$CACHE_DIR" ]; then
    echo "Found Warp cache at: $CACHE_DIR"
    echo -n "Are you sure you want to delete the cache? Press Enter to continue or Ctrl+C to cancel: "
    read
    echo "Removing Warp cache..."
    rm -rf "$CACHE_DIR"
    echo "Warp cache cleared."
else
    echo "No Warp cache found at: $CACHE_DIR"
fi