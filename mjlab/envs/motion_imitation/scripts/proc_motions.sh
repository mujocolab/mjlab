#! /bin/bash

# Exit on error
set -e

# Configuration array for motion data
declare -A MOTIONS=(
    ["dance1_subject2"]="3 20"
    ["jumps1_subject1"]="6 10.3"
    ["fightAndSports1_subject1"]="17.5 24.5"
    ["fallAndGetUp2_subject3"]="55 64"
    ["walk1_subject1"]="0 11.5"
)

# Function to augment dataset
augment_dataset() {
    local name=$1
    echo "Augmenting dataset for $name..."
    python mjlab/envs/motion_imitation/scripts/augment_dataset.py --name "$name"
}

# Function to upsample motion
upsample_motion() {
    local name=$1
    local slice_times=$2
    local repeat_last_frame=$3
    local npz_file="mjlab/envs/motion_imitation/data/processed/${name}.npz"

    echo "Upsampling motion for $name..."
    python mjlab/envs/motion_imitation/scripts/upsample_motion.py \
        --npz-file "$npz_file" \
        --target-dt 0.02 \
        --slice-times $slice_times \
        --repeat-last-frame $repeat_last_frame
}

# Main processing loop
main() {
    echo "Starting motion processing..."

    # Process each motion
    for name in "${!MOTIONS[@]}"; do
        echo "Processing $name..."
        augment_dataset "$name"
        if [ "$name" = "fallAndGetUp2_subject3" ]; then
            upsample_motion "$name" "${MOTIONS[$name]}" 3.0
        else
            upsample_motion "$name" "${MOTIONS[$name]}" 0.0
        fi
    done

    echo "Motion processing completed successfully!"
}

# Run main function
main
