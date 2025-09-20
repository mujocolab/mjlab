# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
export MJLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# Helper functions
#==

# setup uv environment for Mjlab
setup_uv_env() {
    # get environment name from input
    local env_name="env_mjlab"
    local python_path="env_mjlab/bin/python"

    # check uv is installed
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    # check if the environment exists
    local env_path="${MJLAB_PATH}/${env_name}"
    if [ ! -d "${env_path}" ]; then
        uv cache clean
        echo -e "[INFO] Creating uv environment named '${env_name}'..."
        uv venv --clear --python $(which python3) "${env_path}"
    else
        echo "[INFO] uv environment '${env_name}' already exists."
    fi

    # ensure activate file exists
    touch "${env_path}/bin/activate"

    uv pip install -e . --python ${MJLAB_PATH}/${python_path}

    # TODO LOUIS: test if not already written
    # add variables to environment during activation
    cat >> "${env_path}/bin/activate" <<EOF
export MJLAB_PATH="${MJLAB_PATH}"
alias mjlab="${MJLAB_PATH}/${python_path}"
EOF

    cat >> ~/.bashrc <<EOF
alias env_mjlab="source ${env_path}/bin/activate"
EOF

    # add information to the user about alias
    echo -e "[INFO] Added 'env_mjlab' alias to ~/.bashrc."
    echo -e "[INFO] Added 'env_mjlab' alias to activate uv environment for 'mjlab' to run script."
    echo -e "[INFO] Created uv environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the alias, run once:            source ~/.bashrc"
    echo -e "\t\t2. To activate the environment, run:           env_mjlab"
    echo -e "\t\t3. To test the alias/environment, run:         mjlab scripts/list_envs.py"
    echo -e "\t\t4. To deactivate the environment, run:         deactivate"
    echo -e "\n"
}


# TODO LOUIS: add format
# TODO LOUIS: add test
# TODO LOUIS: add ext project generator
# TODO LOUIS: add docs (in the future) if using sphinx
# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-u] -- Utility to manage Mjlab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-u, --uv [NAME]      Create the uv environment for Mjlab. Default name is 'env_mjlab'."
    echo -e "\n" >&2
}


#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 0
fi

# pass the arguments
while [[ $# -gt 0 ]]; do
    # read the key
    case "$1" in
        -u|--uv)
            echo "[INFO] Using default uv environment name: env_mjlab"
            # setup the uv environment for Mjlab
            setup_uv_env
            shift # past argument
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *) # unknown option
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done