#!/bin/bash
# Install SC2 and add the custom (smac + smacv2) maps
sudo apt install unzip

cd "$HOME"
export SC2PATH="$HOME/StarCraftII"
echo 'SC2PATH is set to '$SC2PATH

LOCAL_PATH="${HOME}/MAPG"

# Function to print usage information
print_usage() {
    echo "Usage: $0 [LOCAL|REMOTE]"
}


if [ $# -ne 1 ]; then
    echo "Error: METHOD is required."
    print_usage
    exit 1
fi

USER_METHOD=$1


if [[ "$USER_METHOD" != "LOCAL" && "$USER_METHOD" != "REMOTE" ]]; then
    echo "Error: METHOD must be either LOCAL or REMOTE."
    print_usage
    exit 1
fi


if [[ -d "$LOCAL_PATH" && "$USER_METHOD" == "LOCAL" ]]; then
    METHOD="LOCAL"
else
    METHOD="REMOTE"
fi

echo "Selected METHOD: $METHOD"

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        if [[ "$METHOD" == "LOCAL" ]]; then
            unzip -P iagreetotheeula ${LOCAL_PATH}/SC2.4.10.zip
        else
            wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
            unzip -P iagreetotheeula SC2.4.10.zip
            rm -rf SC2.4.10.zip
        fi
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

if [[ "$METHOD" == "LOCAL" ]]; then
    unzip ${LOCAL_PATH}/SMAC_Maps.zip
else
    wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
    unzip SMAC_Maps.zip
fi

mkdir -p "$MAP_DIR/SMAC_Maps"
mv *.SC2Map "$MAP_DIR/SMAC_Maps"

echo 'StarCraft II and SMAC are installed.'


# TODO: smac and smacv2