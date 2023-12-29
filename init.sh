# !/bin/bash

# Miniconda 설치 스크립트
MINICONDA_INSTALLER_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_PREFIX="${HOME}/miniconda3"

# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER_SCRIPT} -O /tmp/${MINICONDA_INSTALLER_SCRIPT}

# Miniconda 설치
bash /tmp/${MINICONDA_INSTALLER_SCRIPT} -b -p ${MINICONDA_PREFIX}

# Miniconda 초기화 및 PATH 설정
eval "$(${MINICONDA_PREFIX}/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
conda update --all -y
conda update -n base -c defaults conda

# 임시 파일 삭제
rm /tmp/${MINICONDA_INSTALLER_SCRIPT}

# data 다운로드
if [ -d "../train" ] && [ -d "../eval" ]; then
	echo -e "\e[34m'train' and 'eval' folders exist in the parent directory\e[0m"
else
	cd ..
	wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000263/data/data.tar.gz
	tar -zxvf data.tar.gz
	rm data.tar.gz
	cd level1-imageclassification-cv-08
fi
echo -e "\e[34mFin data download\e[0m"

conda env create -f environment.yml
source activate level1
echo -e "\e[34mFin conda env\e[0m"

# git 설정
git config --global commit.template ./.commit_template
git config --global core.editor "code --wait"
echo -e "\e[34mFin git config\e[0m"

# pre-commit 설정
pre-commit autoupdate
pre-commit install
echo -e "\e[34mFin pre-commit\e[0m"

echo -e "\e[34mFin init\e[0m"
