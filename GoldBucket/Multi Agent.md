[Setup | MetaGPT (deepwisdom.ai)](https://docs.deepwisdom.ai/main/en/guide/get_started/setup.html)

# 1. Try 1st

## 1.1. Environment

- Ubuntu 22.04 
- Python 3.9+
- install `nodejs` first, then use `npm` to install `mermaid-cli`
- **playwright** As `pyppeteer` is no longer maintained, it recommends using playwright-python as a replacement.
	- `pyppeteer` is a web automation testing tool implemented in Python that can execute JavaScript scripts.
- Install a browser first and set the browser path when running.
	- `playwright install --with-deps chromium`
- `git clone`
```Bash
git clone https://github.com/geekan/MetaGPT.git
cd /your/path/to/MetaGPT
pip install -e .
```

### 1.1.2. Comparison
![[Pasted image 20240116144139.png]]
## 1.2. Setup
- connecting with model providers.
- complete the configuration before use.
### 1.2.1. OpenAI API

- Use a config or key file. This is the recommended way, best for continuous and full-feature usage and development.
	- Find `config.yaml` in repo.
	- Fill your own values:
		- `OPENAI_API_KEY: 'sk-...' # YOUR_API_KEY` 
		- `OPENAI_API_MODEL: 'intended model' # gpt-4, gpt-3.5-turbo, etc.`
### 1.2.2. We often want agents to use tools.
- Web searching API
- Web browsing
- **Azure Text to Speech**
- Stable Diffusion local deployment
# 2. Concepts

## 2.1. MetaGPT's concept of agent and environment
- an agent should be able to think or plan like human
	- possesses memories or even emotions, 
	- is equipped with a certain skill set
	- interact with the environment, other agents, and human.
	- 
## 2.2. Agents interact with each other.
## 2.3. A multi-agent collaboration.

# Installation in Dev
```Bash
(base) ➜  MetaGPT git:(main) conda create --name aicompany python=3.9 
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.1.0
  latest version: 24.1.0

Please update conda by running

    $ conda update -n base -c defaults conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.1.0



## Package Plan ##

  environment location: /home/neon/softwares/anaconda3/envs/aicompany

  added / updated specs:
    - python=3.9


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    bzip2-1.0.8                |       hd590300_5         248 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    ca-certificates-2024.2.2   |       hbcca054_0         152 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libgcc-ng-13.2.0           |       h807b86a_5         752 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libgomp-13.2.0             |       h807b86a_5         410 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libnsl-2.0.1               |       hd590300_0          33 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libsqlite-3.45.1           |       h2797004_0         839 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libuuid-2.38.1             |       h0b41bf4_0          33 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libxcrypt-4.4.36           |       hd590300_1          98 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    libzlib-1.2.13             |       hd590300_5          60 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    openssl-3.2.1              |       hd590300_0         2.7 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    pip-24.0                   |     pyhd8ed1ab_0         1.3 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    python-3.9.18              |h0755675_1_cpython        22.7 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    tk-8.6.13                  |noxft_h4845f30_101         3.2 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    tzdata-2024a               |       h0c530f3_0         117 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    ------------------------------------------------------------
                                           Total:        32.6 MB

The following NEW packages will be INSTALLED:

  _libgcc_mutex      anaconda/cloud/conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge 
  _openmp_mutex      anaconda/cloud/conda-forge/linux-64::_openmp_mutex-4.5-2_gnu 
  bzip2              anaconda/cloud/conda-forge/linux-64::bzip2-1.0.8-hd590300_5 
  ca-certificates    anaconda/cloud/conda-forge/linux-64::ca-certificates-2024.2.2-hbcca054_0 
  ld_impl_linux-64   anaconda/cloud/conda-forge/linux-64::ld_impl_linux-64-2.40-h41732ed_0 
  libffi             anaconda/cloud/conda-forge/linux-64::libffi-3.4.2-h7f98852_5 
  libgcc-ng          anaconda/cloud/conda-forge/linux-64::libgcc-ng-13.2.0-h807b86a_5 
  libgomp            anaconda/cloud/conda-forge/linux-64::libgomp-13.2.0-h807b86a_5 
  libnsl             anaconda/cloud/conda-forge/linux-64::libnsl-2.0.1-hd590300_0 
  libsqlite          anaconda/cloud/conda-forge/linux-64::libsqlite-3.45.1-h2797004_0 
  libuuid            anaconda/cloud/conda-forge/linux-64::libuuid-2.38.1-h0b41bf4_0 
  libxcrypt          anaconda/cloud/conda-forge/linux-64::libxcrypt-4.4.36-hd590300_1 
  libzlib            anaconda/cloud/conda-forge/linux-64::libzlib-1.2.13-hd590300_5 
  ncurses            anaconda/cloud/conda-forge/linux-64::ncurses-6.4-h59595ed_2 
  openssl            anaconda/cloud/conda-forge/linux-64::openssl-3.2.1-hd590300_0 
  pip                anaconda/cloud/conda-forge/noarch::pip-24.0-pyhd8ed1ab_0 
  python             anaconda/cloud/conda-forge/linux-64::python-3.9.18-h0755675_1_cpython 
  readline           anaconda/cloud/conda-forge/linux-64::readline-8.2-h8228510_1 
  setuptools         anaconda/cloud/conda-forge/noarch::setuptools-69.0.3-pyhd8ed1ab_0 
  tk                 anaconda/cloud/conda-forge/linux-64::tk-8.6.13-noxft_h4845f30_101 
  tzdata             anaconda/cloud/conda-forge/noarch::tzdata-2024a-h0c530f3_0 
  wheel              anaconda/cloud/conda-forge/noarch::wheel-0.42.0-pyhd8ed1ab_0 
  xz                 anaconda/cloud/conda-forge/linux-64::xz-5.2.6-h166bdaf_0 


Proceed ([y]/n)? y


Downloading and Extracting Packages
                                                                                                                                           
Preparing transaction: done                                                                                                                
Verifying transaction: done                                                                                                                
Executing transaction: done                                                                                                                
#                                                                                                                                          
# To activate this environment, use                                                                                                        
#                                                                                                                                          
#     $ conda activate aicompany                                                                                                           
#                                                                                                                                          
# To deactivate an active environment, use                                                                                                 
#                                                                                                                                          
#     $ conda deactivate
```
