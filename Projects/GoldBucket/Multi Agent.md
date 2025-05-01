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
(aicompany) ➜  MetaGPT git:(main) pip install -e .
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Obtaining file:///home/neon/projects/MetaGPT
  Preparing metadata (setup.py) ... done
Collecting aiohttp==3.8.4 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8b/60/91ef15bf94415c6749edb6fd168394c1f451e7bb4b3f32c023654e8ea91e/aiohttp-3.8.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 5.5 MB/s eta 0:00:00
Collecting channels==4.0.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f5/52/a233dc63996547f171c2013f2d0505dcfa7d0557e7cde8748a2bd70b5a31/channels-4.0.0-py3-none-any.whl (28 kB)
Collecting faiss_cpu==1.7.4 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b5/0c/97faf412f20b3e0f52c02c894871f3213c72c73d57d1f039fe867d3c3437/faiss_cpu-1.7.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.6/17.6 MB 3.4 MB/s eta 0:00:00
Collecting fire==0.4.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/11/07/a119a1aa04d37bc819940d95ed7e135a7dcca1c098123a3764a6dcace9e7/fire-0.4.0.tar.gz (87 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.7/87.7 kB 2.0 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting typer==0.9.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bf/0e/c68adf10adda05f28a6ed7b9f4cd7b8e07f641b44af88ba72d9c89e4de7a/typer-0.9.0-py3-none-any.whl (45 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.9/45.9 kB 1.4 MB/s eta 0:00:00
Collecting lancedb==0.4.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3e/35/2dfde0b42b6f86cdb52eec79beb3ed2a78f8f22331e3cbfdedeae746c415/lancedb-0.4.0-py3-none-any.whl (81 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.4/81.4 kB 1.8 MB/s eta 0:00:00
Collecting langchain==0.0.352 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0f/36/58f4d9df45436670a5b6b82ff48522b6233fa35bd21b133b149c1c7ec8bd/langchain-0.0.352-py3-none-any.whl (794 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 794.4/794.4 kB 3.7 MB/s eta 0:00:00
Collecting loguru==0.6.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fe/21/e1d1da2586865a159fc73b611f36bdd50b6c4043cb6132d3d5e972988028/loguru-0.6.0-py3-none-any.whl (58 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 2.0 MB/s eta 0:00:00
Collecting meilisearch==0.21.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2d/d8/cc75e30ce1a43898621d5b2fe780b213ad7e4c8f2adccd29851d47d39a11/meilisearch-0.21.0-py3-none-any.whl (19 kB)
Collecting numpy>=1.24.3 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/54/30/c2a907b9443cf42b90c17ad10c1e8fa801975f01cb9764f3f8eb8aea638b/numpy-1.26.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 3.5 MB/s eta 0:00:00
Collecting openai==1.6.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/07/de/ef3534d9417f7c72c75036fae6c85d9071aebbce8aa3616d3e69b9f0ca4d/openai-1.6.0-py3-none-any.whl (225 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 225.4/225.4 kB 2.7 MB/s eta 0:00:00
Collecting openpyxl (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6a/94/a59521de836ef0da54aaf50da6c4da8fb4072fb3053fa71f052fd9399e7a/openpyxl-3.1.2-py2.py3-none-any.whl (249 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 250.0/250.0 kB 2.9 MB/s eta 0:00:00
Collecting beautifulsoup4==4.12.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/57/f4/a69c20ee4f660081a7dedb1ac57f29be9378e04edfcb90c526b923d4bebc/beautifulsoup4-4.12.2-py3-none-any.whl (142 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 143.0/143.0 kB 2.2 MB/s eta 0:00:00
Collecting pandas==2.0.3 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9e/0d/91a9fd2c202f2b1d97a38ab591890f86480ecbb596cbc56d035f6f23fdcc/pandas-2.0.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.4/12.4 MB 6.0 MB/s eta 0:00:00
Collecting pydantic==2.5.3 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dd/b7/9aea7ee6c01fe3f3c03b8ca3c7797c866df5fecece9d6cb27caa138db2e2/pydantic-2.5.3-py3-none-any.whl (381 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 381.9/381.9 kB 6.9 MB/s eta 0:00:00
Collecting python_docx==0.8.11 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8b/a0/52729ce4aa026f31b74cc877be1d11e4ddeaa361dc7aebec148171644b33/python-docx-0.8.11.tar.gz (5.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.6/5.6 MB 5.7 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting PyYAML==6.0.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7d/39/472f2554a0f1e825bd7c5afc11c817cd7a2f3657460f7159f691fbb37c51/PyYAML-6.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (738 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 738.9/738.9 kB 2.7 MB/s eta 0:00:00
Collecting setuptools==65.6.3 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ef/e3/29d6e1a07e8d90ace4a522d9689d03e833b67b50d1588e693eec15f26251/setuptools-65.6.3-py3-none-any.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 2.8 MB/s eta 0:00:00
Collecting tenacity==8.2.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e7/b0/c23bd61e1b32c9b96fbca996c87784e196a812da8d621d8d04851f6c8181/tenacity-8.2.2-py3-none-any.whl (24 kB)
Collecting tiktoken==0.5.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d2/3a/64a173d645cdf5609e2e7969b4f7cd3dd48f8cb2f6d0b29a34d245f3cbdf/tiktoken-0.5.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 3.1 MB/s eta 0:00:00
Collecting tqdm==4.65.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e6/02/a2cff6306177ae6bc73bc0665065de51dfb3b9db7373e122e2735faf0d97/tqdm-4.65.0-py3-none-any.whl (77 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.1/77.1 kB 1.7 MB/s eta 0:00:00
Collecting anthropic==0.8.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d3/69/a575efc29ce7ddb0fe8ff43e78d9767b3855bd8a13043b59717533e3db4b/anthropic-0.8.1-py3-none-any.whl (826 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 826.7/826.7 kB 3.5 MB/s eta 0:00:00
Collecting typing-inspect==0.8.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/be/01/59b743dca816c4b6ca891b9e0f84d20513cd61bdbbaa8615de8f5aab68c1/typing_inspect-0.8.0-py3-none-any.whl (8.7 kB)
Collecting libcst==1.0.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/af/3a/90caf552575489412bc4b041ca9894062ca2f0200cd7df8884325f0575bf/libcst-1.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 5.1 MB/s eta 0:00:00
Collecting qdrant-client==1.7.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/80/43/cf47f8c8c231612ab3f92f5ddcbd78b07bfca3e31656a9b87d200d545cb2/qdrant_client-1.7.0-py3-none-any.whl (203 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 203.7/203.7 kB 3.2 MB/s eta 0:00:00
Collecting ta==0.10.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9a/1b/b8efb240be6d904c2f0ec54dd52d55bafccbcfa72d7e688e0e10aef63e00/ta-0.10.2.tar.gz (25 kB)
  Preparing metadata (setup.py) ... done
Collecting semantic-kernel==0.4.3.dev0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/14/a2/468d2caa9c14de4d9e2d9c6adb39e71d91a2ee874ecb525838fcf2c92784/semantic_kernel-0.4.3.dev0-py3-none-any.whl (214 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 214.9/214.9 kB 4.6 MB/s eta 0:00:00
Collecting wrapt==1.15.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dd/eb/389f9975a6be31ddd19d29128a11f1288d07b624e464598a4b450f8d007e/wrapt-1.15.0-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.3/78.3 kB 2.6 MB/s eta 0:00:00
Collecting aioredis~=2.0.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9b/a9/0da089c3ae7a31cbcd2dcf0214f6f571e1295d292b6139e2bac68ec081d0/aioredis-2.0.1-py3-none-any.whl (71 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.2/71.2 kB 2.4 MB/s eta 0:00:00
Collecting websocket-client==1.6.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4b/4a/3176388095e5bae6e6a1fbee66c438809230ae0196e7de4af12c5e75c509/websocket_client-1.6.2-py3-none-any.whl (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.0/57.0 kB 2.1 MB/s eta 0:00:00
Collecting aiofiles==23.2.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c5/19/5af6804c4cc0fed83f47bff6e413a98a36618e7d40185cd36e69737f3b0e/aiofiles-23.2.1-py3-none-any.whl (15 kB)
Collecting gitpython==3.1.40 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8d/c4/82b858fb6483dfb5e338123c154d19c043305b01726a67d89532b8f8f01b/GitPython-3.1.40-py3-none-any.whl (190 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 190.6/190.6 kB 4.1 MB/s eta 0:00:00
Collecting zhipuai==2.0.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8f/05/c3d4556886b5c6cf8c0b96eb80448ee8154c0dcc87086df018e817779ed4/zhipuai-2.0.1-py3-none-any.whl (26 kB)
Collecting rich==13.6.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/be/2a/4e62ff633612f746f88618852a626bbe24226eba5e7ac90e91dcfd6a414e/rich-13.6.0-py3-none-any.whl (239 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 239.8/239.8 kB 4.9 MB/s eta 0:00:00
Collecting nbclient==0.9.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6b/3a/607149974149f847125c38a62b9ea2b8267eb74823bbf8d8c54ae0212a00/nbclient-0.9.0-py3-none-any.whl (24 kB)
Collecting nbformat==5.9.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f4/e7/ef30a90b70eba39e675689b9eaaa92530a71d7435ab8f9cae520814e0caf/nbformat-5.9.2-py3-none-any.whl (77 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.6/77.6 kB 2.7 MB/s eta 0:00:00
Collecting ipython==8.17.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/20/45/18f0dc2cbc3ee6680a004f620fb1400c6511ded0a76a2dd241813786ce73/ipython-8.17.2-py3-none-any.whl (808 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 808.4/808.4 kB 5.6 MB/s eta 0:00:00
Collecting ipykernel==6.27.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1e/36/1c316a31d42d323de41616c98e353bd1db1b716980c90929832de4755f80/ipykernel-6.27.0-py3-none-any.whl (114 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 114.6/114.6 kB 3.3 MB/s eta 0:00:00
Collecting scikit_learn==1.3.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/25/89/dce01a35d354159dcc901e3c7e7eb3fe98de5cb3639c6cd39518d8830caa/scikit_learn-1.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.9/10.9 MB 5.7 MB/s eta 0:00:00
Collecting typing-extensions==4.9.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b7/f4/6a90020cd2d93349b442bfcb657d0dc91eee65491600b2cb1d388bc98e6b/typing_extensions-4.9.0-py3-none-any.whl (32 kB)
Collecting socksio~=1.0.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/37/c3/6eeb6034408dac0fa653d126c9204ade96b819c936e136c5e8a6897eee9c/socksio-1.0.0-py3-none-any.whl (12 kB)
Collecting gitignore-parser==0.1.9 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0f/0d/c3278a636547407d2d4d57d3e229492f4bf5417147aa9b5cac144cf6abfc/gitignore_parser-0.1.9.tar.gz (5.3 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting websockets~=12.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/06/dd/e8535f54b4aaded1ed44041ca8eb9de8786ce719ff148b56b4a903ef93e6/websockets-12.0-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.0/130.0 kB 2.3 MB/s eta 0:00:00
Collecting networkx~=3.2.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d5/f0/8fbc882ca80cf077f1b246c0e3c3465f7f415439bdea6b899f6b19f61f70/networkx-3.2.1-py3-none-any.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 2.9 MB/s eta 0:00:00
Collecting google-generativeai==0.3.2 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b5/7f/35f89209487f8473edc9d2cecef894a54680cf666e32893a767d12a8dba9/google_generativeai-0.3.2-py3-none-any.whl (146 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 146.9/146.9 kB 2.5 MB/s eta 0:00:00
Collecting playwright>=1.26 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/aa/485efb664d9808715d1a130b6bc33aac12d2156a40310bdc385811d95048/playwright-1.41.2-py3-none-manylinux1_x86_64.whl (37.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.4/37.4 MB 3.2 MB/s eta 0:00:00
Collecting anytree (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6a/fb/ff946843e6b55ae9fda84df3964d6c233cd2261dface789f5be02ab79bc5/anytree-2.12.1-py3-none-any.whl (44 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.9/44.9 kB 1.5 MB/s eta 0:00:00
Collecting ipywidgets==8.1.1 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4a/0e/57ed498fafbc60419a9332d872e929879ceba2d73cb11d284d7112472b3e/ipywidgets-8.1.1-py3-none-any.whl (139 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.4/139.4 kB 2.5 MB/s eta 0:00:00
Collecting Pillow (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/87/0d/8f5136a5481731c342a901ff155c587ce7804114db069345e1894ab4978a/pillow-10.2.0-cp39-cp39-manylinux_2_28_x86_64.whl (4.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 6.4 MB/s eta 0:00:00
Collecting imap_tools==1.5.0 (from metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8a/21/83cd4c8de113ab92e302f83b7843a51588bcd85e4b9baf765e5b0a04ef22/imap_tools-1.5.0-py3-none-any.whl (32 kB)
Collecting attrs>=17.3.0 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e0/44/827b2a91a5816512fcaf3cc4ebc465ccd5d598c45cefa6703fcf4a79018f/attrs-23.2.0-py3-none-any.whl (60 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.8/60.8 kB 2.5 MB/s eta 0:00:00
Collecting charset-normalizer<4.0,>=2.0 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/69/5d8751b4b670d623aa7a47bef061d69c279e9f922f6705147983aa76c3ce/charset_normalizer-3.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.3/142.3 kB 1.6 MB/s eta 0:00:00
Collecting multidict<7.0,>=4.5 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/a9/1f8d42c8103bcb1da6bb719f1bc018594b5acc8eae56b3fec4720ebee225/multidict-6.0.5-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (123 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 123.8/123.8 kB 3.4 MB/s eta 0:00:00
Collecting async-timeout<5.0,>=4.0.0a3 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a7/fa/e01228c2938de91d47b307831c62ab9e4001e747789d0b05baf779a6488c/async_timeout-4.0.3-py3-none-any.whl (5.7 kB)
Collecting yarl<2.0,>=1.0 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/69/ea/d7e961ea9b1b818a43b155ee512117be6ab9ab67c1e94967b2e64126e8e4/yarl-1.9.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (304 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 304.3/304.3 kB 3.3 MB/s eta 0:00:00
Collecting frozenlist>=1.1.1 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/70/b0/6f1ebdabfb604e39a0f84428986b89ab55f246b64cddaa495f2c953e1f6b/frozenlist-1.4.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (240 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 240.7/240.7 kB 5.2 MB/s eta 0:00:00
Collecting aiosignal>=1.1.2 (from aiohttp==3.8.4->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/76/ac/a7305707cb852b7e16ff80eaf5692309bde30e2b1100a1fcacdc8f731d97/aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
Collecting anyio<5,>=3.5.0 (from anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bf/cd/d6d9bb1dadf73e7af02d18225cbd2c93f8552e13130484f1c8dcfece292b/anyio-4.2.0-py3-none-any.whl (85 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.5/85.5 kB 3.1 MB/s eta 0:00:00
Collecting distro<2,>=1.7.0 (from anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/12/b3/231ffd4ab1fc9d679809f356cebee130ac7daa00d6d6f3206dd4fd137e9e/distro-1.9.0-py3-none-any.whl (20 kB)
Collecting httpx<1,>=0.23.0 (from anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/9b/4937d841aee9c2c8102d9a4eeb800c7dad25386caabb4a1bf5010df81a57/httpx-0.26.0-py3-none-any.whl (75 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.9/75.9 kB 2.8 MB/s eta 0:00:00
Collecting sniffio (from anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c3/a0/5dba8ed157b0136607c7f2151db695885606968d1fae123dc3391e0cfdbf/sniffio-1.3.0-py3-none-any.whl (10 kB)
Collecting tokenizers>=0.13.0 (from anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a5/bc/ec39dae3b0ea00724c0fea287091d62b0ccaa45c7a947004714e882d193d/tokenizers-0.15.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.6/3.6 MB 9.5 MB/s eta 0:00:00
Collecting soupsieve>1.2 (from beautifulsoup4==4.12.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4c/f3/038b302fdfbe3be7da016777069f26ceefe11a681055ea1f7817546508e3/soupsieve-2.5-py3-none-any.whl (36 kB)
Collecting Django>=3.2 (from channels==4.0.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/20/8c/498ce1ac30228f3bc36d4d427fe10268bd01e6f0361ead6c60b2b255c6b6/Django-4.2.10-py3-none-any.whl (8.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.0/8.0 MB 8.1 MB/s eta 0:00:00
Collecting asgiref<4,>=3.5.0 (from channels==4.0.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9b/80/b9051a4a07ad231558fcd8ffc89232711b4e618c15cb7a392a17384bbeef/asgiref-3.7.2-py3-none-any.whl (24 kB)
Collecting six (from fire==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d9/5a/e7c31adbe875f2abbb91bd84cf2dc52d792b5a01506781dbcf25c91daf11/six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting termcolor (from fire==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d9/5f/8c716e47b3a50cbd7c146f45881e11d9414def768b7cd9c5e6650ec2a80a/termcolor-2.4.0-py3-none-any.whl (7.7 kB)
Collecting gitdb<5,>=4.0.1 (from gitpython==3.1.40->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fd/5b/8f0c4a5bb9fd491c277c21eff7ccae71b47d43c4446c9d0c6cff2fe8c2c4/gitdb-4.0.11-py3-none-any.whl (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.7/62.7 kB 1.7 MB/s eta 0:00:00
Collecting google-ai-generativelanguage==0.4.0 (from google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/40/c2/d28988d3cba74e712f47a498e2b3e3b58ac215106019bf5d8c20f8ab9822/google_ai_generativelanguage-0.4.0-py3-none-any.whl (598 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 598.7/598.7 kB 2.8 MB/s eta 0:00:00
Collecting google-auth (from google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/82/41/7fb855444cead5b2213e053447ce3a0b7bf2c3529c443e0cf75b2f13b405/google_auth-2.27.0-py2.py3-none-any.whl (186 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 186.8/186.8 kB 2.6 MB/s eta 0:00:00
Collecting google-api-core (from google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/60/51/2054dfc08dda9a3add0d715cee98d6f8211c99bd6e5bff0ff1bdd3cf3384/google_api_core-2.17.0-py3-none-any.whl (136 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 136.9/136.9 kB 2.2 MB/s eta 0:00:00
Collecting protobuf (from google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/81/9e/63501b8d5b4e40c7260049836bd15ec3270c936e83bc57b85e4603cc212c/protobuf-4.25.2-cp37-abi3-manylinux2014_x86_64.whl (294 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.6/294.6 kB 2.8 MB/s eta 0:00:00
Collecting comm>=0.1.1 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6e/c1/e7335bd49aa3fa3bd453e34a4580b0076804f219897ad76d4d5aa4d8f22f/comm-0.2.1-py3-none-any.whl (7.2 kB)
Collecting debugpy>=1.6.5 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ba/2c/69244a26dc484b37db8fc76ba5af107d501d0e45be27b3042b74aa332aa4/debugpy-1.8.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 3.2 MB/s eta 0:00:00
Collecting jupyter-client>=6.1.12 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/43/ae/5f4f72980765e2e5e02b260f9c53bcc706cefa7ac9c8d7240225c55788d4/jupyter_client-8.6.0-py3-none-any.whl (105 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.9/105.9 kB 2.4 MB/s eta 0:00:00
Collecting jupyter-core!=5.0.*,>=4.12 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/86/a1/354cade6907f2fbbd32d89872ec64b62406028e7645ac13acfdb5732829e/jupyter_core-5.7.1-py3-none-any.whl (28 kB)
Collecting matplotlib-inline>=0.1 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f2/51/c34d7a1d528efaae3d8ddb18ef45a41f284eacf9e514523b191b7d0872cc/matplotlib_inline-0.1.6-py3-none-any.whl (9.4 kB)
Collecting nest-asyncio (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a0/c4/c2971a3ba4c6103a3d10c4b0f24f461ddc027f0f09763220cf35ca1401b3/nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
Collecting packaging (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ec/1a/610693ac4ee14fcdf2d9bf3c493370e4f2ef7ae2e19217d7a237ff42367d/packaging-23.2-py3-none-any.whl (53 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 53.0/53.0 kB 1.7 MB/s eta 0:00:00
Collecting psutil (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c5/4f/0e22aaa246f96d6ac87fe5ebb9c5a693fbe8877f537a1022527c47ca43c5/psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (288 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 288.2/288.2 kB 3.6 MB/s eta 0:00:00
Collecting pyzmq>=20 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/76/8b/6fca99e22c6316917de32b17be299dea431544209d619da16b6d9ec85c83/pyzmq-25.1.2-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 4.0 MB/s eta 0:00:00
Collecting tornado>=6.1 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9f/12/11d0a757bb67278d3380d41955ae98527d5ad18330b2edbdc8de222b569b/tornado-6.4-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (435 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 435.4/435.4 kB 4.6 MB/s eta 0:00:00
Collecting traitlets>=5.4.0 (from ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/45/34/5dc77fdc7bb4bd198317eea5679edf9cc0a186438b5b19dbb9062fb0f4d5/traitlets-5.14.1-py3-none-any.whl (85 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.4/85.4 kB 3.0 MB/s eta 0:00:00
Collecting decorator (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d5/50/83c593b07763e1161326b3b8c6686f0f4b0f24d5526546bee538c89837d6/decorator-5.1.1-py3-none-any.whl (9.1 kB)
Collecting jedi>=0.16 (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/20/9f/bc63f0f0737ad7a60800bfd472a4836661adae21f9c2535f3957b1e54ceb/jedi-0.19.1-py2.py3-none-any.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 4.8 MB/s eta 0:00:00
Collecting prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ee/fd/ca7bf3869e7caa7a037e23078539467b433a4e01eebd93f77180ab927766/prompt_toolkit-3.0.43-py3-none-any.whl (386 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 386.1/386.1 kB 4.8 MB/s eta 0:00:00
Collecting pygments>=2.4.0 (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/97/9c/372fef8377a6e340b1704768d20daaded98bf13282b5327beb2e2fe2c7ef/pygments-2.17.2-py3-none-any.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 4.7 MB/s eta 0:00:00
Collecting stack-data (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f1/7b/ce1eafaf1a76852e2ec9b22edecf1daa58175c090266e9f6c64afcd81d91/stack_data-0.6.3-py3-none-any.whl (24 kB)
Collecting exceptiongroup (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b8/9a/5028fd52db10e600f1c4674441b968cf2ea4959085bfb5b99fb1250e5f68/exceptiongroup-1.2.0-py3-none-any.whl (16 kB)
Collecting pexpect>4.3 (from ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9e/c3/059298687310d527a58bb01f3b1965787ee3b40dce76752eda8b44e9a2c5/pexpect-4.9.0-py2.py3-none-any.whl (63 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.8/63.8 kB 792.8 kB/s eta 0:00:00
Collecting widgetsnbextension~=4.0.9 (from ipywidgets==8.1.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/99/bc/82a8c3985209ca7c0a61b383c80e015fd92e74f8ba0ec1af98f9d6ca8dce/widgetsnbextension-4.0.10-py3-none-any.whl (2.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 6.8 MB/s eta 0:00:00
Collecting jupyterlab-widgets~=3.0.9 (from ipywidgets==8.1.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/24/da/db1cb0387a7e4086780aff137987ee924e953d7f91b2a870f994b9b1eeb8/jupyterlab_widgets-3.0.10-py3-none-any.whl (215 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 215.0/215.0 kB 5.8 MB/s eta 0:00:00
Collecting deprecation (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/02/c3/253a89ee03fc9b9682f1541728eb66db7db22148cd94f89ab22528cd1e1b/deprecation-2.1.0-py2.py3-none-any.whl (11 kB)
Collecting pylance==0.9.0 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ec/74/09b864f25d108d977b4416d760269e874018192a147f662a5ed2727b9b53/pylance-0.9.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (19.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.5/19.5 MB 3.3 MB/s eta 0:00:00
Collecting ratelimiter~=1.0 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/51/80/2164fa1e863ad52cc8d870855fba0fbb51edd943edffd516d54b5f6f8ff8/ratelimiter-1.2.0.post0-py3-none-any.whl (6.6 kB)
Collecting retry>=0.9.2 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4b/0d/53aea75710af4528a25ed6837d71d117602b01946b307a3912cb3cfcbcba/retry-0.9.2-py2.py3-none-any.whl (8.0 kB)
Collecting semver>=3.0 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9a/77/0cc7a8a3bc7e53d07e8f47f147b92b0960e902b8254859f4aee5c4d7866b/semver-3.0.2-py3-none-any.whl (17 kB)
Collecting cachetools (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a2/91/2d843adb9fbd911e0da45fbf6f18ca89d07a087c3daa23e955584f90ebf4/cachetools-5.3.2-py3-none-any.whl (9.3 kB)
Collecting click>=8.1.7 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/00/2e/d53fa4befbf2cfa713304affc7ca780ce4fc1fd8710527771b58311a3229/click-8.1.7-py3-none-any.whl (97 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 97.9/97.9 kB 2.1 MB/s eta 0:00:00
Collecting requests>=2.31.0 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/70/8e/0e2d847013cb52cd35b38c009bb167a1a26b2ce6cd6965bf26b47bc0bf44/requests-2.31.0-py3-none-any.whl (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 1.9 MB/s eta 0:00:00
Collecting overrides>=0.7 (from lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2c/ab/fc8290c6a4c722e5514d80f62b2dc4c4df1a68a41d1364e625c35990fcf3/overrides-7.7.0-py3-none-any.whl (17 kB)
Collecting SQLAlchemy<3,>=1.4 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4d/d2/ac0e31fbd7af39f099bc2c7b769ed6c91d63e46e2ed15f224101ef107b70/SQLAlchemy-2.0.26-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 3.4 MB/s eta 0:00:00
Collecting dataclasses-json<0.7,>=0.5.7 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/91/ca/7219b838086086972e662c19e908694bdc6744537fb41b70392501b8b5e4/dataclasses_json-0.6.4-py3-none-any.whl (28 kB)
Collecting jsonpatch<2.0,>=1.33 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/73/07/02e16ed01e04a374e644b575638ec7987ae846d25ad97bcc9945a3ee4b0e/jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
Collecting langchain-community<0.1,>=0.0.2 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bf/b4/1b1b22ab0c57320c5476b735cfe1500e49ddc4425df9e4c2e569e4c4472e/langchain_community-0.0.19-py3-none-any.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 6.0 MB/s eta 0:00:00
Collecting langchain-core<0.2,>=0.1 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c8/69/f9cd4ef398973830afe697417070d712ef1fbb3b9a768b8599f555deb687/langchain_core-0.1.22-py3-none-any.whl (239 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 239.4/239.4 kB 6.0 MB/s eta 0:00:00
Collecting langsmith<0.1.0,>=0.0.70 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/71/41/02beb3f8e22c258e0643c8b1e2ccf0d47888edcaae6895580a38f708b9ee/langsmith-0.0.90-py3-none-any.whl (55 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.7/55.7 kB 2.5 MB/s eta 0:00:00
Collecting camel-converter[pydantic] (from meilisearch==0.21.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/55/fd/065986b2e7d609f57ad94c3b6f883d375235a68ccf1adfa4976d31f67614/camel_converter-3.1.1-py3-none-any.whl (5.7 kB)
Collecting fastjsonschema (from nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9c/b9/79691036d4a8f9857e74d1728b23f34f583b81350a27492edda58d5604e1/fastjsonschema-2.19.1-py3-none-any.whl (23 kB)
Collecting jsonschema>=2.6 (from nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/9d/b035d024c62c85f2e2d4806a59ca7b8520307f34e0932fbc8cc75fe7b2d9/jsonschema-4.21.1-py3-none-any.whl (85 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.5/85.5 kB 3.4 MB/s eta 0:00:00
Collecting python-dateutil>=2.8.2 (from pandas==2.0.3->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/36/7a/87837f39d0296e723bb9b62bbb257d0355c7f6128853c78955f57342a56d/python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 247.7/247.7 kB 5.8 MB/s eta 0:00:00
Collecting pytz>=2020.1 (from pandas==2.0.3->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9c/3d/a121f284241f08268b21359bd425f7d4825cffc5ac5cd0e1b3d82ffd2b10/pytz-2024.1-py2.py3-none-any.whl (505 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 505.5/505.5 kB 6.4 MB/s eta 0:00:00
Collecting tzdata>=2022.1 (from pandas==2.0.3->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/65/58/f9c9e6be752e9fcb8b6a0ee9fb87e6e7a1f6bcab2cdc73f02bb7ba91ada0/tzdata-2024.1-py2.py3-none-any.whl (345 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 345.4/345.4 kB 6.4 MB/s eta 0:00:00
Collecting annotated-types>=0.4.0 (from pydantic==2.5.3->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/28/78/d31230046e58c207284c6b2c4e8d96e6d3cb4e52354721b944d3e1ee4aa5/annotated_types-0.6.0-py3-none-any.whl (12 kB)
Collecting pydantic-core==2.14.6 (from pydantic==2.5.3->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/55/0f/45626f8bf7f7973320531bb384ac302eb9b05a70885b9db2bf1db4cf447b/pydantic_core-2.14.6-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 8.4 MB/s eta 0:00:00
Collecting lxml>=2.3.2 (from python_docx==0.8.11->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cb/1c/32a4a764ffce986a0fdd35409f26349ba3181ba1c557c151454a2fb831ac/lxml-5.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.0/8.0 MB 9.4 MB/s eta 0:00:00
Collecting grpcio>=1.41.0 (from qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/09/94/63ac81ac451d1de8351681a36146353e57afa1fcc897a4a67d765904e3a0/grpcio-1.60.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 3.1 MB/s eta 0:00:00
Collecting grpcio-tools>=1.41.0 (from qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c0/2d/a184ed1d28086536423c5dfbcf5ecfca515810001eb5a8670be3b28725b8/grpcio_tools-1.60.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 5.1 MB/s eta 0:00:00
Collecting portalocker<3.0.0,>=2.7.0 (from qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/17/9e/87671efcca80ba6203811540ed1f9c0462c1609d2281d7b7f53cef05da3d/portalocker-2.8.2-py3-none-any.whl (17 kB)
Collecting urllib3<2.0.0,>=1.26.14 (from qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b0/53/aa91e163dcfd1e5b82d8a890ecf13314e3e149c05270cc644581f77f17fd/urllib3-1.26.18-py2.py3-none-any.whl (143 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 143.8/143.8 kB 1.6 MB/s eta 0:00:00
Collecting markdown-it-py>=2.2.0 (from rich==13.6.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.5/87.5 kB 3.2 MB/s eta 0:00:00
Collecting scipy>=1.5.0 (from scikit_learn==1.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a6/9d/f864266894b67cdb5731ab531afba68713da3d6d8252f698ccab775d3f68/scipy-1.12.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.5/38.5 MB 3.3 MB/s eta 0:00:00
Collecting joblib>=1.1.1 (from scikit_learn==1.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/10/40/d551139c85db202f1f384ba8bcf96aca2f329440a844f924c8a0040b6d02/joblib-1.3.2-py3-none-any.whl (302 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 302.2/302.2 kB 1.9 MB/s eta 0:00:00
Collecting threadpoolctl>=2.0.0 (from scikit_learn==1.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/81/12/fd4dea011af9d69e1cad05c75f3f7202cdcbeac9b712eea58ca779a72865/threadpoolctl-3.2.0-py3-none-any.whl (15 kB)
Collecting motor<4.0.0,>=3.3.1 (from semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3f/9a/1a43a329dffbd1a631c52e64c1e9c036621afdfd7f42096ae4bf2de4132b/motor-3.3.2-py3-none-any.whl (70 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70.6/70.6 kB 2.1 MB/s eta 0:00:00
Collecting openapi_core<0.19.0,>=0.18.0 (from semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3c/b0/7e386f89c336d21577e01b77efbba60edfe1b5732124b746bc1d02efdd72/openapi_core-0.18.2-py3-none-any.whl (82 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 82.4/82.4 kB 2.2 MB/s eta 0:00:00
Collecting prance<24.0.0.0,>=23.6.21.0 (from semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c9/db/4fb4901ee61274d0ab97746461fc5f2637e5d73aa73f34ee28e941a699a1/prance-23.6.21.0-py3-none-any.whl (36 kB)
Collecting python-dotenv==1.0.0 (from semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/44/2f/62ea1c8b593f4e093cc1a7768f0d46112107e790c3e478532329e434f00b/python_dotenv-1.0.0-py3-none-any.whl (19 kB)
Collecting regex<2024.0.0,>=2023.6.3 (from semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/05/9e/80c20f1151432a6025690c9c2037053039b028a7b236fa81d7e7ac9dec60/regex-2023.12.25-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (773 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 773.4/773.4 kB 2.8 MB/s eta 0:00:00
Collecting mypy-extensions>=0.3.0 (from typing-inspect==0.8.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/e2/5d3f6ada4297caebe1a2add3b126fe800c96f56dbe5d1988a2cbe0b267aa/mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
Collecting pyjwt~=2.8.0 (from zhipuai==2.0.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2b/4f/e04a8067c7c96c364cef7ef73906504e2f40d690811c021e1a1901473a19/PyJWT-2.8.0-py3-none-any.whl (22 kB)
Collecting proto-plus<2.0.0dev,>=1.22.3 (from google-ai-generativelanguage==0.4.0->google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ad/41/7361075f3a31dcd05a6a38cfd807a6eecbfb6dbfe420d922cd400fc03ac1/proto_plus-1.23.0-py3-none-any.whl (48 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.8/48.8 kB 1.4 MB/s eta 0:00:00
Collecting pyarrow>=12 (from pylance==0.9.0->lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f3/7a/58a68be90992ea3c1d7a6b578d7ee8bc103213c4902d10fed25e1845f0d4/pyarrow-15.0.0-cp39-cp39-manylinux_2_28_x86_64.whl (38.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.3/38.3 MB 3.2 MB/s eta 0:00:00
Collecting greenlet==3.0.3 (from playwright>=1.26->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/af/05/b7e068070a6c143f34dfcd7e9144684271b8067e310f6da68269580db1d8/greenlet-3.0.3-cp39-cp39-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (614 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 614.3/614.3 kB 3.2 MB/s eta 0:00:00
Collecting pyee==11.0.1 (from playwright>=1.26->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7a/40/bfe7fd2cb55ca7dbb4c56e9f0060567c3e84dd9bdf8a782261f1d2d7c32f/pyee-11.0.1-py3-none-any.whl (15 kB)
Collecting et-xmlfile (from openpyxl->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/96/c2/3dd434b0108730014f1b96fd286040dc3bcb70066346f7e01ec2ac95865f/et_xmlfile-1.1.0-py3-none-any.whl (4.7 kB)
Collecting idna>=2.8 (from anyio<5,>=3.5.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c2/e7/a82b05cf63a603df6e68d59ae6a68bf5064484a0718ea5033660af4b54a9/idna-3.6-py3-none-any.whl (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.6/61.6 kB 2.0 MB/s eta 0:00:00
Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/57/e9/4368d49d3b462da16a3bac976487764a84dd85cef97232c7bd61f5bdedf3/marshmallow-3.20.2-py3-none-any.whl (49 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.4/49.4 kB 1.8 MB/s eta 0:00:00
Collecting sqlparse>=0.3.1 (from Django>=3.2->channels==4.0.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/5a/66d7c9305baa9f11857f247d4ba761402cea75db6058ff850ed7128957b7/sqlparse-0.4.4-py3-none-any.whl (41 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.2/41.2 kB 1.5 MB/s eta 0:00:00
Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython==3.1.40->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a7/a5/10f97f73544edcdef54409f1d839f6049a0d79df68adbc1ceb24d1aaca42/smmap-5.0.1-py3-none-any.whl (24 kB)
Collecting googleapis-common-protos<2.0.dev0,>=1.56.2 (from google-api-core->google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f0/43/c9d8f75ddf08e2a0a27db243c13a700c3cc7ec615b545b697cf6f715ad92/googleapis_common_protos-1.62.0-py2.py3-none-any.whl (228 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 228.7/228.7 kB 3.3 MB/s eta 0:00:00
Collecting pyasn1-modules>=0.2.1 (from google-auth->google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cd/8e/bea464350e1b8c6ed0da3a312659cb648804a08af6cacc6435867f74f8bd/pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 2.2 MB/s eta 0:00:00
Collecting rsa<5,>=3.1.4 (from google-auth->google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/49/97/fa78e3d2f65c02c8e1268b9aba606569fe97f6c8f7c2d74394553347c145/rsa-4.9-py3-none-any.whl (34 kB)
Collecting certifi (from httpx<1,>=0.23.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ba/06/a07f096c664aeb9f01624f858c3add0a4e913d6c96257acb4fce61e7de14/certifi-2024.2.2-py3-none-any.whl (163 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 163.8/163.8 kB 1.5 MB/s eta 0:00:00
Collecting httpcore==1.* (from httpx<1,>=0.23.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/56/ba/78b0a99c4da0ff8b0f59defa2f13ca4668189b134bd9840b6202a93d9a0f/httpcore-1.0.2-py3-none-any.whl (76 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.9/76.9 kB 2.6 MB/s eta 0:00:00
Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx<1,>=0.23.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/95/04/ff642e65ad6b90db43e668d70ffb6736436c7ce41fcc549f4e9472234127/h11-0.14.0-py3-none-any.whl (58 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 2.0 MB/s eta 0:00:00
Collecting h2<5,>=3 (from httpx[http2]>=0.14.0->qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/e5/db6d438da759efbb488c4f3fbdab7764492ff3c3f953132efa6b9f0e9e53/h2-4.1.0-py3-none-any.whl (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.5/57.5 kB 1.8 MB/s eta 0:00:00
Collecting parso<0.9.0,>=0.8.3 (from jedi>=0.16->ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/05/63/8011bd08a4111858f79d2b09aad86638490d62fbf881c44e434a6dfca87b/parso-0.8.3-py2.py3-none-any.whl (100 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 2.2 MB/s eta 0:00:00
Collecting jsonpointer>=1.9 (from jsonpatch<2.0,>=1.33->langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/12/f6/0232cc0c617e195f06f810534d00b74d2f348fe71b2118009ad8ad31f878/jsonpointer-2.4-py2.py3-none-any.whl (7.8 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=2.6->nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ee/07/44bd408781594c4d0a027666ef27fab1e441b109dc3b76b4f836f8fd04fe/jsonschema_specifications-2023.12.1-py3-none-any.whl (18 kB)
Collecting referencing>=0.28.4 (from jsonschema>=2.6->nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/90/10/1c92edb0a0a14b67ff825bc338e74bc49ab27d3f3bae3f9a02838cba546f/referencing-0.33.0-py3-none-any.whl (26 kB)
Collecting rpds-py>=0.7.1 (from jsonschema>=2.6->nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c2/e9/190521d63b504c12bdcffb27ea6aaac1dbb2521be983c3a2a0ab4a938b8c/rpds_py-0.17.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 4.2 MB/s eta 0:00:00
Collecting importlib-metadata>=4.8.3 (from jupyter-client>=6.1.12->ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c0/8b/d8427f023c081a8303e6ac7209c16e6878f2765d5b59667f3903fbcfd365/importlib_metadata-7.0.1-py3-none-any.whl (23 kB)
Collecting platformdirs>=2.5 (from jupyter-core!=5.0.*,>=4.12->ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/55/72/4898c44ee9ea6f43396fbc23d9bfaf3d06e01b83698bdf2e4c919deceb7c/platformdirs-4.2.0-py3-none-any.whl (17 kB)
Collecting langsmith<0.1.0,>=0.0.70 (from langchain==0.0.352->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/94/99/762b50b229516dd133e09c16213736b88d50d75e262b976e20cc244280ed/langsmith-0.0.87-py3-none-any.whl (55 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.4/55.4 kB 2.2 MB/s eta 0:00:00
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich==13.6.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Collecting pymongo<5,>=4.5 (from motor<4.0.0,>=3.3.1->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/68/f3/f0909bd0498c1e34d9fbdc432feb55cb25d58aeb1a2022be9827afabdc61/pymongo-4.6.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (676 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 676.3/676.3 kB 4.9 MB/s eta 0:00:00
Collecting isodate (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b6/85/7882d311924cbcfc70b1890780763e36ff0b140c7e51c110fc59a532f087/isodate-0.6.1-py2.py3-none-any.whl (41 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.7/41.7 kB 1.3 MB/s eta 0:00:00
Collecting jsonschema-spec<0.3.0,>=0.2.3 (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d9/a2/7759a4268e1d6d74559de8fb5be6c77d621b822ae64d28ab4f7467c22f63/jsonschema_spec-0.2.4-py3-none-any.whl (14 kB)
Collecting more-itertools (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/50/e2/8e10e465ee3987bb7c9ab69efb91d867d93959095f4807db102d07995d94/more_itertools-10.2.0-py3-none-any.whl (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.0/57.0 kB 2.2 MB/s eta 0:00:00
Collecting openapi-schema-validator<0.7.0,>=0.6.0 (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b3/dc/9aefae8891454130968ff079ece851d1ae9ccf6fb7965761f47c50c04853/openapi_schema_validator-0.6.2-py3-none-any.whl (8.8 kB)
Collecting openapi-spec-validator<0.8.0,>=0.7.1 (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2b/4d/e744fff95aaf3aeafc968d5ba7297c8cda0d1ecb8e3acd21b25adae4d835/openapi_spec_validator-0.7.1-py3-none-any.whl (38 kB)
Collecting parse (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ce/f0/30fe1494f1910ad3ea40639b13ac48cdb16a8600e8861cbfc2c560661ddf/parse-1.20.1-py2.py3-none-any.whl (20 kB)
Collecting werkzeug (from openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c3/fc/254c3e9b5feb89ff5b9076a23218dafbc99c96ac5941e900b71206e6313b/werkzeug-3.0.1-py3-none-any.whl (226 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226.7/226.7 kB 5.2 MB/s eta 0:00:00
Collecting ptyprocess>=0.5 (from pexpect>4.3->ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/22/a6/858897256d0deac81a172289110f31629fc4cee19b6f01283303e18c8db3/ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
Collecting chardet>=3.0 (from prance<24.0.0.0,>=23.6.21.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/38/6f/f5fbc992a329ee4e0f288c1fe0e2ad9485ed064cac731ed2fe47dcc38cbf/chardet-5.2.0-py3-none-any.whl (199 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 199.4/199.4 kB 4.8 MB/s eta 0:00:00
Collecting ruamel.yaml>=0.17.10 (from prance<24.0.0.0,>=23.6.21.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/73/67/8ece580cc363331d9a53055130f86b096bf16e38156e33b1d3014fffda6b/ruamel.yaml-0.18.6-py3-none-any.whl (117 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.8/117.8 kB 2.8 MB/s eta 0:00:00
Collecting wcwidth (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fd/84/fd2ba7aafacbad3c4201d395674fc6348826569da3c0937e75505ead3528/wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)
Collecting py<2.0.0,>=1.4.26 (from retry>=0.9.2->lancedb==0.4.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f6/f0/10642828a8dfb741e5f3fbaac830550a518a775c7fff6f04a007259b0548/py-1.11.0-py2.py3-none-any.whl (98 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98.7/98.7 kB 3.4 MB/s eta 0:00:00
Collecting huggingface_hub<1.0,>=0.16.4 (from tokenizers>=0.13.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/28/03/7d3c7153113ec59cfb31e3b8ee773f5f420a0dd7d26d40442542b96675c3/huggingface_hub-0.20.3-py3-none-any.whl (330 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 330.1/330.1 kB 5.7 MB/s eta 0:00:00
Collecting executing>=1.2.0 (from stack-data->ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/80/03/6ea8b1b2a5ab40a7a60dc464d3daa7aa546e0a74d74a9f8ff551ea7905db/executing-2.0.1-py2.py3-none-any.whl (24 kB)
Collecting asttokens>=2.1.0 (from stack-data->ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/45/86/4736ac618d82a20d87d2f92ae19441ebc7ac9e7a581d7e58bbe79233b24a/asttokens-2.4.1-py2.py3-none-any.whl (27 kB)
Collecting pure-eval (from stack-data->ipython==8.17.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2b/27/77f9d5684e6bce929f5cfe18d6cfbe5133013c06cb2fbf5933670e60761d/pure_eval-0.2.2-py3-none-any.whl (11 kB)
Collecting grpcio-status<2.0.dev0,>=1.33.2 (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.0->google-ai-generativelanguage==0.4.0->google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/eb/97/e7dfe2d5566bca05f52af5d4f4a67ccb90878586d3cadbdf8de5a5d4be00/grpcio_status-1.60.1-py3-none-any.whl (14 kB)
Collecting hyperframe<7,>=6.0 (from h2<5,>=3->httpx[http2]>=0.14.0->qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d7/de/85a784bcc4a3779d1753a7ec2dee5de90e18c7bcf402e71b51fcf150b129/hyperframe-6.0.1-py3-none-any.whl (12 kB)
Collecting hpack<5,>=4.0 (from h2<5,>=3->httpx[http2]>=0.14.0->qdrant-client==1.7.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d5/34/e8b383f35b77c402d28563d2b8f83159319b509bc5f760b15d60b0abf165/hpack-4.0.0-py3-none-any.whl (32 kB)
Collecting filelock (from huggingface_hub<1.0,>=0.16.4->tokenizers>=0.13.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/81/54/84d42a0bee35edba99dee7b59a8d4970eccdd44b99fe728ed912106fc781/filelock-3.13.1-py3-none-any.whl (11 kB)
Collecting fsspec>=2023.5.0 (from huggingface_hub<1.0,>=0.16.4->tokenizers>=0.13.0->anthropic==0.8.1->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ad/30/2281c062222dc39328843bd1ddd30ff3005ef8e30b2fd09c4d2792766061/fsspec-2024.2.0-py3-none-any.whl (170 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.9/170.9 kB 4.2 MB/s eta 0:00:00
Collecting zipp>=0.5 (from importlib-metadata>=4.8.3->jupyter-client>=6.1.12->ipykernel==6.27.0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d9/66/48866fc6b158c81cc2bfecc04c480f105c6040e8b077bc54c634b4a67926/zipp-3.17.0-py3-none-any.whl (7.4 kB)
Collecting pathable<0.5.0,>=0.4.1 (from jsonschema-spec<0.3.0,>=0.2.3->openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5b/0a/acfb251ba01009d3053f04f4661e96abf9d485266b04a0a4deebc702d9cb/pathable-0.4.3-py3-none-any.whl (9.6 kB)
Collecting referencing>=0.28.4 (from jsonschema>=2.6->nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/be/8e/56d6f1e2d591f4d6cbcba446cac4a1b0dc4f584537e2071d9bcee8eeab6b/referencing-0.30.2-py3-none-any.whl (25 kB)
INFO: pip is looking at multiple versions of jsonschema-specifications to determine which version is compatible with other requirements. This could take a while.
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=2.6->nbformat==5.9.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d7/48/b62ccba8f4ac91817d6a11b340e63806175dafb10234a8cf7140bd389da5/jsonschema_specifications-2023.11.2-py3-none-any.whl (17 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/20/a9/384ec45013ab883d7c2bf120f2988682986fdead973decf0bae28a4523e7/jsonschema_specifications-2023.11.1-py3-none-any.whl (17 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1c/24/83349ac2189cc2435e84da3f69ba3c97314d3c0622628e55171c6798ed80/jsonschema_specifications-2023.7.1-py3-none-any.whl (17 kB)
Collecting rfc3339-validator (from openapi-schema-validator<0.7.0,>=0.6.0->openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7b/44/4e421b96b67b2daff264473f7465db72fbdf36a07e05494f50300cc7b0c6/rfc3339_validator-0.1.4-py2.py3-none-any.whl (3.5 kB)
Collecting jsonschema-path<0.4.0,>=0.3.1 (from openapi-spec-validator<0.8.0,>=0.7.1->openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7f/5a/f405ced79c55191e460fc6d17a14845fddf09f601e39cfcab28cc1d3ff1c/jsonschema_path-0.3.2-py3-none-any.whl (14 kB)
Collecting lazy-object-proxy<2.0.0,>=1.7.1 (from openapi-spec-validator<0.8.0,>=0.7.1->openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ab/be/d0a76dd4404ee68c7dd611c9b48e58b5c70ac5458e4c951b2c8923c24dd9/lazy_object_proxy-1.10.0-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (67 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.1/67.1 kB 2.2 MB/s eta 0:00:00
Collecting pyasn1<0.6.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth->google-generativeai==0.3.2->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d1/75/4686d2872bf2fc0b37917cbc8bbf0dd3a5cdb0990799be1b9cbf1e1eb733/pyasn1-0.5.1-py2.py3-none-any.whl (84 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.9/84.9 kB 3.1 MB/s eta 0:00:00
Collecting dnspython<3.0.0,>=1.16.0 (from pymongo<5,>=4.5->motor<4.0.0,>=3.3.1->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b6/83/4a684a63d395007670bc95c1947c07045fe66141574e2f7e9e347df8499a/dnspython-2.5.0-py3-none-any.whl (305 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 305.4/305.4 kB 5.5 MB/s eta 0:00:00
Collecting ruamel.yaml.clib>=0.2.7 (from ruamel.yaml>=0.17.10->prance<24.0.0.0,>=23.6.21.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7c/b2/389b345a60131593028b0263fddaa580edb4081697a3f3aa1f168f67519f/ruamel.yaml.clib-0.2.8-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl (562 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 562.1/562.1 kB 5.2 MB/s eta 0:00:00
Collecting MarkupSafe>=2.1.1 (from werkzeug->openapi_core<0.19.0,>=0.18.0->semantic-kernel==0.4.3.dev0->metagpt==0.7.0)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5f/5a/360da85076688755ea0cceb92472923086993e86b5613bbae9fbc14136b0/MarkupSafe-2.1.5-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
WARNING: The candidate selected for download or install is a yanked version: 'ipykernel' candidate (version 6.27.0 at https://pypi.tuna.tsinghua.edu.cn/packages/1e/36/1c316a31d42d323de41616c98e353bd1db1b716980c90929832de4755f80/ipykernel-6.27.0-py3-none-any.whl#sha256=4388caa3c2cba0a381e20d289545e88a8aef1fe57a884d4c018718ec8c23c121 (from https://pypi.tuna.tsinghua.edu.cn/simple/ipykernel/) (requires-python:>=3.8))
Reason for being yanked: broke %edit magic
Building wheels for collected packages: fire, gitignore-parser, python_docx, ta
  Building wheel for fire (setup.py) ... done
  Created wheel for fire: filename=fire-0.4.0-py2.py3-none-any.whl size=115927 sha256=47363e3c41fa7c4d6556f7df13d435678b991f23a95ddaca01a25ec679e35616
  Stored in directory: /home/neon/.cache/pip/wheels/4b/10/de/aeaeb6a631b5c89fa8d339e6b7fd9d3e0271c60d27b8f3ee15
  Building wheel for gitignore-parser (pyproject.toml) ... done
  Created wheel for gitignore-parser: filename=gitignore_parser-0.1.9-py3-none-any.whl size=4956 sha256=4d7d51dce5ca85e77dc58d11755091b47749d2a57da252c89640cd7b146855fc
  Stored in directory: /home/neon/.cache/pip/wheels/07/8c/08/e9e8f48c32b18b23c9e92ee82db9149ff4da87dbff571561fd
  Building wheel for python_docx (setup.py) ... done
  Created wheel for python_docx: filename=python_docx-0.8.11-py3-none-any.whl size=184488 sha256=843c7290d970146ffa83da0fcfbdcdf0dd0c4627c20623b17605266fc9ea5869
  Stored in directory: /home/neon/.cache/pip/wheels/39/ca/c1/d3e7abe5ce2e8423382d816e689c056bc26590f48fad8f20ac
  Building wheel for ta (setup.py) ... done
  Created wheel for ta: filename=ta-0.10.2-py3-none-any.whl size=29088 sha256=fc1120e22eba3e3435040a1e99d135f4f3d0e92b09689f3feb4eddb723d6c19b
  Stored in directory: /home/neon/.cache/pip/wheels/53/e0/94/d58d9ed165e75be3bba7d3d6ad9609cc56e2a9a01b3d7ada93
Successfully built fire gitignore-parser python_docx ta
Installing collected packages: wcwidth, ratelimiter, pytz, pure-eval, ptyprocess, parse, imap_tools, gitignore-parser, fastjsonschema, faiss_cpu, zipp, wrapt, widgetsnbextension, websockets, websocket-client, urllib3, tzdata, typing-extensions, traitlets, tqdm, tornado, threadpoolctl, termcolor, tenacity, sqlparse, soupsieve, socksio, sniffio, smmap, six, setuptools, semver, ruamel.yaml.clib, rpds-py, regex, pyzmq, PyYAML, python-dotenv, pyjwt, pygments, pyasn1, py, psutil, protobuf, prompt-toolkit, portalocker, platformdirs, Pillow, pexpect, pathable, parso, packaging, overrides, numpy, networkx, nest-asyncio, mypy-extensions, multidict, more-itertools, mdurl, MarkupSafe, lxml, loguru, lazy-object-proxy, jupyterlab-widgets, jsonpointer, joblib, idna, hyperframe, hpack, h11, grpcio, greenlet, fsspec, frozenlist, filelock, executing, exceptiongroup, et-xmlfile, dnspython, distro, decorator, debugpy, click, charset-normalizer, chardet, certifi, camel-converter, cachetools, attrs, async-timeout, annotated-types, aiofiles, yarl, werkzeug, typing-inspect, typer, SQLAlchemy, scipy, ruamel.yaml, rsa, rfc3339-validator, retry, requests, referencing, python_docx, python-dateutil, pymongo, pyee, pydantic-core, pyasn1-modules, pyarrow, proto-plus, openpyxl, matplotlib-inline, marshmallow, markdown-it-py, jupyter-core, jsonpatch, jedi, isodate, importlib-metadata, httpcore, h2, grpcio-tools, googleapis-common-protos, gitdb, fire, deprecation, comm, beautifulsoup4, asttokens, asgiref, anytree, anyio, aiosignal, aioredis, tiktoken, stack-data, scikit_learn, rich, pylance, pydantic, prance, playwright, pandas, motor, libcst, jupyter-client, jsonschema-specifications, jsonschema-spec, jsonschema-path, huggingface_hub, httpx, grpcio-status, google-auth, gitpython, Django, dataclasses-json, aiohttp, zhipuai, tokenizers, ta, openai, langsmith, lancedb, jsonschema, ipython, google-api-core, channels, qdrant-client, openapi-schema-validator, nbformat, meilisearch, langchain-core, ipywidgets, ipykernel, anthropic, openapi-spec-validator, nbclient, langchain-community, google-ai-generativelanguage, openapi_core, langchain, google-generativeai, semantic-kernel, metagpt
  Attempting uninstall: setuptools
    Found existing installation: setuptools 69.0.3
    Uninstalling setuptools-69.0.3:
      Successfully uninstalled setuptools-69.0.3
  Running setup.py develop for metagpt
Successfully installed Django-4.2.10 MarkupSafe-2.1.5 Pillow-10.2.0 PyYAML-6.0.1 SQLAlchemy-2.0.26 aiofiles-23.2.1 aiohttp-3.8.4 aioredis-2.0.1 aiosignal-1.3.1 annotated-types-0.6.0 anthropic-0.8.1 anyio-4.2.0 anytree-2.12.1 asgiref-3.7.2 asttokens-2.4.1 async-timeout-4.0.3 attrs-23.2.0 beautifulsoup4-4.12.2 cachetools-5.3.2 camel-converter-3.1.1 certifi-2024.2.2 channels-4.0.0 chardet-5.2.0 charset-normalizer-3.3.2 click-8.1.7 comm-0.2.1 dataclasses-json-0.6.4 debugpy-1.8.1 decorator-5.1.1 deprecation-2.1.0 distro-1.9.0 dnspython-2.5.0 et-xmlfile-1.1.0 exceptiongroup-1.2.0 executing-2.0.1 faiss_cpu-1.7.4 fastjsonschema-2.19.1 filelock-3.13.1 fire-0.4.0 frozenlist-1.4.1 fsspec-2024.2.0 gitdb-4.0.11 gitignore-parser-0.1.9 gitpython-3.1.40 google-ai-generativelanguage-0.4.0 google-api-core-2.17.0 google-auth-2.27.0 google-generativeai-0.3.2 googleapis-common-protos-1.62.0 greenlet-3.0.3 grpcio-1.60.1 grpcio-status-1.60.1 grpcio-tools-1.60.1 h11-0.14.0 h2-4.1.0 hpack-4.0.0 httpcore-1.0.2 httpx-0.26.0 huggingface_hub-0.20.3 hyperframe-6.0.1 idna-3.6 imap_tools-1.5.0 importlib-metadata-7.0.1 ipykernel-6.27.0 ipython-8.17.2 ipywidgets-8.1.1 isodate-0.6.1 jedi-0.19.1 joblib-1.3.2 jsonpatch-1.33 jsonpointer-2.4 jsonschema-4.21.1 jsonschema-path-0.3.2 jsonschema-spec-0.2.4 jsonschema-specifications-2023.7.1 jupyter-client-8.6.0 jupyter-core-5.7.1 jupyterlab-widgets-3.0.10 lancedb-0.4.0 langchain-0.0.352 langchain-community-0.0.19 langchain-core-0.1.22 langsmith-0.0.87 lazy-object-proxy-1.10.0 libcst-1.0.1 loguru-0.6.0 lxml-5.1.0 markdown-it-py-3.0.0 marshmallow-3.20.2 matplotlib-inline-0.1.6 mdurl-0.1.2 meilisearch-0.21.0 metagpt-0.7.0 more-itertools-10.2.0 motor-3.3.2 multidict-6.0.5 mypy-extensions-1.0.0 nbclient-0.9.0 nbformat-5.9.2 nest-asyncio-1.6.0 networkx-3.2.1 numpy-1.26.4 openai-1.6.0 openapi-schema-validator-0.6.2 openapi-spec-validator-0.7.1 openapi_core-0.18.2 openpyxl-3.1.2 overrides-7.7.0 packaging-23.2 pandas-2.0.3 parse-1.20.1 parso-0.8.3 pathable-0.4.3 pexpect-4.9.0 platformdirs-4.2.0 playwright-1.41.2 portalocker-2.8.2 prance-23.6.21.0 prompt-toolkit-3.0.43 proto-plus-1.23.0 protobuf-4.25.2 psutil-5.9.8 ptyprocess-0.7.0 pure-eval-0.2.2 py-1.11.0 pyarrow-15.0.0 pyasn1-0.5.1 pyasn1-modules-0.3.0 pydantic-2.5.3 pydantic-core-2.14.6 pyee-11.0.1 pygments-2.17.2 pyjwt-2.8.0 pylance-0.9.0 pymongo-4.6.1 python-dateutil-2.8.2 python-dotenv-1.0.0 python_docx-0.8.11 pytz-2024.1 pyzmq-25.1.2 qdrant-client-1.7.0 ratelimiter-1.2.0.post0 referencing-0.30.2 regex-2023.12.25 requests-2.31.0 retry-0.9.2 rfc3339-validator-0.1.4 rich-13.6.0 rpds-py-0.17.1 rsa-4.9 ruamel.yaml-0.18.6 ruamel.yaml.clib-0.2.8 scikit_learn-1.3.2 scipy-1.12.0 semantic-kernel-0.4.3.dev0 semver-3.0.2 setuptools-65.6.3 six-1.16.0 smmap-5.0.1 sniffio-1.3.0 socksio-1.0.0 soupsieve-2.5 sqlparse-0.4.4 stack-data-0.6.3 ta-0.10.2 tenacity-8.2.2 termcolor-2.4.0 threadpoolctl-3.2.0 tiktoken-0.5.2 tokenizers-0.15.2 tornado-6.4 tqdm-4.65.0 traitlets-5.14.1 typer-0.9.0 typing-extensions-4.9.0 typing-inspect-0.8.0 tzdata-2024.1 urllib3-1.26.18 wcwidth-0.2.13 websocket-client-1.6.2 websockets-12.0 werkzeug-3.0.1 widgetsnbextension-4.0.10 wrapt-1.15.0 yarl-1.9.4 zhipuai-2.0.1 zipp-3.17.0
```
