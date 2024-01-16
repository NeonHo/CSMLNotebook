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
	- 