# Install
```Zsh
sudo npm install -g openclaw@latest
openclaw onboard --install-daemon
```
If the node is not the latest stable version:
```
nvm install --lts
nvm use --lts 
```

重新进入下载页面，进行配置。
这里我用OpenRouter配置了API Key
使用Claude sonnet-4.5


让openclaw停下：
```zsh
openclaw gateway stop
```

最大token太大了，会导致对话不回复。
