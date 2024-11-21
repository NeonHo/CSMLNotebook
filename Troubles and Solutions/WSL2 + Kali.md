[Operation Reference](https://juejin.cn/post/6921700500450574350)
Bug happens when install Kali Linux.
[Solution](https://blog.csdn.net/weixin_43891732/article/details/133672607)
```
username: neon
password: hnfq150
```
Graphic Frame:
https://zhuanlan.zhihu.com/p/681824309

[Win-Hex](https://www.kali.org/docs/wsl/win-kex/)

To start Win-KeX in Window mode with sound support, run either:

- Inside of Kali WSL: `kex --win -s`
- On Window’s command prompt: `wsl -d kali-linux kex --win -s`

==If it doesn't work, simply use `sudo kex` instead of `kex` seems to work well.==
[reference](https://github.com/microsoft/WSL/discussions/6675)


# Install Chrome
To install Google Chrome from the terminal, get the DEB file using the wget command:

```
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
```

Now you can use dpkg to install Chrome from the downloaded DEB file:

```
sudo dpkg -i google-chrome-stable_current_amd64.deb
```