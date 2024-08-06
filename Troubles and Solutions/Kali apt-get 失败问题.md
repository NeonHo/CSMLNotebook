我们需要微Kali的apt配置文件进行修改。
https://ultahost.com/knowledge-base/fix-unable-to-locate-package-kali-linux/

Step 4: Check Your Repositories
Verify your /etc/apt/sources.list file contains the correct entries for Kali Linux repositories:
```Bash
cat /etc/apt/sources.list
```
The official Kali Linux repository should look something like this:
```Shell
deb https://http.kali.org/kali kali-rolling main contrib non-free
```
If it’s missing, add it by editing the `sources.list` file using a text editor like Nano:
```
sudo nano /etc/apt/sources.list
```
Add the correct repository lines then save the file and exit.
