# Kali Linux on Windows Subsystem for Linux (WSL)

This doc describes how to install Kali Linux on WSLv2.

## WSL Docs

https://docs.microsoft.com/en-us/windows/wsl/about

## Install

** Requires Windows 10  build 18917 or higher (Windows Insider program) **
 https://docs.microsoft.com/en-us/windows/wsl/wsl2-install

### Enable WSL
Run these commands from an elevated powershell prompt then reboot:
```
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
```

### WSL v2
Set v2 as the default for all new VM's:
```
wsl --set-default-version 2
```

### Install Kali
Download the appx installer for Kali:
```
Invoke-WebRequest -Uri https://aka.ms/wsl-kali-linux-new -OutFile Kali.appx -UseBasicParsing
```

Run the installer:
```
Add-AppxPackage .\kali.appx
```

Follow the prompts to setup a new user.

##  Change Install Location

By default your install will be located here:
```
C:\Users\<user>\AppData\Local\Packages\KaliLinux.54290C8133FEE_ey8k8hqnwqnmg
```

To move it, we'll export it, uninstall the appx and import to a new location.
You won't be able to just type "kali" and run it, but you can have multiple instances of kali and/or backups.

### Export Kali

Setup a directory somewhere you want to store your exports and VM's.  In my examples D:\VM:
```
 wsl --export kali-linux d:\wsl\exported\kali-linux.wsl
```

Go to add/remove programs and uninstall "Kali Linux". This will remove it from the original path and unregister it.
List all WSL VM's to ensure it's gone:
```
wsl -l
```

### Import a New VM
Import your exported instance, choosing a new name and location:
```
wsl --import kali-linux d:\wsl\kali-linux D:\wsl\exported\kali-linux.wsl
```

Now you can run this with:
```
wsl.exe ~ -d kali-linux -u <username>
```

You can run this again to install a second instance:
```
wsl --import kali2 d:\wsl\kali2 D:\wsl\exported\kali-linux.wsl
wsl.exe ~ -d kali2 -u <username>
```

## WSL Common Commands

Run a specific distribution:

```
wsl --distribution, -d <Distro>
```

List registered:

```
wsl -l all| running -v
```

```
wsl -t <Distro>
wsl --unregister <Distro>
```

## Run Desktop Apps from Kali on WSL2

### Install XFCE Desktop
Install a desktop for Kali, in this case XFCE:
```
sudo apt-get install kali-desktop-xfce
```

### Setup client
I use X410 from the Windows Store as client.  Start X410 in Windowed Apps mode with "Public Access" enabled.
On Kali, export desktop to your Windows client's IP address (localhost won't work):
```
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2; exit;}'):0.0
```

Run anything like burpsuite, firefox, maltego, zenmap and use it from Windows.





