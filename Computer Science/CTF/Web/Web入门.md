# Web 1
See the source code of the web-page.
![[Pasted image 20240812171512.png]]
![[Pasted image 20240812171742.png]]
# Web 2
![[Pasted image 20240812171721.png]]
view-source是一种协议，早期基本上每个浏览器都支持这个协议。后来Microsoft考虑安全性，对于WindowsXP pack2以及更高版本以后IE就不再支持此协议。但是这个方法在FireFox和Chrome浏览器都还可以使用。 如果要在IE下查看源代码,只能使用查看中的"查看源代码"命令.以前的使用方法：在浏览器地址栏中输入 view-source: sURL
![[Pasted image 20240812172744.png]]
# Web 3
确实是抓个包就行
![[Pasted image 20240812173754.png]]
![[Pasted image 20240812173730.png]]
# Web 4
![[Pasted image 20240812193351.png]]
[关于网站robots协议，看这篇就够了 - SEO轻松学的文章 - 知乎](https://zhuanlan.zhihu.com/p/342575122)
![[Pasted image 20240812194206.png]]
![[Pasted image 20240812194235.png]]
# Web 5
![[Pasted image 20240812194926.png]]
直接访问`{URL}/index.phps`。
![[Pasted image 20240812195224.png]]

phps导致源码泄露  
phps文件就是php的源代码文件，通常用于提供给用户（访问者）直接通过Web浏览器查看php代码的内容。  
因为用户无法直接通过Web浏览器“看到”php文件的内容，所以需要用phps文件代替。  
并不是所有的php文件都存在.phps后缀，不是默认带有，只会在特殊情况下存在  
详情见百度  
解：  
利用目录扫描工具，例如御剑，扫描出靶场目录下有个index.php（一般可以猜出来），将index.php改为index.phps访问，自动下载了index.php的源码，打开获取flag
## `dirsearch`
http://t.csdnimg.cn/2HdOt
## 御剑
其实也搜不出什么。
# Web 6
![[Pasted image 20240812211156.png]]
总算通过dirsearch搜到一个[[#`dirsearch`]]

![[Pasted image 20240812211926.png]]
# Web 7
![[Pasted image 20240812212138.png]]
![[Pasted image 20240812213245.png]]
![[Pasted image 20240812213338.png]]
# Web 8
![[Pasted image 20240812213937.png]]

![[Pasted image 20240812213911.png]]

![[Pasted image 20240812213704.png]]