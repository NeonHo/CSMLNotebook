# Introduction
VSCode Remote SSH make our PC connect to the remote Sever, so that we can develop on the remote sever by using our PC.
# Forward a Port
![[Pasted image 20230904150557.png]]
When developing a light service, we may need to preview the effects locally.
However, our shell, environments, code, etc. are all on the remote server.
Therefore, [[HTTP server]] is also on the remote server.

If we need to preview the effects locally, we need to forward a port to access the running services locally.

for example:
![[Pasted image 20230904151633.png]]
If we forward a port from `localhost:5174` to remote port: `5174`, as soon as we visit th