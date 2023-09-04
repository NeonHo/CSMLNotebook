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
If we forward a port from `localhost:5174` to remote port: `5174`, as soon as we visit `localhost:5174`, VSCode will post the request to the port `5174` on the remote server.

If we do this, our local PC will forward the request from the local preview front end to the remote back end by using VSCode Remote SSH.

The local PC will afford the load of front end and be responsible for forwarding the request through VSCode to the remote back end.
The load of back end will be afforded by the remote server.


