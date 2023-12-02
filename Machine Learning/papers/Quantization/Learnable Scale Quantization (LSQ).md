Training the quantization scale with the parameters.

Update the scale with backward propagation.

![[Drawing 2023-12-02 17.52.08.excalidraw]]

In the LSQ, the gradients are changed suddenly too in the transition point, so it will be more similar to the real quantization than the ==QIL== and the ==PACT==.

