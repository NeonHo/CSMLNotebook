```Python
quant_single_onnx_network(args.config_path,calib_dataset,hm_yolov3.model,with_label=True,analyze=False, device=args.device)
```

In this API, we need to note that the arguments we need to set:
| arguments | mean |
| ---- | ---- |
| `with_label`  | If it is `True`, the API will know that our dataset has input with label |