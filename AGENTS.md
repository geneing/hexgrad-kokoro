
uv add iree-base-compiler[onnx] iree-base-runtime

uv run iree-import-onnx onnx/bert.onnx --opset-version 17 -o onnx/bert.mlir
uv run iree-import-onnx onnx/duration_predictor.onnx --opset-version 17 -o onnx/duration_predictor.mlir
uv run iree-import-onnx onnx/text_encoder.onnx --opset-version 17 -o onnx/text_encoder.mlir

uv run iree-import-onnx onnx/bert.fp16.onnx --opset-version 17 -o onnx/bert.fp16.mlir
uv run iree-import-onnx onnx/duration_predictor.fp16.onnx --opset-version 17 -o onnx/duration_predictor.fp16.mlir
uv run iree-import-onnx onnx/text_encoder.fp16.onnx --opset-version 17 -o onnx/text_encoder.fp16.mlir


uv run iree-compile --iree-hal-target-device=vulkan --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target valhall -o onnx/bert.fp16.vmfb onnx/bert.fp16.mlir
  
  
  
  
  
  --iree-hal-dump-executable-files-to=dump/