COMMANDS:

time uv run python vocos_export.py --checkpoint models/vocos/last.pt --output-dir runs/litert_vocos_pixel10_gpu_fp16 --sample-count 3 --pixel10-fp16-gpu --dynamic-frames --pixel10-multisignature-static

time uv run python wavehax_export.py --checkpoint models/wavehax/last.pt --output-dir runs/wavehax --sample-count 3 --dynamic-frames --multisignature-static --export-all-sample-lengths

time uv run python android/run_plan_bench.py --family wavehax --adb /mnt/c/Users/genei/Downloads/platform-tools/adb.exe --log-wait-seconds 80 --case-timeout-seconds 80 --log-wait-seconds-1754 100 --case-timeout-seconds-1754 100
