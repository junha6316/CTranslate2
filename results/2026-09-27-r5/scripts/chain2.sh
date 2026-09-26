#!/bin/bash
cd /opt/real/r6/r5t
echo "=== uaf load-only control (1a7abfb, no decode) $(date +%T)"
( source /opt/real/r6/r5t/env.sh ref
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --track-stream-ordered-races use-after-free \
    python -c "import ctranslate2; m = ctranslate2.models.Whisper('/opt/models/faster-whisper-small', device='cuda', compute_type='float16'); print('loaded')" > gsan_uaf_ref_loadonly.log 2>&1 )
echo "load-only: $(grep 'ERROR SUMMARY' gsan_uaf_ref_loadonly.log | head -1)"
./gtokens.sh > gtokens.log 2>&1
./glong.sh > glong.log 2>&1
python3 glong_report.py gl > glong_report.txt 2>&1
./gmemgather.sh > gmemgather.log 2>&1
./glarge.sh > glarge.log 2>&1
echo "=== CHAIN2DONE $(date +%T)"
