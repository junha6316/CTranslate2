#!/bin/bash
# FA2 tail-read check on flash_b5: initcheck totals (recorded+skipped) with gather-only vs
# gather+flash_fwd instrumentation, s1 (r6) and base. Output /opt/real/r6/san/fa2chk.txt
cd /opt/real/r6
out=san/fa2chk.txt; : > $out
tot() { local f=$1; local r=$(grep -m1 -o "ERROR SUMMARY: [0-9]*" $f | grep -o "[0-9]*$"); local s=$(grep -o "[0-9]* errors were skipped" $f | grep -o "^[0-9]*"); echo $(( ${r:-0} + ${s:-0} )); }
for t in r6:-:s1 base:/opt/real/r6/pybase:base; do IFS=: read L P T <<< "$t"
  for re in "gather_rows_kernel" "gather_rows_kernel|flash_fwd"; do
    tagx=fa_${T}_$(echo $re | tr -c 'a-z\n' '_' | cut -c1-12)
    KVRE="regex=$re" TMO=1800 PL=100 SANX="--show-backtrace no" bash sanone.sh $L $P $tagx flash_b5 init_kv > /dev/null 2>&1
    f=san/${tagx}_flash_b5_init_kv.log
    echo "$T [$re] total=$(tot $f) $(grep -h SANCELL $f) $(tail -1 $f) kernels: $(grep -o 'at [a-zA-Z_:]*' $f | sort | uniq -c | tr '\n' ' ')" >> $out
  done
done
echo FA2DONE >> $out
