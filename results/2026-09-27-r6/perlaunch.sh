#!/bin/bash
# Per-launch gather initcheck error totals: perlaunch.sh LIB PYDIR TAG N [ENV...]
LIB=$1; PYD=$2; TAG=$3; N=$4; shift 4
out=/opt/real/r6/san/perlaunch_$TAG.txt; : > $out
for k in $(seq 0 $N); do
  PL=100 SANX="--show-backtrace no -s $k -c 1" bash /opt/real/r6/sanone.sh $LIB $PYD pl_$TAG b5 init_gather "$@" > /dev/null 2>&1
  f=/opt/real/r6/san/pl_${TAG}_b5_init_gather.log
  rec=$(grep -o "ERROR SUMMARY: [0-9]*" $f | grep -o "[0-9]*$"); sk=$(grep -o "[0-9]* errors were skipped" $f | grep -o "^[0-9]*")
  first=$(grep -m1 "by thread" $f | sed 's/.*thread //'); addr=$(grep -m1 "Address" $f | awk '{print $NF}')
  echo "$k $(( ${rec:-0} + ${sk:-0} )) $first $addr" >> $out
done
echo PLDONE >> $out
