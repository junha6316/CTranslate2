#!/bin/bash
cd /opt/real/r6
echo "== ggt start $(date +%T)"
bash usepy.sh base; bash ggt.sh base base
bash usepy.sh r6; bash ggt.sh r6s2 s2
echo "GGT_DONE $(date +%T)"
