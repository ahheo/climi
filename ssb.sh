#!/bin/bash
#
me=$(basename "$0")
while test $# -gt 0; do
 case "$1" in
  -h|--help)
   echo "$me - sbatch *STR*.sh"
   echo " "
   echo "$me [options] STR"
   echo " "
   echo "options:"
   echo "-h, --help                show brief help"
   echo "-s, --sleep=SLP(s)        sleep SLP seconds after each sbatch"
   echo "-v                        verbose mode"
   exit 0
   ;;
  -s|--sleep)
   shift
   if test $# -gt 0; then
    SLP_="$1"
    shift
   fi
   ;;
  --sleep*)
   SLP_=$(echo $1 | sed -e 's/^[^=]*=//g')
   shift
   ;;
  -v)
   v_='-v'
   shift
   ;;
  *)
   STR_="$1"
   shift
   ;;
 esac
done

if [ "${STR_}" ]; then
 for i in [^\.]*${STR_}*.sh
 do
  if [ -f "$i" ]; then
   if [ "${v_}" ]; then
    echo 'sbatch '$i
   fi
   chmod +x $i
   sbatch $i
   if [ "${SLP_}" ]; then
    sleep "${SLP_}"s
   fi
  else
   "not a file; SKIP!"
  fi
 done
else
 echo 'STR not specified; SKIP!'
fi
