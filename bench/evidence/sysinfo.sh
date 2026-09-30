#!/usr/bin/env bash
# System information for a benchmark record. Usage: sysinfo.sh OUTFILE [--hash-model]
. "$(dirname "$0")/../lib.sh"; need MODEL MTP_MODEL
O=${1:-sysinfo.txt}
MDIR=$(dirname "$MODEL")
{
  echo "## date";      date -Is
  echo "## kernel";    uname -a
  echo "## os";        cat /etc/os-release 2>/dev/null | head -4
  echo "## cpu";       lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L2|L3|NUMA|Flags" | sed 's/Flags:.*\(avx[^ ]*\).*/Flags: (see below)/'
  echo "## cpu flags"; grep -o -w -E "avx|avx2|avx512[a-z_]*|avx_vnni|amx[a-z_]*|fma|f16c" /proc/cpuinfo | sort -u | tr '\n' ' '; echo
  echo "## hybrid cores"; for t in core atom; do [ -f /sys/devices/cpu_$t/cpus ] && echo "cpu_$t: $(cat /sys/devices/cpu_$t/cpus)"; done
  echo "## memory";    free -b; grep -E "MemTotal|SwapTotal" /proc/meminfo
  echo "## gpu";       nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv
  for d in /sys/bus/pci/devices/*; do [ "$(cat $d/vendor 2>/dev/null)" = 0x10de ] && [ -f $d/current_link_speed ] && echo "$(basename $d): $(cat $d/current_link_speed) x$(cat $d/current_link_width) (max $(cat $d/max_link_speed) x$(cat $d/max_link_width))"; done
  echo "## cuda";      (nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | tail -2
  echo "## compiler";  gcc --version | head -1; cmake --version | head -1
  echo "## storage";   lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN | grep -v loop; for n in /sys/class/nvme/nvme*; do echo "$(cat $n/model) fw $(cat $n/firmware_rev)"; done
  echo "## filesystem of the model"; findmnt -T "$MDIR" -o TARGET,FSTYPE,OPTIONS
  echo "## cpufreq";   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null
  echo "## models";    ls -l "$MDIR" "$MTP_MODEL"
  if [ "${2:-}" = "--hash-model" ]; then echo "## sha256"; sha256sum "$MDIR"/*.gguf "$MTP_MODEL"; fi
} > "$O" 2>&1
