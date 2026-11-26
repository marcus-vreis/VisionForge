#!/usr/bin/env bash
# A bubblewrap sandbox in front of a VisionForge server job (ADR-254). Linux.
#
#   export VISIONFORGE_JOB_WRAPPER="/opt/visionforge/deploy/server/job-sandbox-bwrap.sh --ro /opt/visionforge --rw /srv/vf-cache"
#   visionforge gui --server ...
#
# The job sees the system's programs and libraries read-only, the folders named
# with --ro (the Python environment VisionForge is installed in, at least) and
# --rw (the weight caches: HF_HOME, TORCH_HOME), and the three places the server
# names for this job: its working folder and the person's runs (read-write) and
# the dataset folders that person may read (read-only). Nothing else of the disk
# exists inside: not the config folder with the tokens, not the other people's
# folders, not the server's own code unless it is under a --ro folder.
#
# Kept on purpose: the network (pretrained weights are downloaded) and the GPU
# device files. A job killed by the server takes its whole sandbox with it
# (--die-with-parent, and a PID namespace of its own).
set -euo pipefail

sandbox=(
  --die-with-parent
  --unshare-pid --unshare-ipc --unshare-uts
  --proc /proc
  --dev /dev
  --tmpfs /tmp
  --ro-bind /usr /usr
)
for dir in /bin /sbin /lib /lib32 /lib64 /opt/conda; do
  # On a merged-/usr system these are links into /usr; keep them as links.
  if [ -L "$dir" ]; then
    sandbox+=(--symlink "$(readlink "$dir")" "$dir")
  elif [ -e "$dir" ]; then
    sandbox+=(--ro-bind "$dir" "$dir")
  fi
done
for file in /etc/resolv.conf /etc/hosts /etc/nsswitch.conf /etc/passwd /etc/group \
  /etc/ld.so.cache /etc/ld.so.conf /etc/ld.so.conf.d /etc/ssl /etc/ca-certificates \
  /etc/pki /etc/alternatives /etc/localtime /etc/OpenCL /etc/vulkan; do
  sandbox+=(--ro-bind-try "$file" "$file")
done
# The GPU: NVIDIA's device files, and the shared memory DataLoader workers use.
for device in /dev/nvidia* /dev/dxg; do
  [ -e "$device" ] && sandbox+=(--dev-bind "$device" "$device")
done
sandbox+=(--tmpfs /dev/shm)

while [ $# -gt 0 ]; do
  case "$1" in
    --ro) sandbox+=(--ro-bind "$2" "$2"); shift 2 ;;
    --rw) sandbox+=(--bind "$2" "$2"); shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "job-sandbox-bwrap.sh: no command to run" >&2
  exit 64
fi

IFS=: read -r -a writable <<<"${VISIONFORGE_JOB_WRITE:-}"
for dir in "${writable[@]}"; do
  [ -n "$dir" ] || continue
  # A person's first run: the folder is created here, outside the sandbox.
  mkdir -p "$dir"
  sandbox+=(--bind "$dir" "$dir")
done
IFS=: read -r -a readable <<<"${VISIONFORGE_JOB_READ:-}"
for dir in "${readable[@]}"; do
  [ -n "$dir" ] && sandbox+=(--ro-bind-try "$dir" "$dir")
done
# Last, so the working folder is writable even inside a read-only folder.
workdir="${VISIONFORGE_JOB_WORKDIR:?the server sets it for each job}"
sandbox+=(--bind "$workdir" "$workdir" --chdir "$workdir")

exec bwrap "${sandbox[@]}" -- "$@"
