#!/usr/bin/env bash
set -euo pipefail

DEFAULT_NODES="ds35,ds36"
DEFAULT_SLOTS="8"
DEFAULT_HOSTFILE="/home/workspace/hostfile_2node_gpu2"
DEFAULT_KEY_PATH="${HOME}/.ssh/id_rsa"
DEFAULT_PORTS="22"
DEFAULT_SSH_CONFIG="${HOME}/.ssh/config"
SSH_CONFIG_BEGIN_MARKER="# >>> llamafactory multinode ssh config >>>"
SSH_CONFIG_END_MARKER="# <<< llamafactory multinode ssh config <<<"
SSHD_CONFIG_BEGIN_MARKER="# >>> llamafactory multinode sshd config >>>"
SSHD_CONFIG_END_MARKER="# <<< llamafactory multinode sshd config <<<"

SSH_USER="${USER:-}"
SSH_PASSWORD=""
NODES_CSV="${DEFAULT_NODES}"
HOSTS_CSV=""
SLOTS_CSV="${DEFAULT_SLOTS}"
HOSTFILE="${DEFAULT_HOSTFILE}"
KEY_PATH="${DEFAULT_KEY_PATH}"
PORTS_CSV="${DEFAULT_PORTS}"
SSH_CONFIG="${DEFAULT_SSH_CONFIG}"
CLEAN_ONLY=0
CLEAN_HOSTFILE=0
CLEAN_KNOWN_HOSTS=0
CLEAN_REMOTE_KEY=0
SETUP_LOCAL_SSHD=1
SKIP_LOCAL_SSHD=0
SSHD_PORT=""
SSHD_CONFIG="/etc/ssh/sshd_config"
SELF_NODE=""
PREPARE_ONLY=0
REQUIRE_PDSH=1

usage() {
  cat <<'EOF'
Configure passwordless SSH for LLaMA-Factory DeepSpeed multinode training.

Usage:
  setup_multinode_ssh.sh [options]

Options:
  --user <name>             SSH user for all nodes. Defaults to current $USER.
  --password <password>     Password used by sshpass for first-time key copy.
                            If omitted and sshpass exists, the script prompts
                            interactively. Without sshpass, SSH prompts normally.
  --nodes <csv>             Node aliases, comma-separated. Default: ds35,ds36.
  --hosts <csv>             Real IPs or hostnames for the aliases in --nodes.
                            If set, the script writes a managed block to
                            ~/.ssh/config before connecting.
  --ports <n|csv>           SSH port per node. Default: 22.
  --slots <n|csv>           GPU slots per node. Default: 8.
                            Use one value for all nodes or a CSV matching nodes.
  --hostfile <path>         Hostfile to write. Default:
                            /home/workspace/hostfile_2node_gpu2
  --key-path <path>         SSH private key path. Default: ~/.ssh/id_rsa.
  --ssh-config <path>       SSH config path to update when --hosts is set.
                            Default: ~/.ssh/config.
  --setup-local-sshd        Configure and restart local sshd before key copy.
                            Enabled by default. This does not install
                            openssh-server.
  --skip-local-sshd         Do not configure local sshd.
  --sshd-port <port>        Local sshd port for --setup-local-sshd.
                            Default: first value in --ports.
  --self-node <alias>       Current container alias. If --sshd-port is omitted,
                            use this node's matching value from --ports.
  --prepare-only            Only configure local sshd, SSH aliases, key, and
                            hostfile; do not connect to remote nodes.
  --skip-pdsh-check         Do not require or verify pdsh. Use only for SSH
                            preparation without launching DeepSpeed.
  --sshd-config <path>      Local sshd config path. Default:
                            /etc/ssh/sshd_config.
  --clean                   Remove this script's managed ~/.ssh/config block
                            and exit. Combine with cleanup options below.
  --clean-hostfile          With --clean, also remove --hostfile.
  --clean-known-hosts       With --clean, also remove known_hosts entries for
                            aliases in --nodes.
  --clean-remote-key        With --clean, also remove --key-path public key from
                            authorized_keys on aliases in --nodes.
  -h, --help                Show this help.

Example:
  bash scripts/setup_multinode_ssh.sh --user root \
    --nodes ds35,ds36 --hosts 192.168.1.35,192.168.1.36 --slots 8
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "[setup-multinode-ssh] $*"
}

expand_path() {
  local value="$1"
  case "${value}" in
    "~") printf '%s\n' "${HOME}" ;;
    "~/"*) printf '%s/%s\n' "${HOME}" "${value#~/}" ;;
    *) printf '%s\n' "${value}" ;;
  esac
}

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  local old_ifs="${IFS}"
  IFS=","
  read -r -a out_ref <<<"${csv}"
  IFS="${old_ifs}"

  local item
  local cleaned=()
  for item in "${out_ref[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "${item}" ]] && cleaned+=("${item}")
  done
  out_ref=("${cleaned[@]}")
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    die "missing required command: ${cmd}. Please ask the administrator to install it first."
  fi
}

has_command() {
  command -v "$1" >/dev/null 2>&1
}

check_pdsh_install() {
  local output
  local status

  set +e
  output="$(pdsh -q 2>&1)"
  status=$?
  set -e

  if [[ ${status} -ne 0 ]] && grep -qiE "module path .*insecure|couldn't load any pdsh modules|could not load any pdsh modules" <<<"${output}"; then
    printf '%s\n' "${output}" >&2
    die "pdsh is installed but its module path is insecure. Check ownership with: stat -c '%U:%G %a %n' /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh ; fix as root with: chown root:root /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh && chmod go-w /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh"
  fi
}

restart_sshd() {
  local sshd_bin="$1"
  local config_path="$2"

  if [[ -x /etc/init.d/ssh ]] && /etc/init.d/ssh restart; then
    return
  fi
  if has_command service && service ssh restart; then
    return
  fi
  if has_command systemctl && (systemctl restart ssh || systemctl restart sshd); then
    return
  fi

  info "Service restart failed or is unavailable; starting sshd directly."
  "${sshd_bin}" -f "${config_path}"
}

setup_local_sshd() {
  local port="$1"
  local config_path="$2"
  local tmp_path="${config_path}.tmp.$$"
  local backup_path="${config_path}.bak.$(date +%Y%m%d%H%M%S)"
  local sshd_bin=""
  local skip=0

  [[ "${port}" =~ ^[1-9][0-9]*$ ]] || die "invalid --sshd-port: ${port}"
  [[ -f "${config_path}" ]] || die "sshd config not found: ${config_path}. Install openssh-server first: apt update && apt install openssh-server -y"
  if has_command sshd; then
    sshd_bin="$(command -v sshd)"
  elif [[ -x /usr/sbin/sshd ]]; then
    sshd_bin="/usr/sbin/sshd"
  else
    die "sshd command not found. Install openssh-server first: apt update && apt install openssh-server -y"
  fi

  info "Configuring local sshd on port ${port}"
  cp "${config_path}" "${backup_path}"
  info "Backed up ${config_path} to ${backup_path}"

  {
    while IFS= read -r line || [[ -n "${line}" ]]; do
      if [[ "${line}" == "${SSHD_CONFIG_BEGIN_MARKER}" ]]; then
        skip=1
        continue
      fi
      if [[ "${line}" == "${SSHD_CONFIG_END_MARKER}" ]]; then
        skip=0
        continue
      fi
      [[ ${skip} -eq 1 ]] && continue
      if [[ "${line}" =~ ^[[:space:]]*(Port|PermitRootLogin|PubkeyAuthentication|PasswordAuthentication)[[:space:]]+ ]]; then
        printf '# %s\n' "${line}"
      else
        printf '%s\n' "${line}"
      fi
    done <"${config_path}"
    printf '\n%s\n' "${SSHD_CONFIG_BEGIN_MARKER}"
    printf 'Port %s\n' "${port}"
    printf 'PermitRootLogin yes\n'
    printf 'PubkeyAuthentication yes\n'
    printf 'PasswordAuthentication yes\n'
    printf '%s\n' "${SSHD_CONFIG_END_MARKER}"
  } >"${tmp_path}"

  mv "${tmp_path}" "${config_path}"
  ssh-keygen -A
  mkdir -p /run/sshd
  "${sshd_bin}" -t -f "${config_path}"
  restart_sshd "${sshd_bin}" "${config_path}"
  info "Local sshd restarted"
}

make_target() {
  local node="$1"
  if [[ "${node}" == *@* ]]; then
    printf '%s\n' "${node}"
  else
    [[ -n "${SSH_USER}" ]] || die "--user is required when node aliases do not include user@host"
    printf '%s@%s\n' "${SSH_USER}" "${node}"
  fi
}

write_ssh_config() {
  local -n nodes_ref="$1"
  local -n hosts_ref="$2"
  local -n ports_ref="$3"
  local config_path="$4"
  local tmp_path="${config_path}.tmp.$$"
  local skip=0

  mkdir -p "$(dirname "${config_path}")"
  chmod 700 "$(dirname "${config_path}")"

  {
    printf '%s\n' "${SSH_CONFIG_BEGIN_MARKER}"
    for index in "${!nodes_ref[@]}"; do
      printf 'Host %s\n' "${nodes_ref[${index}]}"
      printf '  HostName %s\n' "${hosts_ref[${index}]}"
      [[ -n "${SSH_USER}" ]] && printf '  User %s\n' "${SSH_USER}"
      printf '  Port %s\n' "${ports_ref[${index}]}"
      printf '  StrictHostKeyChecking accept-new\n'
      printf '\n'
    done
    printf '%s\n' "${SSH_CONFIG_END_MARKER}"

    if [[ -f "${config_path}" ]]; then
      while IFS= read -r line || [[ -n "${line}" ]]; do
        if [[ "${line}" == "${SSH_CONFIG_BEGIN_MARKER}" ]]; then
          skip=1
          continue
        fi
        if [[ "${line}" == "${SSH_CONFIG_END_MARKER}" ]]; then
          skip=0
          continue
        fi
        [[ ${skip} -eq 0 ]] && printf '%s\n' "${line}"
      done <"${config_path}"
    fi
  } >"${tmp_path}"

  mv "${tmp_path}" "${config_path}"
  chmod 600 "${config_path}"
  info "Wrote SSH alias config to ${config_path}"
}

remove_managed_ssh_config() {
  local config_path="$1"
  local tmp_path="${config_path}.tmp.$$"
  local skip=0

  if [[ ! -f "${config_path}" ]]; then
    info "SSH config does not exist: ${config_path}"
    return
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" == "${SSH_CONFIG_BEGIN_MARKER}" ]]; then
      skip=1
      continue
    fi
    if [[ "${line}" == "${SSH_CONFIG_END_MARKER}" ]]; then
      skip=0
      continue
    fi
    [[ ${skip} -eq 0 ]] && printf '%s\n' "${line}"
  done <"${config_path}" >"${tmp_path}"

  mv "${tmp_path}" "${config_path}"
  chmod 600 "${config_path}"
  info "Removed managed SSH alias config from ${config_path}"
}

clean_known_hosts() {
  local known_hosts="${HOME}/.ssh/known_hosts"
  local node

  if [[ ! -f "${known_hosts}" ]]; then
    info "known_hosts does not exist: ${known_hosts}"
    return
  fi

  for node in "$@"; do
    ssh-keygen -R "${node}" -f "${known_hosts}" >/dev/null 2>&1 || true
    info "Removed known_hosts entry for ${node} if present"
  done
}

clean_remote_key() {
  local public_key_path="$1"
  shift
  local public_key
  local escaped_public_key
  local node
  local target

  [[ -f "${public_key_path}" ]] || die "public key not found: ${public_key_path}"
  public_key="$(cat "${public_key_path}")"
  escaped_public_key="${public_key//\'/\'\\\'\'}"

  for node in "$@"; do
    target="$(make_target "${node}")"
    info "Removing public key from ${target}"
    ssh -o StrictHostKeyChecking=accept-new "${target}" \
      "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && grep -vxF '${escaped_public_key}' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp || true; mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
  done
}

write_hostfile() {
  local -n nodes_ref="$1"
  local -n slots_ref="$2"
  local hostfile_path="$3"

  info "Writing DeepSpeed hostfile: ${hostfile_path}"
  mkdir -p "$(dirname "${hostfile_path}")"
  : >"${hostfile_path}"
  for index in "${!nodes_ref[@]}"; do
    printf '%s slots=%s\n' "${nodes_ref[${index}]}" "${slots_ref[${index}]}" >>"${hostfile_path}"
  done
}

append_known_host() {
  local node="$1"
  local scan_host="$2"
  local port="$3"
  local known_hosts="${HOME}/.ssh/known_hosts"

  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"
  touch "${known_hosts}"
  chmod 600 "${known_hosts}"

  if ssh-keygen -F "${node}" -f "${known_hosts}" >/dev/null 2>&1; then
    info "known_hosts already contains ${node}"
    return
  fi

  info "Scanning host key for ${scan_host}:${port}"
  if ! ssh-keyscan -p "${port}" -H "${scan_host}" >>"${known_hosts}" 2>/dev/null; then
    info "ssh-keyscan failed for ${scan_host}:${port}; continuing and letting ssh handle host key acceptance."
  fi
}

run_password_ssh() {
  local target="$1"
  shift
  SSHPASS="${SSH_PASSWORD}" sshpass -e ssh \
    -o StrictHostKeyChecking=accept-new \
    -o PreferredAuthentications=password,keyboard-interactive,publickey \
    "${target}" "$@"
}

run_password_ssh_copy_id() {
  local target="$1"
  shift
  SSHPASS="${SSH_PASSWORD}" sshpass -e ssh-copy-id "$@" "${target}"
}

run_interactive_ssh() {
  local target="$1"
  shift
  ssh -o StrictHostKeyChecking=accept-new "${target}" "$@"
}

run_interactive_ssh_copy_id() {
  local target="$1"
  shift
  ssh-copy-id "$@" "${target}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || die "--user requires a value"
      SSH_USER="$2"
      shift 2
      ;;
    --password)
      [[ $# -ge 2 ]] || die "--password requires a value"
      SSH_PASSWORD="$2"
      shift 2
      ;;
    --nodes)
      [[ $# -ge 2 ]] || die "--nodes requires a value"
      NODES_CSV="$2"
      shift 2
      ;;
    --hosts)
      [[ $# -ge 2 ]] || die "--hosts requires a value"
      HOSTS_CSV="$2"
      shift 2
      ;;
    --ports)
      [[ $# -ge 2 ]] || die "--ports requires a value"
      PORTS_CSV="$2"
      shift 2
      ;;
    --slots)
      [[ $# -ge 2 ]] || die "--slots requires a value"
      SLOTS_CSV="$2"
      shift 2
      ;;
    --hostfile)
      [[ $# -ge 2 ]] || die "--hostfile requires a value"
      HOSTFILE="$2"
      shift 2
      ;;
    --key-path)
      [[ $# -ge 2 ]] || die "--key-path requires a value"
      KEY_PATH="$2"
      shift 2
      ;;
    --ssh-config)
      [[ $# -ge 2 ]] || die "--ssh-config requires a value"
      SSH_CONFIG="$2"
      shift 2
      ;;
    --setup-local-sshd)
      SETUP_LOCAL_SSHD=1
      shift
      ;;
    --skip-local-sshd)
      SKIP_LOCAL_SSHD=1
      SETUP_LOCAL_SSHD=0
      shift
      ;;
    --sshd-port)
      [[ $# -ge 2 ]] || die "--sshd-port requires a value"
      SSHD_PORT="$2"
      shift 2
      ;;
    --self-node)
      [[ $# -ge 2 ]] || die "--self-node requires a value"
      SELF_NODE="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --skip-pdsh-check)
      REQUIRE_PDSH=0
      shift
      ;;
    --sshd-config)
      [[ $# -ge 2 ]] || die "--sshd-config requires a value"
      SSHD_CONFIG="$2"
      shift 2
      ;;
    --clean)
      CLEAN_ONLY=1
      shift
      ;;
    --clean-hostfile)
      CLEAN_HOSTFILE=1
      shift
      ;;
    --clean-known-hosts)
      CLEAN_KNOWN_HOSTS=1
      shift
      ;;
    --clean-remote-key)
      CLEAN_REMOTE_KEY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

KEY_PATH="$(expand_path "${KEY_PATH}")"
HOSTFILE="$(expand_path "${HOSTFILE}")"
SSH_CONFIG="$(expand_path "${SSH_CONFIG}")"
SSHD_CONFIG="$(expand_path "${SSHD_CONFIG}")"

declare -a NODES
declare -a HOSTS
declare -a SLOTS
declare -a PORTS
split_csv "${NODES_CSV}" NODES
split_csv "${HOSTS_CSV}" HOSTS
split_csv "${SLOTS_CSV}" SLOTS
split_csv "${PORTS_CSV}" PORTS

[[ ${#NODES[@]} -gt 0 ]] || die "--nodes must contain at least one node"
if [[ ${#HOSTS[@]} -gt 0 && ${#HOSTS[@]} -ne ${#NODES[@]} ]]; then
  die "--hosts must be empty or a CSV with the same length as --nodes"
fi
if [[ ${#SLOTS[@]} -eq 1 && ${#NODES[@]} -gt 1 ]]; then
  single_slot="${SLOTS[0]}"
  SLOTS=()
  for _ in "${NODES[@]}"; do
    SLOTS+=("${single_slot}")
  done
fi
[[ ${#SLOTS[@]} -eq ${#NODES[@]} ]] || die "--slots must be one value or a CSV with the same length as --nodes"
if [[ ${#PORTS[@]} -eq 1 && ${#NODES[@]} -gt 1 ]]; then
  single_port="${PORTS[0]}"
  PORTS=()
  for _ in "${NODES[@]}"; do
    PORTS+=("${single_port}")
  done
fi
[[ ${#PORTS[@]} -eq ${#NODES[@]} ]] || die "--ports must be one value or a CSV with the same length as --nodes"

for slot in "${SLOTS[@]}"; do
  [[ "${slot}" =~ ^[1-9][0-9]*$ ]] || die "invalid slot count: ${slot}"
done
for port in "${PORTS[@]}"; do
  [[ "${port}" =~ ^[1-9][0-9]*$ ]] || die "invalid SSH port: ${port}"
done
if [[ -n "${SSHD_PORT}" ]]; then
  [[ "${SSHD_PORT}" =~ ^[1-9][0-9]*$ ]] || die "invalid --sshd-port: ${SSHD_PORT}"
fi
if [[ -n "${SELF_NODE}" ]]; then
  self_found=0
  for index in "${!NODES[@]}"; do
    if [[ "${NODES[${index}]}" == "${SELF_NODE}" ]]; then
      self_found=1
      if [[ -z "${SSHD_PORT}" ]]; then
        SSHD_PORT="${PORTS[${index}]}"
      fi
      break
    fi
  done
  [[ ${self_found} -eq 1 ]] || die "--self-node must match one alias in --nodes"
fi

if [[ ${CLEAN_ONLY} -eq 1 ]]; then
  require_command ssh-keygen
  remove_managed_ssh_config "${SSH_CONFIG}"
  if [[ ${CLEAN_HOSTFILE} -eq 1 ]]; then
    if [[ -f "${HOSTFILE}" ]]; then
      rm -f "${HOSTFILE}"
      info "Removed hostfile: ${HOSTFILE}"
    else
      info "Hostfile does not exist: ${HOSTFILE}"
    fi
  fi
  if [[ ${CLEAN_KNOWN_HOSTS} -eq 1 ]]; then
    clean_known_hosts "${NODES[@]}"
  fi
  if [[ ${CLEAN_REMOTE_KEY} -eq 1 ]]; then
    require_command ssh
    clean_remote_key "${KEY_PATH}.pub" "${NODES[@]}"
  fi
  info "Clean complete."
  exit 0
fi

require_command ssh
require_command ssh-keygen
require_command ssh-keyscan
require_command ssh-copy-id

if [[ ${SETUP_LOCAL_SSHD} -eq 1 ]]; then
  if [[ -z "${SSHD_PORT}" ]]; then
    SSHD_PORT="${PORTS[0]}"
    info "--sshd-port/--self-node not provided; using first --ports value: ${SSHD_PORT}"
  fi
  setup_local_sshd "${SSHD_PORT}" "${SSHD_CONFIG}"
else
  if [[ ${SKIP_LOCAL_SSHD} -eq 1 ]]; then
    info "Skipping local sshd setup by --skip-local-sshd."
  fi
fi

if [[ ${#HOSTS[@]} -gt 0 ]]; then
  write_ssh_config NODES HOSTS PORTS "${SSH_CONFIG}"
else
  info "No --hosts provided; assuming node aliases are already resolvable by SSH."
fi

HAS_SSHPASS=0
if has_command sshpass; then
  HAS_SSHPASS=1
elif [[ -n "${SSH_PASSWORD}" ]]; then
  die "--password requires sshpass, but sshpass is not installed. Re-run without --password for manual SSH prompts."
else
  info "sshpass not found; falling back to interactive SSH prompts."
fi

if [[ ${HAS_SSHPASS} -eq 1 && -z "${SSH_PASSWORD}" ]]; then
  read -r -s -p "SSH password for ${SSH_USER:-configured user}: " SSH_PASSWORD
  echo
fi
if [[ ${HAS_SSHPASS} -eq 1 && -z "${SSH_PASSWORD}" ]]; then
  die "empty password is not allowed when using sshpass"
fi

HAS_PDSH=0
if has_command pdsh; then
  HAS_PDSH=1
  check_pdsh_install
elif [[ ${REQUIRE_PDSH} -eq 1 ]]; then
  die "missing required command: pdsh. Install it first: apt install -y pdsh"
else
  info "pdsh not found; skipping pdsh verification by --skip-pdsh-check. DeepSpeed multinode launch may still require pdsh."
fi

if [[ ! -f "${KEY_PATH}" ]]; then
  info "Creating SSH key: ${KEY_PATH}"
  mkdir -p "$(dirname "${KEY_PATH}")"
  chmod 700 "$(dirname "${KEY_PATH}")"
  ssh-keygen -t rsa -b 4096 -N "" -f "${KEY_PATH}" -C "llamafactory-multinode-$(hostname)"
else
  info "Using existing SSH key: ${KEY_PATH}"
fi
[[ -f "${KEY_PATH}.pub" ]] || die "public key not found: ${KEY_PATH}.pub"

write_hostfile NODES SLOTS "${HOSTFILE}"

if [[ ${PREPARE_ONLY} -eq 1 ]]; then
  info "Prepare-only mode complete. Run the full command after every training container has been prepared."
  info "Local public key:"
  cat "${KEY_PATH}.pub"
  exit 0
fi

for index in "${!NODES[@]}"; do
  node="${NODES[${index}]}"
  scan_host="${node}"
  if [[ ${#HOSTS[@]} -gt 0 ]]; then
    scan_host="${HOSTS[${index}]}"
  fi
  target="$(make_target "${node}")"
  append_known_host "${node}" "${scan_host}" "${PORTS[${index}]}"

  info "Preparing ~/.ssh on ${target}"
  if [[ ${HAS_SSHPASS} -eq 1 ]]; then
    run_password_ssh "${target}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
  else
    run_interactive_ssh "${target}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
  fi

  info "Copying public key to ${target}"
  if [[ ${HAS_SSHPASS} -eq 1 ]]; then
    run_password_ssh_copy_id "${target}" \
      -i "${KEY_PATH}.pub" \
      -o StrictHostKeyChecking=accept-new
  else
    run_interactive_ssh_copy_id "${target}" \
      -i "${KEY_PATH}.pub" \
      -o StrictHostKeyChecking=accept-new
  fi
done

node_list="$(IFS=,; echo "${NODES[*]}")"
info "Verifying passwordless ssh"
for node in "${NODES[@]}"; do
  target="$(make_target "${node}")"
  ssh -i "${KEY_PATH}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 "${target}" hostname
done

if [[ ${HAS_PDSH} -eq 1 ]]; then
  info "Verifying pdsh"
  PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-i ${KEY_PATH} -o IdentitiesOnly=yes -o BatchMode=yes" \
    pdsh -S -f 1024 -w "${node_list}" hostname
else
  info "Skipped pdsh verification because pdsh is not installed."
fi

info "Verifying GPU visibility"
for node in "${NODES[@]}"; do
  target="$(make_target "${node}")"
  ssh -i "${KEY_PATH}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 "${target}" nvidia-smi -L
done

info "Done. Hostfile content:"
cat "${HOSTFILE}"
