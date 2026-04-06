#define _GNU_SOURCE

/*
 * Build:
 *   cc -O2 -Wall -Wextra -std=c11 tools/profiling/ncu_capwrap.c -o ncu_capwrap
 *
 * Install with file capabilities:
 *   sudo install -o root -g root -m 0755 ncu_capwrap /usr/local/bin/ncu_capwrap
 *   sudo setcap cap_sys_admin=ep /usr/local/bin/ncu_capwrap
 *
 * Install with setuid root:
 *   sudo install -o root -g root -m 4755 ncu_capwrap /usr/local/bin/ncu_capwrap
 *
 * Usage:
 *   /usr/local/bin/ncu_capwrap [ncu options ...] -- /abs/path/to/allowed/binary [binary args ...]
 */

#include <errno.h>
#include <grp.h>
#include <linux/capability.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

static const char* const kNcuPath = "/usr/local/cuda-13.0/bin/ncu";
static const char* const kAllowedTargetPrefixes[] = {
    "/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/",
};

static const char* const kSafeEnvVars[] = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "NEMOTRON_FORWARD_MANIFEST",
    "TERM",
    "TMPDIR",
    "USER",
    "CUDA_VISIBLE_DEVICES",
};

static const int kSysAdminIndex = CAP_SYS_ADMIN / 32;
static const unsigned int kSysAdminMask = 1u << (CAP_SYS_ADMIN % 32);

static void Die(const char* message) {
  fprintf(stderr, "ncu_capwrap: %s\n", message);
  exit(1);
}

static void DieErrno(const char* message) {
  fprintf(stderr, "ncu_capwrap: %s: %s\n", message, strerror(errno));
  exit(1);
}

static bool StartsWith(const char* value, const char* prefix) {
  return strncmp(value, prefix, strlen(prefix)) == 0;
}

static bool IsAllowedTarget(const char* path) {
  size_t i = 0;
  for (i = 0; i < sizeof(kAllowedTargetPrefixes) / sizeof(kAllowedTargetPrefixes[0]); ++i) {
    if (StartsWith(path, kAllowedTargetPrefixes[i])) {
      return true;
    }
  }
  return false;
}

static void RequireAllowedExecutable(const char* target_path) {
  if (target_path == NULL || target_path[0] != '/') {
    Die("target path must be absolute and follow `--`");
  }

  char resolved[PATH_MAX];
  if (realpath(target_path, resolved) == NULL) {
    DieErrno("failed to resolve target path");
  }
  if (!IsAllowedTarget(resolved)) {
    Die("target path is outside the allowed build directory");
  }

  struct stat st;
  if (stat(resolved, &st) != 0) {
    DieErrno("failed to stat target path");
  }
  if (!S_ISREG(st.st_mode)) {
    Die("target path is not a regular file");
  }
  if (access(resolved, X_OK) != 0) {
    DieErrno("target path is not executable");
  }
}

static long CapSet(struct __user_cap_header_struct* header,
                   const struct __user_cap_data_struct* data) {
  return syscall(SYS_capset, header, data);
}

static void SetOnlySysAdminCapability() {
  struct __user_cap_header_struct header = {};
  struct __user_cap_data_struct data[2] = {};
  header.version = _LINUX_CAPABILITY_VERSION_3;
  header.pid = 0;

  data[kSysAdminIndex].effective = kSysAdminMask;
  data[kSysAdminIndex].permitted = kSysAdminMask;
  data[kSysAdminIndex].inheritable = kSysAdminMask;

  if (CapSet(&header, data) != 0) {
    DieErrno("failed to set CAP_SYS_ADMIN");
  }
}

static void RaiseAmbientSysAdmin() {
  if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0) {
    DieErrno("failed to clear ambient capabilities");
  }
  if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, CAP_SYS_ADMIN, 0, 0) != 0) {
    DieErrno("failed to raise ambient CAP_SYS_ADMIN");
  }
}

static void SanitizeEnvironment() {
  struct SavedVar {
    const char* name;
    char* value;
  } saved[sizeof(kSafeEnvVars) / sizeof(kSafeEnvVars[0])];

  for (size_t i = 0; i < sizeof(saved) / sizeof(saved[0]); ++i) {
    const char* value = getenv(kSafeEnvVars[i]);
    saved[i].name = kSafeEnvVars[i];
    saved[i].value = value == NULL ? NULL : strdup(value);
    if (value != NULL && saved[i].value == NULL) {
      DieErrno("failed to copy environment value");
    }
  }

  if (clearenv() != 0) {
    DieErrno("failed to clear environment");
  }
  if (setenv("PATH", "/usr/bin:/bin", 1) != 0) {
    DieErrno("failed to set PATH");
  }

  for (size_t i = 0; i < sizeof(saved) / sizeof(saved[0]); ++i) {
    if (saved[i].value != NULL) {
      if (setenv(saved[i].name, saved[i].value, 1) != 0) {
        DieErrno("failed to restore environment value");
      }
      free(saved[i].value);
    }
  }
}

static void DropToRealUserKeepSysAdmin() {
  const uid_t real_uid = getuid();
  const gid_t real_gid = getgid();

  if (geteuid() == 0) {
    if (prctl(PR_SET_KEEPCAPS, 1L, 0L, 0L, 0L) != 0) {
      DieErrno("failed to enable keepcaps");
    }
    if (setgroups(0, NULL) != 0) {
      DieErrno("failed to clear supplementary groups");
    }
    if (setresgid(real_gid, real_gid, real_gid) != 0) {
      DieErrno("failed to drop group privileges");
    }
    if (setresuid(real_uid, real_uid, real_uid) != 0) {
      DieErrno("failed to drop user privileges");
    }
  }

  SetOnlySysAdminCapability();
  RaiseAmbientSysAdmin();
}

static void PrintUsage() {
  fprintf(
      stderr,
      "Usage: ncu_capwrap [ncu options ...] -- /abs/path/to/allowed/binary [binary args ...]\n"
      "Allowed target prefixes:\n"
      "  %s\n",
      kAllowedTargetPrefixes[0]);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage();
    return 1;
  }

  int separator = -1;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--") == 0) {
      separator = i;
      break;
    }
  }
  if (separator < 0 || separator + 1 >= argc) {
    PrintUsage();
    return 1;
  }

  RequireAllowedExecutable(argv[separator + 1]);
  SanitizeEnvironment();
  DropToRealUserKeepSysAdmin();

  char** exec_argv = (char**) calloc((size_t) argc + 1, sizeof(char*));
  if (exec_argv == NULL) {
    DieErrno("failed to allocate argv");
  }

  int out = 0;
  exec_argv[out++] = (char*) kNcuPath;
  for (int i = 1; i < separator; ++i) {
    exec_argv[out++] = argv[i];
  }
  for (int i = separator + 1; i < argc; ++i) {
    exec_argv[out++] = argv[i];
  }
  exec_argv[out] = NULL;

  execv(kNcuPath, exec_argv);
  DieErrno("failed to exec ncu");
}
