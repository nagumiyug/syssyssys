#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import uuid
from pathlib import Path

# 尝试导入 BCC 库 (需要 sudo apt install python3-bcc)
try:
    from bcc import BPF
except ImportError:
    raise SystemExit("错误: 未找到 bcc 模块。请运行 'sudo apt install python3-bcc' 或确保使用正确的 Python 环境。")

# ==========================================
# 1. eBPF 内核态 C 语言代码 (核心逻辑)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// 向用户态传递的事件结构体
struct event_t {
    u64 ts_ns;          // 时间戳 (纳秒)
    u64 duration_us;    // 耗时 (微秒)
    u32 pid;
    u32 tid;
    u32 syscall_id;     // 系统调用号
    int ret;            // 返回值
    char comm[TASK_COMM_LEN]; // 进程名
    char path[256];     // 提取的文件路径
};
BPF_PERF_OUTPUT(events);

// 状态追踪哈希表：用于在 enter 和 exit 之间传递时间戳和字符串指针
struct call_state_t {
    u64 ts;
    const char *path_ptr;
};
BPF_HASH(call_state, u64, struct call_state_t);

// 只监控特定 PID 及其子进程
BPF_HASH(tracked_pids, u32, u32);

// 拦截系统调用入口
TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u32 sys_id = args->id;
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    // 检查是否是被追踪的进程
    u32 *is_tracked = tracked_pids.lookup(&pid);
    if (!is_tracked) return 0;

    
    struct call_state_t state = {};
    state.ts = bpf_ktime_get_ns();
    state.path_ptr = 0;
    u64 arg_val = 0;
    // 修复：不要修改 args 指针本身，而是直接访问 args->args[n]
    // 参数0包含路径的调用号
    if (sys_id == 2 || sys_id == 4 || sys_id == 6 || sys_id == 21 || sys_id == 59 || 
        sys_id == 82 || sys_id == 83 || sys_id == 84 || sys_id == 87 || sys_id == 89 || 
        sys_id == 90 || sys_id == 92) {
        //state.path_ptr = (const char *)args->args[0];
        bpf_probe_read(&arg_val, sizeof(arg_val), &args->args[0]);
        state.path_ptr = (const char *)arg_val;
    } 
    // 参数1包含路径的调用号
    else if (sys_id == 257 || sys_id == 258 || sys_id == 260 || sys_id == 262 || 
             sys_id == 263 || sys_id == 264 || sys_id == 267 || sys_id == 268 || 
             sys_id == 269 || sys_id == 322) {
        //state.path_ptr = (const char *)args->args[1];
        bpf_probe_read(&arg_val, sizeof(arg_val), &args->args[1]);
        state.path_ptr = (const char *)arg_val;
    }

    call_state.update(&id, &state);
    return 0;
}

// 拦截系统调用出口
TRACEPOINT_PROBE(raw_syscalls, sys_exit) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    // 检查是否是被监控进程或其产生的系统调用
    struct call_state_t *sp = call_state.lookup(&id);
    if (sp == 0) return 0; // 错过了 enter 阶段

    u32 sys_id = args->id;
    
    // 进程追踪逻辑：如果是 clone/fork 家族，且成功返回了子 PID，将子 PID 加入追踪名单
    if ((sys_id == 56 /* clone */ || sys_id == 57 /* fork */ || sys_id == 58 /* vfork */) && args->ret > 0) {
        u32 child_pid = args->ret;
        u32 val = 1;
        tracked_pids.update(&child_pid, &val);
    }

    // 构造发往用户态的事件
    struct event_t event = {};
    event.ts_ns = sp->ts;
    event.duration_us = (bpf_ktime_get_ns() - sp->ts) / 1000;
    event.pid = pid;
    event.tid = (u32)id;
    event.syscall_id = sys_id;
    event.ret = args->ret;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    // 如果有记录到路径指针，安全地从用户态空间读取字符串
    if (sp->path_ptr != 0) {
        bpf_probe_read_user_str(&event.path, sizeof(event.path), sp->path_ptr);
    }

    events.perf_submit(args, &event, sizeof(event));
    call_state.delete(&id);
    return 0;
}
"""

# ==========================================
# 2. 用户态映射表与特征推断函数 (完全兼容你的代码)
# ==========================================

# Linux x86_64 常用系统调用字典 (对应你的 op_category)
SYSCALL_MAP = {
    # 文件操作
    2: "open", 257: "openat", 3: "close", 0: "read", 1: "write", 17: "pread64", 18: "pwrite64", 8: "lseek",
    4: "stat", 6: "lstat", 5: "fstat", 262: "newfstatat", 87: "unlink", 263: "unlinkat", 82: "rename",
    264: "renameat", 316: "renameat2", 83: "mkdir", 258: "mkdirat", 84: "rmdir", 21: "access", 269: "faccessat",
    217: "getdents64", 89: "readlink", 267: "readlinkat", 74: "fsync", 75: "fdatasync", 90: "chmod", 91: "fchmod",
    268: "fchmodat", 92: "chown", 93: "fchown", 260: "fchownat", 9: "mmap", 11: "munmap", 10: "mprotect",
    # 进程操作
    59: "execve", 322: "execveat", 56: "clone", 57: "fork", 58: "vfork", 61: "wait4", 247: "waitid",
    60: "exit", 231: "exit_group", 62: "kill", 234: "tgkill", 13: "rt_sigaction", 14: "rt_sigprocmask",
    # 网络操作
    41: "socket", 42: "connect", 43: "accept", 288: "accept4", 49: "bind", 50: "listen", 44: "sendto",
    45: "recvfrom", 46: "sendmsg", 47: "recvmsg", 48: "shutdown", 51: "getsockname", 52: "getpeername",
    54: "setsockopt", 55: "getsockopt"
}

def infer_op_category(syscall_name: str) -> str:
    file_ops = {"open", "openat", "close", "read", "write", "pread64", "pwrite64", "lseek", "stat", "lstat", "fstat", "newfstatat", "unlink", "unlinkat", "rename", "renameat", "renameat2", "mkdir", "mkdirat", "rmdir", "access", "faccessat", "getdents64", "readlink", "readlinkat", "fsync", "fdatasync", "chmod", "fchmod", "fchmodat", "chown", "fchown", "fchownat"}
    process_ops = {"execve", "execveat", "clone", "fork", "vfork", "wait4", "waitid", "exit", "exit_group", "kill", "tgkill", "rt_sigaction", "rt_sigprocmask"}
    network_ops = {"socket", "connect", "accept", "accept4", "bind", "listen", "sendto", "recvfrom", "sendmsg", "recvmsg", "shutdown", "getsockname", "getpeername", "setsockopt", "getsockopt"}
    
    if syscall_name in file_ops: return "file"
    if syscall_name in process_ops: return "process"
    if syscall_name in network_ops: return "network"
    return "other"

def infer_path_type(path: str | None) -> str:
    if not path: return "unknown"
    normalized = path.replace("\\", "/").lower()
    if "source_copy" in normalized: return "source_copy_target"
    if "exfil" in normalized: return "bulk_export_target"
    if "/scan" in normalized: return "scan_target"
    if "/exports/" in normalized or "/gerber/" in normalized or "/drill/" in normalized: return "export_target"
    if "/data/samples/freecad" in normalized: return "freecad_project_source"
    if "/data/samples/kicad" in normalized or "/usr/share/kicad/demos" in normalized: return "kicad_project_source"
    if normalized.startswith("/usr/lib") or normalized.startswith("/lib") or normalized.startswith("/etc"): return "system"
    if normalized.startswith("/tmp"): return "temp"
    return "other"


# ==========================================
# 3. 主程序逻辑与数据写入
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use BCC to perfectly replicate strace dataset collection.")
    parser.add_argument("--software", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--label", required=True, choices=["normal", "abnormal"])
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("Missing workload command")

    output_path = Path(args.output)
    session_id = str(uuid.uuid4())
    rows = []

    # 1. 编译并加载 BPF
    print(f"[BCC] 正在编译 eBPF 探针并附加到内核...")
    b = BPF(text=bpf_text)
    
    # 2. 启动目标工作负载
    print(f"[BCC] 启动目标程序: {' '.join(command)}")
    workload = subprocess.Popen(command)
    
    # 将父进程 PID 写入 BPF 哈希表，开始精准追踪
    tracked_pids = b.get_table("tracked_pids")
    tracked_pids[tracked_pids.Key(workload.pid)] = tracked_pids.Leaf(1)

    # 3. 定义数据接收回调函数
    def process_event(cpu, data, size):
        event = b["events"].event(data)
        syscall_name = SYSCALL_MAP.get(event.syscall_id, f"sys_{event.syscall_id}")
        
        # 为了与之前的 strace -e trace=file,process,network 效果保持一致，这里进行主动过滤
        op_category = infer_op_category(syscall_name)
        if op_category == "other":
            return

        path_str = event.path.decode('utf-8', 'replace').strip() if event.path else None
        if path_str == "": path_str = None
        
        file_ext = Path(path_str).suffix.lower() if path_str else "unknown"

        rows.append({
            "timestamp": event.ts_ns,
            "timestamp_ns": event.ts_ns,
            "session_id": session_id,
            "software": args.software,
            "scenario": args.scenario,
            "label": args.label,
            "pid": event.pid,
            "tid": event.tid,
            "comm": event.comm.decode('utf-8', 'replace').strip(),
            "event_kind": "call", # 统一合并为完整的 call (等价于 strace)
            "syscall_name": syscall_name,
            "return_code": event.ret,
            "duration_us": event.duration_us,
            "op_category": op_category,
            "path": path_str,
            "path_type": infer_path_type(path_str),
            "file_ext": file_ext,
        })

    # 4. 开启缓冲区轮询
    b["events"].open_perf_buffer(process_event, page_cnt=1024)
    
    print(f"[BCC] 监听中 (PID: {workload.pid})... 等待程序执行完毕。")
    try:
        # 当目标子进程仍在运行时，不断拉取内核数据
        while workload.poll() is None:
            b.perf_buffer_poll(timeout=100)
        
        # 程序结束后，额外拉取 0.5 秒，确保缓冲区排空
        end_time = time.time() + 0.5
        while time.time() < end_time:
            b.perf_buffer_poll(timeout=100)
            
    except KeyboardInterrupt:
        print("\n[BCC] 强制中断，正在保存当前数据...")
    
    # 5. 写入 CSV 数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "timestamp_ns", "session_id", "software", "scenario", "label",
        "pid", "tid", "comm", "event_kind", "syscall_name", "return_code", 
        "duration_us", "op_category", "path", "path_type", "file_ext"
    ]
    
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"session_id={session_id}")
    print(f"output={output_path}")
    print(f"events={len(rows)}")

if __name__ == "__main__":
    main()
