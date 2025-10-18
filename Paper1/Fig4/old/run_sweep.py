import os, subprocess, time, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 路徑設定 ----
# Windows exe：WSL 建議寫成 /mnt/d/...；如果在 PowerShell 跑，寫 D:\... 也可以
EXE_PATH = r"/mnt/d/DANIEL/RA/Projects/BaseStation/Win32/Debug/Project2.exe"

# 這份 CSV 放在 Fig4 目錄
OUT_CSV_POSIX = os.path.join(HERE, "raw_runs.csv")

# 把 /mnt/d/... 轉成 D:\...（Windows exe 只吃 Windows 路徑）
def to_windows_path(p):
    if p.startswith("/mnt/") and len(p) > 6 and p[6] == '/':
        drive = p[5].upper()
        rest = p[7:].replace('/', '\\')
        return f"{drive}:\\{rest}"
    return p

OUT_CSV = to_windows_path(OUT_CSV_POSIX)
VERSION  = "BaseStation"

# ---- 固定參數 ----
FIXED = dict(
    mu=1.5,
    e=2.4,
    T=10000,
    N=1,
    rtx=1,
    slots=0,
    alwaysCharge=True,
)

# ---- 掃描設定 ----
CS = [5, 10, 100]                 # 你要的 C
LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [12345, 23456, 34567, 45678, 56789]

# ---- 執行控制 ----
SLEEP_BETWEEN = 0.02  # 兩次 run 之間稍微停一下（可設 0）
TIMEOUT_SEC   = 60    # 單個參數組的最大牆鐘秒數（不保證一定要 0.2s 完成，所以給安全值）
RETRY_ON_FAIL = 1     # 失敗重試次數

def ensure_dir(path_windows):
    d = os.path.dirname(path_windows)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def run_one(Cval, lam, seed):
    cmd = [
        EXE_PATH, "--headless",
        f"--mu={FIXED['mu']}",
        f"--lambda={lam}",
        f"--e={FIXED['e']}",
        f"--C={Cval}",
        f"--T={FIXED['T']}",
        f"--seed={seed}",
        f"--N={FIXED['N']}",
        f"--rtx={FIXED['rtx']}",
        f"--slots={FIXED['slots']}",
        "--out=" + OUT_CSV,
        "--version=" + VERSION,
    ]
    if FIXED.get("alwaysCharge", False):
        cmd.append("--alwaysCharge")

    print("RUN:", " ".join(cmd))
    try:
        r = subprocess.run(cmd, shell=False, timeout=TIMEOUT_SEC)
        if r.returncode != 0:
            raise RuntimeError(f"rc={r.returncode}")
    except Exception as e:
        raise

def main():
    ensure_dir(OUT_CSV)
    # 清舊檔（要累加請註解掉）
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    # 開跑
    total = len(CS) * len(LAMBDAS) * len(SEEDS)
    done = 0
    for Cval in CS:
        for lam in LAMBDAS:
            for seed in SEEDS:
                attempts = 0
                while True:
                    try:
                        run_one(Cval, lam, seed)
                        break
                    except Exception as e:
                        attempts += 1
                        if attempts > RETRY_ON_FAIL:
                            print(f"[FAIL] C={Cval}, λ={lam}, seed={seed}: {e}")
                            break
                        print(f"[RETRY] C={Cval}, λ={lam}, seed={seed} ...")
                        time.sleep(0.2)
                done += 1
                print(f"progress: {done}/{total}")
                if SLEEP_BETWEEN > 0:
                    time.sleep(SLEEP_BETWEEN)

    print("All runs done ->", OUT_CSV)
    # 同步一份到 WSL 路徑，方便你在 bash 下直接 ls
    try:
        if OUT_CSV != OUT_CSV_POSIX:
            shutil.copyfile(OUT_CSV, OUT_CSV_POSIX)
            print("Also copied to:", OUT_CSV_POSIX)
    except Exception:
        pass

if __name__ == "__main__":
    main()
