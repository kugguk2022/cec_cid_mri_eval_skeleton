
# Budget schedules (choose based on CEC signature; Lambert-W-free log-boost variant)
import math

def log_boost_schedule(t, lam0=1.0, gamma=0.5):
    return lam0 + gamma * math.log(t+1.0)

def plateau_then_boost(t, warmup=1000, lam_warm=0.5, lam_boost=0.02):
    if t < warmup:
        return lam_warm
    return lam_warm + lam_boost * math.log(t - warmup + 2.0)

def burst_schedule(t, period=500, peak=2.0, base=0.5):
    import math
    return base + max(0.0, peak*math.sin(2*math.pi*(t%period)/period))
