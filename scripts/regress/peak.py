import resource, subprocess, sys, time
t = time.time(); r = subprocess.call(sys.argv[1:])
print(f"PEAK_RSS_GB {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6:.2f} WALL_S {time.time() - t:.0f} EXIT {r}", flush=True)
