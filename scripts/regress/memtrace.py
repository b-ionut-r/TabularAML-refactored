"""Run a script, logging RSS and the main thread's innermost repo frames every 5 s."""
import sys, threading, time, runpy, traceback, os
def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS'): return int(l.split()[1]) / 1e6
main = threading.main_thread().ident
def loop():
    t0 = time.time(); peak = 0
    while True:
        time.sleep(5)
        r = rss(); peak = max(peak, r)
        f = sys._current_frames().get(main)
        st = [f"{os.path.basename(x.filename)}:{x.lineno}:{x.name}" for x in traceback.extract_stack(f) if 'tabularaml' in x.filename or 'scripts' in x.filename][-4:]
        print(f"MEM {time.time()-t0:6.0f}s rss={r:5.2f}GB peak={peak:5.2f} {' > '.join(st)}", file=sys.stderr, flush=True)
threading.Thread(target=loop, daemon=True).start()
sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')
