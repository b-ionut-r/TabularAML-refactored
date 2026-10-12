"""Stream the single csv of a Kaggle dataset file download (a deflate zip) without storing it."""
import io, ssl, struct, sys, zlib, urllib.request
ctx = ssl.create_default_context(cafile='/root/.ccr/ca-bundle.crt')
class NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
def url_of(ref, fname):
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), NoRedir)
    try:
        op.open(f'https://www.kaggle.com/api/v1/datasets/download/{ref}/{fname}'); raise SystemExit('no redirect')
    except urllib.error.HTTPError as e:
        return e.headers['Location']
class Inflate(io.RawIOBase):
    def __init__(s, ref, fname):
        s.r = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)).open(url_of(ref, fname))
        h = s.r.read(30); assert h[:4] == b'PK\x03\x04', h[:4]
        n, m = struct.unpack('<HH', h[26:30]); s.r.read(n + m)
        assert struct.unpack('<H', h[8:10])[0] == 8
        s.d, s.buf, s.done = zlib.decompressobj(-15), b'', False
    def readable(s): return True
    def readinto(s, b):
        while len(s.buf) < len(b) and not s.done:
            c = s.r.read(1 << 20)
            if not c:
                s.buf += s.d.flush(); s.done = True; break
            s.buf += s.d.decompress(c)
            if s.d.eof: s.done = True
        n = min(len(b), len(s.buf)); b[:n] = s.buf[:n]; s.buf = s.buf[n:]; return n
def open_csv(ref, fname):
    return io.BufferedReader(Inflate(ref, fname), 1 << 20)
if __name__ == '__main__':
    f = open_csv(sys.argv[1], sys.argv[2])
    for _ in range(3): print(f.readline().decode().strip())
