"""Download the three approved public files from the official Occ3D folder.

Uses only Python's standard library. Partial files are retained on failure;
resuming requires the server to honor byte ranges. No archives are extracted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request

FILES={'annotations.json':'1_YVM6uqhw-Mxfs0jkXhHtOeGKwkvfuFo',
       'gts.tar.gz':'17HubGsfioQr1d_39VwVPXelobAFo4Xqh',
       'imgs.tar.gz':'1EZdkeZTxib1rdwmDsZW8uDfSxLzwYqGi'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--files',nargs='+',choices=tuple(FILES),default=list(FILES))
    args=parser.parse_args()
    dest=args.destination.resolve(); dest.mkdir(parents=True,exist_ok=True)
    status_path=dest/'download_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'files':{}}
    lock=threading.Lock()
    def update(name,**values):
        with lock:
            status['files'].setdefault(name,{}).update(values)
            status['updated_unix']=time.time()
            tmp=status_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(status,indent=2),encoding='utf-8'); tmp.replace(status_path)
    def download(name):
        path=dest/name; part=dest/(name+'.part')
        if path.exists():
            saved=status['files'].get(name,{})
            if saved.get('phase')!='complete' or path.stat().st_size!=saved.get('bytes'):
                raise RuntimeError(f'Existing unverified file: {path}')
            print(f'{name}: already complete',flush=True); return
        offset=part.stat().st_size if part.exists() else 0
        url=f'https://drive.usercontent.google.com/download?id={FILES[name]}&export=download&confirm=t'
        headers={'Range':f'bytes={offset}-'} if offset else {}
        update(name,phase='connecting',bytes=offset,source_id=FILES[name])
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
            if 'text/html' in response.headers.get('Content-Type',''):
                raise RuntimeError('Server returned an HTML error/confirmation, not dataset bytes.')
            if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                raise RuntimeError('Server did not honor resume range; partial file preserved.')
            length=int(response.headers.get('Content-Length','0'))
            if not length: raise RuntimeError('Missing response size; refusing an unverified download.')
            total=offset+length
            if shutil.disk_usage(dest).free<length+20*2**30:
                raise RuntimeError('Insufficient space to download while retaining 20 GiB free.')
            digest=hashlib.sha256()
            if offset:
                with part.open('rb') as existing:
                    for chunk in iter(lambda:existing.read(2**20),b''): digest.update(chunk)
            start=time.monotonic(); last=start; count=offset
            update(name,phase='downloading',bytes=count,total_bytes=total)
            with part.open('ab' if offset else 'wb') as output:
                while True:
                    chunk=response.read(2**20)
                    if not chunk: break
                    if count==0:
                        if name.endswith('.gz') and not chunk.startswith(b'\x1f\x8b'):
                            raise RuntimeError('Response is not a gzip archive.')
                        if name.endswith('.json') and not chunk.lstrip().startswith(b'{'):
                            raise RuntimeError('Response is not JSON annotations.')
                    output.write(chunk); digest.update(chunk); count+=len(chunk)
                    now=time.monotonic()
                    if now-last>=10:
                        speed=(count-offset)/max(now-start,1)
                        update(name,bytes=count,total_bytes=total,bytes_per_second=speed)
                        print(f'{name}: {count/total:.1%}, {count/2**30:.2f}/{total/2**30:.2f} GiB',flush=True)
                        last=now
            if count!=total: raise RuntimeError(f'Incomplete response: {count}/{total} bytes.')
            part.replace(path)
            update(name,phase='complete',bytes=count,total_bytes=total,sha256=digest.hexdigest(),path=str(path))
            print(f'{name}: complete, SHA-256 {digest.hexdigest()}',flush=True)
    failures=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,n):n for n in dict.fromkeys(args.files)}
        for future in as_completed(futures):
            name=futures[future]
            try: future.result()
            except Exception as error:
                failures+=1; update(name,phase='failed',error=str(error))
                print(f'{name}: FAILED: {error}',flush=True)
    return 2 if failures else 0


if __name__=='__main__': raise SystemExit(main())
