"""Extract approved dataset archives with path/type checks and resumable verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time


def checked_target(root, member):
    name=PurePosixPath(member.name)
    if name.is_absolute() or '..' in name.parts or any(':' in p or '\\' in p for p in name.parts):
        raise ValueError(f'Unsafe archive path: {member.name}')
    target=root.joinpath(*name.parts).resolve()
    if not target.is_relative_to(root.resolve()): raise ValueError('Archive escaped destination.')
    if not (member.isdir() or member.isfile()): raise ValueError(f'Unsupported archive entry: {member.name}')
    if not name.parts or name.parts[0] not in ('gts','imgs'): raise ValueError('Unexpected dataset archive layout.')
    return target


def extract(archive, root, progress):
    files=total=0
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            path=checked_target(root,member)
            if member.isdir(): path.mkdir(parents=True,exist_ok=True); continue
            path.parent.mkdir(parents=True,exist_ok=True)
            source=stream.extractfile(member)
            if path.exists():
                if path.stat().st_size!=member.size: raise ValueError(f'Existing file size differs: {path}')
                with path.open('rb') as current:
                    while True:
                        block=source.read(2**20)
                        if not block: break
                        if current.read(len(block))!=block: raise ValueError(f'Existing file differs: {path}')
            else:
                temp=path.with_name(path.name+'.extracting')
                with temp.open('wb') as output: shutil.copyfileobj(source,output,2**20)
                if temp.stat().st_size!=member.size: raise ValueError(f'Truncated archive member: {member.name}')
                temp.replace(path)
            files+=1; total+=member.size
            if files%5000==0: progress(files,total,'extracting')
    progress(files,total,'complete')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args(); root=args.destination.resolve(); root.mkdir(parents=True,exist_ok=True)
    archives=[root/'downloads'/n for n in ('gts.tar.gz','imgs.tar.gz')]
    if not all(a.is_file() for a in archives): raise FileNotFoundError('Dataset downloads are incomplete.')
    if shutil.disk_usage(root).free<50*2**30: raise RuntimeError('Need at least 50 GiB free before extraction.')
    status={}; lock=threading.Lock()
    def work(path):
        def progress(files,size,phase):
            with lock:
                status[path.name]={'files':files,'bytes':size,'phase':phase}
                tmp=root/'extraction_status.tmp'
                tmp.write_text(json.dumps(status,indent=2),encoding='utf-8'); tmp.replace(root/'extraction_status.json')
            print(f'{path.name}: {phase}, {files} files, {size/2**30:.2f} GiB',flush=True)
        try: extract(path,root,progress)
        except Exception as error:
            progress(0,0,'failed: '+str(error)); raise
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(work,archives))


if __name__=='__main__': main()
