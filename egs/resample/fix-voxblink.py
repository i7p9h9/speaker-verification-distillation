#!/usr/bin/env python3
"""Repair OGG packet timestamps with bounded parallelism; optionally write Parquet durations."""
import argparse
import os
import shutil
import subprocess
import sys
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def duration_ms(path):
    p = subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=sample_rate','-of','default=noprint_wrappers=1:nokey=1',str(path)],capture_output=True,text=True,check=True)
    sr = int(p.stdout.strip())
    cmd = ['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:a:0','-f','s16le','-acodec','pcm_s16le','-ac','1','-ar',str(sr),'pipe:1']
    # stderr to a temporary file prevents deadlocks if FFmpeg produces verbose errors.
    import tempfile
    with tempfile.TemporaryFile() as stderr, subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=stderr) as proc:
        byte_count = 0
        while True:
            chunk = proc.stdout.read(1024*1024)
            if not chunk:
                break
            byte_count += len(chunk)
        status = proc.wait()
        if status:
            stderr.seek(0)
            raise RuntimeError(f'Decode failed: {stderr.read(4096).decode(errors="replace")}')
    return sr, round(byte_count / 2 * 1000 / sr)


def process(src, dst, rel, overwrite, metadata):
    if dst.exists() and not overwrite:
        if metadata and src.suffix.lower() == '.ogg':
            sr, ms = duration_ms(dst)
            return rel.as_posix(), sr, ms
        return None
    dst.parent.mkdir(parents=True,exist_ok=True)
    ogg = src.suffix.lower() == '.ogg'
    tmp = dst.with_name(f'.{dst.stem}.{uuid.uuid4().hex}.tmp' + ('.ogg' if ogg else dst.suffix))
    try:
        if ogg:
            subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(src),'-map','0:a:0','-c:a','copy','-bsf:a',r'setts=duration=if(gt(DURATION\,1024)\,512\,DURATION)',str(tmp)],capture_output=True,text=True,check=True)
            row = (rel.as_posix(), *duration_ms(tmp)) if metadata else None
        else:
            shutil.copy2(src,tmp)
            row = None
        os.replace(tmp,dst)
        return row
    finally:
        tmp.unlink(missing_ok=True)


def iter_files(source, destination):
    for root, dirs, names in os.walk(source):
        root = Path(root)
        rel_dir = root.relative_to(source)
        (destination / rel_dir).mkdir(parents=True, exist_ok=True)
        for name in names:
            path = root / name
            if path.is_file() and not path.is_symlink():
                yield path, path.relative_to(source)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('destination',type=Path)
    ap.add_argument('--metadata',type=Path,help='Optional Parquet output')
    ap.add_argument('-j','--workers',type=int,default=4)
    ap.add_argument('--overwrite',action='store_true')
    args = ap.parse_args()
    source, dest = args.source.resolve(), args.destination.resolve()
    if not source.is_dir(): ap.error(f'Source not found: {source}')
    if source == dest or source in dest.parents: ap.error('Destination must be outside source')
    if args.workers < 1: ap.error('workers must be positive')
    if not shutil.which('ffmpeg'): ap.error('ffmpeg not found')
    if args.metadata and not shutil.which('ffprobe'): ap.error('ffprobe not found')
    parquet = args.metadata.resolve() if args.metadata else None
    if parquet and (parquet == source or source in parquet.parents): ap.error('Metadata output cannot be inside source')
    if parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError: ap.error('Parquet metadata requires pyarrow: uv add pyarrow')
        schema = pa.schema([('relative_path',pa.string()),('samplerate',pa.int32()),('duration_ms',pa.int64())])
        parquet.parent.mkdir(parents=True,exist_ok=True)
        temp_parquet = parquet.with_name(f'.{parquet.name}.{uuid.uuid4().hex}.tmp')
        writer = pq.ParquetWriter(temp_parquet,schema,compression='zstd')
    else:
        writer = None
        temp_parquet = None
    rows, errors = [], []
    completed = 0
    print(f'Starting: {source} -> {dest} ({args.workers} workers)',flush=True)
    def complete(future, rel):
        nonlocal completed
        try:
            row = future.result()
            if row is not None: rows.append(row)
        except Exception as exc:
            errors.append((rel.as_posix(),str(exc)))
            if len(errors) <= 20: print(f'ERROR {rel}: {exc}',file=sys.stderr,flush=True)
        completed += 1
        if writer is not None and len(rows) >= 10000: flush_rows()
        if completed % 10000 == 0: print(f'Processed {completed:,} files; errors {len(errors):,}',flush=True)
    def flush_rows():
        if rows:
            columns = list(zip(*rows))
            table = pa.Table.from_arrays([pa.array(columns[0],type=pa.string()),pa.array(columns[1],type=pa.int32()),pa.array(columns[2],type=pa.int64())],schema=schema)
            writer.write_table(table)
            rows.clear()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = deque()
            for src, rel in iter_files(source,dest):
                pending.append((pool.submit(process,src,dest/rel,rel,args.overwrite,bool(parquet)),rel))
                if len(pending) >= args.workers * 4:
                    fut, old_rel = pending.popleft()
                    complete(fut,old_rel)
            while pending:
                fut, rel = pending.popleft()
                complete(fut,rel)
        if writer is not None:
            flush_rows()
            writer.close()
            writer = None
            os.replace(temp_parquet,parquet)
            print(f'Parquet saved: {parquet}',flush=True)
    finally:
        if writer is not None: writer.close()
        if temp_parquet is not None: temp_parquet.unlink(missing_ok=True)
    if errors:
        log = dest / 'conversion_errors.log'
        with log.open('w',encoding='utf-8') as f:
            for rel, error in errors: f.write(f'{rel}\t{error}\n')
        print(f'Errors: {len(errors)}. See {log}',file=sys.stderr,flush=True)
        return 1
    print(f'Done. Processed {completed:,} files',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
