import subprocess, os, glob, time

roots = ['G:/More-models', 'H:/OLLAMA-Models/GGUF']
files = sorted(set(glob.glob(r + '/*.gguf' for r in roots for _ in [0])) ) if False else None
import itertools
files = []
for r in roots:
    for dirpath, _, names in os.walk(r):
        for n in names:
            if n.endswith('.gguf'):
                files.append(os.path.join(dirpath, n))
files.sort()

out_path = 'matrix_results.txt'
with open(out_path, 'w') as out:
    out.write('file\tcpu\tdx\n')
    for f in files:
        name = os.path.basename(f)
        cpu = dx = 'ERR'
        try:
            p = subprocess.run(['build-cpu/zen_generate_ids.exe', f, '--prompt-ids', '1,2,3,4', '--max-new', '2'],
                               capture_output=True, timeout=180)
            cpu = p.returncode
        except Exception as e:
            cpu = f'EXC:{e}'
        try:
            env = dict(os.environ); env['ZEN_BACKEND'] = 'd3d12'
            p = subprocess.run(['build-dx12/zen_generate_ids.exe', f, '--prompt-ids', '1,2,3,4', '--max-new', '2'],
                               capture_output=True, timeout=180, env=env)
            dx = p.returncode
        except Exception as e:
            dx = f'EXC:{e}'
        try:
            out.write(f'{name}\t{cpu}\t{dx}\n'); out.flush()
        except BrokenPipeError:
            pass
with open(out_path, 'a') as out:
    out.write('done\n')
