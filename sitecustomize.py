import sys, time, builtins
orig_import = builtins.__import__

def traced_import(name, globals=None, locals=None, fromlist=(), level=0):
    t = time.time()
    try:
        result = orig_import(name, globals, locals, fromlist, level)
        print(f"IMPORT_OK {t:.6f} {name}")
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"IMPORT_ERR {t:.6f} {name} {e}")
        sys.stdout.flush()
        raise

builtins.__import__ = traced_import
print('sitecustomize loaded')
sys.stdout.flush()
