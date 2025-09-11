import time


def time_dataset_next(dloader, n=3, tag="train"):
    it = iter(dloader)
    for k in range(n):
        t0 = time.time()
        try:
            next(it)
            dt = time.time() - t0
            print(f"[DL/{tag}] next() #{k}: {dt:.3f}s")
        except StopIteration:
            print(f"[DL/{tag}] StopIteration after {k} batches")
            break


import faulthandler, signal, time


def _dump(*_):
    print("[G1] watchdog dump:")
    faulthandler.dump_traceback()
    faulthandler.dump_traceback(all_threads=True)


def timeout_trace(llm, tokens):
    signal.signal(signal.SIGALRM, _dump)
    signal.alarm(60)  # dump after 60s
    with llm.trace(tokens) as tracer:
        print("[G1] in trace")
    signal.alarm(0)
