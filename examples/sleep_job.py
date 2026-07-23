from __future__ import annotations

import time


def main() -> None:
    print("sleep job starting", flush=True)
    time.sleep(2)
    print("sleep job done", flush=True)


if __name__ == "__main__":
    main()
