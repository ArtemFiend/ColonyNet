"""PyInstaller entry point that preserves package-relative imports."""
import sys

from colonynet.app import main, run_once_cli, self_test


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    if "--run-once" in sys.argv:
        raise SystemExit(run_once_cli(sys.argv))
    main()
