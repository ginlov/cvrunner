import pathlib
from unittest import mock
import argparse

import cvrunner.cli as cli

class DummyArgs:
    def __init__(self):
        self.exp = "tests/test_generator/mnist_components.py"
        self.build = False
        self.run_local = True
        self.k8s = False
        self.image = "oel20/cvrunner:latest"

def test_run_local_gpu(monkeypatch):
    args = DummyArgs()
    # convert to namespace using mock
    args = mock.MagicMock(**vars(args))

    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
    with mock.patch("subprocess.run") as m_run:
        cli.run_local(args)
        assert m_run.called
        assert "torchrun" in m_run.call_args[0][0]

def test_run_local_real_subprocess():
    args = DummyArgs()
    # convert to namespace not using mock
    args = argparse.Namespace(**vars(args))
    print(args)

    # Do NOT patch subprocess.run here
    cli.run_local(args)

