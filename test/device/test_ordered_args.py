import ctypes, socket, struct, unittest
from unittest.mock import patch
from tinygrad import Tensor, Device
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.system import RemoteCmd, REMOTE_RESP

class TestOrderedArgsIntegration(unittest.TestCase):
  def test_null_submission(self):
    (Tensor.ones(4, device="NULL") + 1).realize()

  def test_remote_raw_program(self):
    from extra.remote import serve
    # The wire protocol carries already-ordered uint64 arguments, with no TinyELF signature.
    compiler = ClangRenderer(Target("CPU", renderer="CLANG", arch=Device["CPU"].arch)).compiler
    lib = compiler.compile_cached("void remote_args(int value, int *out, int bias) { *out = value + bias; }")
    out = ctypes.c_int()
    client, server = socket.socketpair()
    with patch.object(serve, "programs", []), client, server:
      client.sendall(lib)
      serve.handle(server, RemoteCmd.LOAD_PROG, 0, 0, len(lib), 0, 0)
      status, program, _ = struct.unpack(REMOTE_RESP, client.recv(struct.calcsize(REMOTE_RESP)))
      self.assertEqual(status, 0)
      for value, bias in ((3, 7), (9, 2)):
        client.sendall(struct.pack("<3Q", value, ctypes.addressof(out), bias))
        serve.handle(server, RemoteCmd.EXEC_PROG, 0, 0, program, 3, 1)
        self.assertEqual(struct.unpack(REMOTE_RESP, client.recv(struct.calcsize(REMOTE_RESP)))[0], 0)
        self.assertEqual(out.value, value + bias)

if __name__ == "__main__": unittest.main()
