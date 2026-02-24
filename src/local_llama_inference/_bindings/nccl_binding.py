"""ctypes wrapper for NVIDIA NCCL (libnccl.so.2)."""

import ctypes
from typing import List, Optional

from .._bootstrap.finder import get_nccl_library
from ..exceptions import NCCLError, LibraryNotFound


# NCCL Data Types
class NCCLDataType:
    INT8 = 0
    UINT8 = 1
    INT32 = 2
    UINT32 = 3
    INT64 = 4
    UINT64 = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    BFLOAT16 = 9


# NCCL Reduction Operators
class NCCLRedOp:
    SUM = 0
    PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4


# NCCL Result codes
class NCCLResult:
    SUCCESS = 0
    UNINITIALIZED_ERROR = 1
    INVALID_USAGE = 2
    NOT_SUPPORTED_ERROR = 3
    UNKNOWN_ERROR = 4


class NCCLBinding:
    """ctypes wrapper for libnccl.so.2."""

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize NCCL binding.

        Args:
            lib_path: Path to libnccl.so.2 (auto-detected if None)

        Raises:
            LibraryNotFound: If NCCL library not found
        """
        if lib_path is None:
            lib_path = get_nccl_library()

        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise LibraryNotFound(f"Failed to load NCCL library: {e}") from e

        self._setup_signatures()

    def _setup_signatures(self):
        """Setup ctypes function signatures."""
        # Version
        self._lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.ncclGetVersion.restype = ctypes.c_int

        # Communicator init
        self._lib.ncclCommInitAll.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.ncclCommInitAll.restype = ctypes.c_int

        # Error
        self._lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self._lib.ncclGetErrorString.restype = ctypes.c_char_p

    def _check_result(self, result: int) -> None:
        """Check NCCL result code."""
        if result != NCCLResult.SUCCESS:
            error_str = self._lib.ncclGetErrorString(result)
            msg = error_str.decode() if error_str else f"Unknown error {result}"
            raise NCCLError(f"NCCL operation failed: {msg}")

    def get_version(self) -> int:
        """Get NCCL version."""
        version = ctypes.c_int()
        result = self._lib.ncclGetVersion(ctypes.byref(version))
        self._check_result(result)
        return version.value

    def comm_init_all(self, n_devs: int, dev_list: List[int]) -> List:
        """
        Initialize NCCL communicators for all devices.

        Simpler interface for single-process multi-GPU.

        Args:
            n_devs: Number of devices
            dev_list: List of device indices

        Returns:
            List of communicator handles (as integers)

        Raises:
            NCCLError: If initialization fails
        """
        comms = (ctypes.c_void_p * n_devs)()
        dev_array = (ctypes.c_int * n_devs)(*dev_list)

        result = self._lib.ncclCommInitAll(
            ctypes.byref(comms),
            n_devs,
            ctypes.byref(dev_array),
        )
        self._check_result(result)

        return [int(comms[i]) for i in range(n_devs)]

    def comm_destroy(self, comm: int) -> None:
        """Destroy a communicator."""
        result = self._lib.ncclCommDestroy(ctypes.c_void_p(comm))
        self._check_result(result)

    def comm_finalize(self, comm: int) -> None:
        """Finalize a communicator."""
        result = self._lib.ncclCommFinalize(ctypes.c_void_p(comm))
        self._check_result(result)

    def comm_abort(self, comm: int) -> None:
        """Abort a communicator."""
        result = self._lib.ncclCommAbort(ctypes.c_void_p(comm))
        self._check_result(result)

    def comm_count(self, comm: int) -> int:
        """Get number of ranks in communicator."""
        count = ctypes.c_int()
        result = self._lib.ncclCommCount(ctypes.c_void_p(comm), ctypes.byref(count))
        self._check_result(result)
        return count.value

    def comm_cu_device(self, comm: int) -> int:
        """Get CUDA device for this communicator."""
        device = ctypes.c_int()
        result = self._lib.ncclCommCuDevice(ctypes.c_void_p(comm), ctypes.byref(device))
        self._check_result(result)
        return device.value

    def comm_user_rank(self, comm: int) -> int:
        """Get rank of current process."""
        rank = ctypes.c_int()
        result = self._lib.ncclCommUserRank(ctypes.c_void_p(comm), ctypes.byref(rank))
        self._check_result(result)
        return rank.value

    def group_start(self) -> None:
        """Start a group of collective operations."""
        result = self._lib.ncclGroupStart()
        self._check_result(result)

    def group_end(self) -> None:
        """End a group of collective operations."""
        result = self._lib.ncclGroupEnd()
        self._check_result(result)

    # Collective operations (simplified signatures)
    def all_reduce(
        self,
        sendbuf: int,
        recvbuf: int,
        count: int,
        data_type: int,
        op: int,
        comm: int,
        stream: int = 0,
    ) -> None:
        """All-reduce operation."""
        result = self._lib.ncclAllReduce(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_int64(count),
            ctypes.c_int(data_type),
            ctypes.c_int(op),
            ctypes.c_void_p(comm),
            ctypes.c_void_p(stream),
        )
        self._check_result(result)

    def broadcast(
        self,
        sendbuf: int,
        recvbuf: int,
        count: int,
        data_type: int,
        root: int,
        comm: int,
        stream: int = 0,
    ) -> None:
        """Broadcast operation."""
        result = self._lib.ncclBroadcast(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_int64(count),
            ctypes.c_int(data_type),
            ctypes.c_int(root),
            ctypes.c_void_p(comm),
            ctypes.c_void_p(stream),
        )
        self._check_result(result)

    def all_gather(
        self,
        sendbuf: int,
        recvbuf: int,
        count: int,
        data_type: int,
        comm: int,
        stream: int = 0,
    ) -> None:
        """All-gather operation."""
        result = self._lib.ncclAllGather(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_int64(count),
            ctypes.c_int(data_type),
            ctypes.c_void_p(comm),
            ctypes.c_void_p(stream),
        )
        self._check_result(result)

    def reduce_scatter(
        self,
        sendbuf: int,
        recvbuf: int,
        count: int,
        data_type: int,
        op: int,
        comm: int,
        stream: int = 0,
    ) -> None:
        """Reduce-scatter operation."""
        result = self._lib.ncclReduceScatter(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_int64(count),
            ctypes.c_int(data_type),
            ctypes.c_int(op),
            ctypes.c_void_p(comm),
            ctypes.c_void_p(stream),
        )
        self._check_result(result)

    def send(
        self,
        sendbuf: int,
        count: int,
        data_type: int,
        peer: int,
        comm: int,
        stream: int = 0,
    ) -> None:
        """Send operation."""
        result = self._lib.ncclSend(
            ctypes.c_void_p(sendbuf),
            ctypes.c_int64(count),
            ctypes.c_int(data_type),
            ctypes.c_int(peer),
            ctypes.c_void_p(comm),
            ctypes.c_void_p(stream),
        )
        self._check_result(result)

    def recv(
        self,
        recvbuf: int,
        count: int,
        data_type: int,
        peer: int,
        comm: int,
        stream: int = 0,
    ) -> None:
        """Receive operation."""
        result = self._lib.ncclRecv(
            ctypes.c_void_p(recvbuf),
            ctypes.c_int64(count),
            ctypes.c_int(data_type),
            ctypes.c_int(peer),
            ctypes.c_void_p(comm),
            ctypes.c_void_p(stream),
        )
        self._check_result(result)

    def __repr__(self) -> str:
        version = self.get_version()
        return f"NCCLBinding(version={version})"
