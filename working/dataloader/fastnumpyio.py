# https://github.com/divideconcept/fastnumpyio/blob/main/fastnumpyio.py
from pathlib import Path
import sys
import numpy as np
import numpy.lib.format
import struct


def save(file, array):
    magic_string = b"\x93NUMPY\x01\x00v\x00"
    header = bytes(
        ("{'descr': '" + array.dtype.descr[0][1] + "', 'fortran_order': False, 'shape': " + str(array.shape) + ", }").ljust(
            127 - len(magic_string)
        )
        + "\n",
        "utf-8",
    )
    if isinstance(file, (str, Path)):
        file = open(file, "wb")
    file.write(magic_string)
    file.write(header)
    file.write(array.tobytes())


def pack(array):
    size = len(array.shape)
    return (
        bytes(array.dtype.byteorder.replace("=", "<" if sys.byteorder == "little" else ">") + array.dtype.kind, "utf-8")
        + array.dtype.itemsize.to_bytes(1, byteorder="little")
        + struct.pack(f"<B{size}I", size, *array.shape)
        + array.tobytes()
    )


def load(file):
    if isinstance(file, (str, Path)):
        file = open(file, "rb")
    header = file.read(128)
    if not header:
        raise FileNotFoundError()
        return None
    descr = str(header[19:25], "utf-8").replace("'", "").replace(" ", "")
    shape = tuple(int(num) for num in str(header[60:120], "utf-8").replace(", }", "").replace("(", "").replace(")", "").split(","))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize)).copy()


def unpack(data):
    dtype = str(data[:2], "utf-8")
    dtype += str(data[2])
    size = data[3]
    shape = struct.unpack_from(f"<{size}I", data, 4)
    datasize = data[2]
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=dtype, buffer=data[4 + size * 4 : 4 + size * 4 + datasize])


if __name__ == "__main__":
    import io
    from timeit import default_timer as timer
    from datetime import timedelta

    iterations = 100000
    testarray = np.random.rand(3, 64, 64).astype("float32")

    start = timer()
    for i in range(iterations):
        buffer = io.BytesIO()
        np.save(buffer, testarray)
        numpy_save_data = buffer.getvalue()
    print("numpy.save:", timedelta(seconds=timer() - start))

    start = timer()
    for i in range(iterations):
        buffer = io.BytesIO()
        save(buffer, testarray)
        fastnumpyio_save_data = buffer.getvalue()
    print("fastnumpyio.save:", timedelta(seconds=timer() - start))

    start = timer()
    for i in range(iterations):
        fastnumpyio_pack_data = pack(testarray)
    print("fastnumpyio.pack:", timedelta(seconds=timer() - start))

    start = timer()
    for i in range(iterations):
        buffer = io.BytesIO(numpy_save_data)
        test_numpy_save = np.load(buffer)
    print("numpy.load:", timedelta(seconds=timer() - start))

    start = timer()
    for i in range(iterations):
        buffer = io.BytesIO(fastnumpyio_save_data)
        test_fastnumpyio_save = load(buffer)
    print("fastnumpyio.load:", timedelta(seconds=timer() - start))

    start = timer()
    for i in range(iterations):
        test_fastnumpyio_pack = unpack(fastnumpyio_pack_data)
    print("fastnumpyio.unpack:", timedelta(seconds=timer() - start))

    print("numpy.save+numpy.load == fastnumpyio.save+fastnumpyio.load:", np.array_equal(test_numpy_save, test_fastnumpyio_save))
    print("numpy.save+numpy.load == fastnumpyio.pack+fastnumpyio.unpack:", np.array_equal(test_numpy_save, test_fastnumpyio_pack))
