
import pickle
import pickletools

try:
    import lz4.frame
except ModuleNotFoundError as e:
    print('lz4 module not found, compression not supported.')
    pass


def pickle_and_compress(obj, compression=None, *args, **kwargs):
    """
    Pickle and compress object.
    :param obj: Object to compress. Must be picklable.
    :param compression: Compression type. Currently only 'lz4' is supported.
    :return: Compressed data.
    """
    if compression == 'lz4':
        picklestring = pickle.dumps(obj, *args, **kwargs)
        picklestring = pickletools.optimize(picklestring)
        return lz4.frame.compress(picklestring, return_bytearray=True)
    else:
        picklestring = pickle.dumps(obj, *args, **kwargs)
        return pickletools.optimize(picklestring)


def decompress_and_unpickle(obj, compression=None, *args, **kwargs):
    """
    Decompress and unpickle object.
    :param obj: Object to uncompress. Must be picklable.
    :param compression: Compression type. Currently only 'lz4' is supported.
    :return: Uncompressed data.
    """
    if compression == 'lz4':
        picklestring = lz4.frame.decompress(obj, return_bytearray=True)
        return pickle.loads(picklestring, *args, **kwargs)
    else:
        return pickle.loads(obj, *args, **kwargs)
