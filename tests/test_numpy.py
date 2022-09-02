import unittest
from mpyc.numpy import np


class Arithmetic(unittest.TestCase):

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_item_shape(self):
        for shape in ((), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)):
            a = np.empty(shape)
            for key in ((0,) * a.ndim,
                        (slice(None),) * a.ndim,
                        (0, np.newaxis) * a.ndim,
                        (0,) * a.ndim + (...,),
                        ...,
                        (..., np.newaxis),
                        (np.newaxis, ...),
                        (np.newaxis, ..., np.newaxis)):
                self.assertEqual(np._item_shape(shape, key), a[key].shape)
            key = (0,) * (a.ndim + 1)
            self.assertRaises(IndexError, np._item_shape, shape, key)
            key = (..., ...)
            self.assertRaises(IndexError, np._item_shape, shape, key)
        shape = (1, 2, 3, 4, 5, 6)
        a = np.empty(shape)
        for key in (0, -1, (0, ..., 0), slice(None)):
            self.assertEqual(np._item_shape(shape, key), a[key].shape)
        self.assertRaises(ValueError, np._item_shape, shape, slice(1, 1, 0))  # step cannot be 0

        # NumPy User Guide "Advanced indexing and index tricks" examples:
        a = np.arange(12)**2
        shape = a.shape
        key = np.array([1, 1, 3, 8, 5])
        self.assertEqual(np._item_shape(shape, key), a[key].shape)
        key = range(4, 8)
        self.assertEqual(np._item_shape(shape, key), a[key].shape)
        key = np.array([[3, 4], [9, 7]])
        self.assertEqual(np._item_shape(shape, key), a[key].shape)
        a = a.reshape(3, 4)
        shape = a.shape
        i = np.array([[0, 1], [1, 2]])
        j = np.array([[2, 1], [3, 3]])
        for key in ((i, j), (i, 2), (..., j)):
            self.assertEqual(np._item_shape(shape, key), a[key].shape)
        key = (i, j, 0)
        self.assertRaises(IndexError, np._item_shape, shape, key)
        key = a > 16
        self.assertEqual(np._item_shape(shape, key), a[key].shape)
        key = ((a > 16)[slice(None), -1], slice(2, 4))
        self.assertEqual(np._item_shape(shape, key), a[key].shape)
        b1 = np.array([False, True, True])
        b2 = np.array([True, False, True, False])
        for key in ((b1, ...), (b1,), (..., b2), (b1, b2)):
            self.assertEqual(np._item_shape(shape, key), a[key].shape)
        a = np.empty((3, 5, 4, 6))
        key = (b1, slice(None), b2)
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)
        a = np.empty((3, 5, 6, 4))
        key = (b1, ..., b2)
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)

        # Weird boolean indexing:
        a = np.empty((2, 3, 4))
        key = False
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)
        key = True
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)
        key = (False, True, ...)
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)
        key = (True, [0, 1], True, True, [1], [[2]])
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)
        key = (True, [[0, 1]], True, np.newaxis, True, [1], [[2]])
        self.assertEqual(np._item_shape(a.shape, key), a[key].shape)


if __name__ == "__main__":
    unittest.main()
