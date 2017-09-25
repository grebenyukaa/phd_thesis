import scipy as sp
import scipy.linalg as lp
import numpy as np
import gzip

VALUE_CAP = 1000
MAX_DIGITS = 3

OUPTUT_STEP = 500
DUMP_INFO_STEP = 50

FILE_MASK = '{name}_{cols}x{rows}_{samples}_{comment}.{extension}'
FILE_EXT = 'tsv.gz'
COLSEP = '\t'
ROWSEP = '\n'

def ndarrayToFlat(arr):
	return list(map(lambda elem: str(elem), arr.flatten()))

###

def generateLU(sampleCount, rowCount, columnCount):
	A, P, L, U = [], [], [], []
	for i in range(sampleCount):
		a = np.around(np.random.rand(rowCount, columnCount) * VALUE_CAP, MAX_DIGITS)
		A += [a]

		p, l, u = lp.lu(a)
		P += [p]
		L += [l]
		U += [u]
	return A, P, L, U

def dumpLU(outFile, A, P, L, U, colsep, rowsep, filename, withP):
	count = 0
	total = len(A)

	for a, p, l, u in zip(A, P, L, U):
		flatP = ndarrayToFlat(p)
		strData = colsep.join([
			str(ndarrayToFlat(a) + flatP if withP else ndarrayToFlat(np.dot(p, a))),
			str(flatP if withP else []),
			str(ndarrayToFlat(l)),
			str(ndarrayToFlat(u))
		]) + rowsep
		outFile.write(strData.encode())

		count += 1
		if count % DUMP_INFO_STEP == 0:
			print('  dumping: {2:5}% ({0} of {1})'.format(count, total, str(np.round(100 * count / total))), end = '\r')

	print()


def generateData(name, sampleCount, size, withP = False):
	print('Requested {0} samples of size {1}x{1} of LU decompositions...'.format(sampleCount, size))

	filename = FILE_MASK.format(name = name, cols = size, rows = size, samples = sampleCount, comment = 'P' if withP else 'noP', extension = FILE_EXT)
	with gzip.open(filename, 'wb') as of:
		remainingCount = sampleCount
		while remainingCount > 0:
			curSampleCount = OUPTUT_STEP if remainingCount - OUPTUT_STEP >= 0 else remainingCount
			remainingCount = remainingCount - curSampleCount

			a, p, l, u = generateLU(curSampleCount, size, size)
			dumpLU(of, a, p, l, u, COLSEP, ROWSEP, filename, withP)

			genCnt = sampleCount - remainingCount
			print('Generated: {2:5}% ({0} of {1})'.format(genCnt, sampleCount, str(np.round(100 * genCnt / sampleCount))))