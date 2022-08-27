from cProfile import label
import numpy as np

# inputs are numpy arrays of same shape
# array values are only 0's or 1's
def evalPred(mask1, mask2):
	assert mask1.shape == mask2.shape
	assert ((mask1 == 0) | (mask1 == 1)).all()
	assert ((mask2 == 0) | (mask2 == 1)).all()

	# IoU calculation
	intersection = mask1 * mask2
	num_ones = (intersection == 1).sum()
	union = mask1 + mask2
	num_not_zeros = (union != 0).sum()
	iou = num_ones/num_not_zeros

	if iou >= 0.5:
		# correct detection
		return 1
	else:
		# wrong detection
		return 0

mask1 = np.array([
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
])

mask2 = np.array([
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
	[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
	[0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
])

print(evalPred(mask1, mask2))