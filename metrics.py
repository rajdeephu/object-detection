import numpy as np

# inputs are numpy arrays of shape b x r x c
# array values are only 0's or 1's
# returns batch IoU accuracy
def getBatchIoUAcc(mask1, mask2):
	assert mask1.shape == mask2.shape
	assert ((mask1 == 0) | (mask1 == 1)).all()
	assert ((mask2 == 0) | (mask2 == 1)).all()

	# IoU calculation
	intersection = mask1 * mask2
	num_ones = (intersection == 1).sum(axis=(1,2))
	union = mask1 + mask2
	num_not_zeros = (union != 0).sum(axis=(1,2))
	iou = num_ones/num_not_zeros

	# calculate batch accuracy
	num_correct = (iou >= 0.5).sum()
	acc = num_correct/iou.shape[0]
	
	return acc

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

mask3 = np.array([
	[1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
	[1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
	[1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
	[0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
	[0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
	[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
])

batch1 = np.stack((mask1, mask3))
batch2 = np.stack((mask2, mask2))
print(getBatchIoUAcc(batch1, batch2))