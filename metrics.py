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

# input is a masked numpy array of r x c
# returns the individual masks
def getIndividualMasks(mask):
	masks = []
	explore = np.zeros_like(mask)
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] == 1 and explore[i][j] == 0:
				k, l = i, j
				while k != mask.shape[0] and mask[k][l] == 1:
					k += 1
				height = k - i
				k, l = i, j
				while l != mask.shape[1] and mask[k][l] == 1:
					l += 1
				width = l - j
				found_mask = np.zeros_like(mask)
				for k in range(mask.shape[0]):
					for l in range(mask.shape[1]):
						if (k >= i) and (k < (i + height)) and (l >= j) and (l < (j + width)):
							found_mask[k][l] = 1
							explore[k][l] = 1
				masks.append(found_mask)
	return masks

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

mask4 = np.array([
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 1, 0, 0, 1, 1]
])

# batch1 = np.stack((mask1, mask3))
# batch2 = np.stack((mask2, mask2))
# print(getBatchIoUAcc(batch1, batch2))

masks = getIndividualMasks(mask4)
for mask in masks:
	print(mask)