import numpy as np

# inputs are numpy arrays of shape b x r x c
# array values are only 0's or 1's
# returns batch detection accuracy
def getBatchDetectionAcc(label_mask, pred_mask):
	assert label_mask.shape == pred_mask.shape
	assert ((label_mask == 0) | (label_mask == 1)).all()
	assert ((pred_mask == 0) | (pred_mask == 1)).all()

	masks = getIndividualMasks(label_mask[0])
	detection = []
	for mask in masks:
		mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
		mask = np.repeat(mask, label_mask.shape[0], axis=0)
		intersection = mask * pred_mask
		num_ones = (intersection == 1).sum(axis=(1,2))
		num_ones[num_ones > 0] = 1
		detection.append(num_ones)
	detection = np.column_stack(detection)
	acc = detection.mean(axis=-1)
	batch_acc = acc.mean(axis=-1)

	return batch_acc
	
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
	[1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
])

mask2 = np.array([
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
	[1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
	[1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
])

mask3 = np.array([
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
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

batch1 = np.stack((mask2, mask2))
batch2 = np.stack((mask1, mask3))
print(getBatchDetectionAcc(batch1, batch2))

# masks = getIndividualMasks(mask4)
# for mask in masks:
# 	print(mask)