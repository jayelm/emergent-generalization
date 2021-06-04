import pandas as pd
import numpy as np

attrs = pd.read_csv('CUB_200_2011/attributes/class_attribute_labels_continuous.txt', sep=' ', header=None).to_numpy()
attrnames = list(pd.read_csv('attributes.txt', sep=' ', header=None, names=['asdf', 'attrnames'])['attrnames'])
attrs = attrs > 50

laysan = np.where(attrs[1, :])[0]
print([attrnames[x] for x in laysan])
painted = np.where(attrs[15, :])[0]
print([attrnames[x] for x in painted])
