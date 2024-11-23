import scipy.io as sio
data_path = 'paviaU_7gt.mat'
mat_data = sio.loadmat(data_path)
print(mat_data.keys())
