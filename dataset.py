import lib

class Hfdata (Dataset):
    def __init__(self, file):
        self.file = h5py.File(file, 'r')
        self.data = h5py.file['data'][:]
        self.labels = h5py.file['label'][:]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
    def __len__(self):
        return len(self.data)

data_sp_tr = Hfdata("add your 3D spatial maps training data.h5")
dataloader_sp_tr = DataLoader(data_sp_tr, batch_size=8, shuffle=True)

data_ts_tr = Hfdata("add your 1D time series training data.h5")
dataloader_ts_tr = DataLoader(data_ts_tr, batch_size=8, shuffle=True)

data_sp_val = Hfdata('add your 3D spatial maps Validation data.h5')
dataloader_sp_val = DataLoader(data_sp_val, batch_size=8, shuffle=False)

data_ts_val = Hfdata('add your 1D time series Validation data.h5')
dataloader_ts_val = DataLoader(data_ts_val, batch_size=8, shuffle=False)

class Hfdata_test(Dataset):
    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        self.data = []
        with h5py.File(f"{data_dir}/{file_name}", "r") as f:
            self.data.append(f["data"][:])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
       
        return x
    
folder_path_sp = 'Add your 3D spatial maps testing data'
file_list_sp = glob.glob(os.path.join(folder_path_sp, '*.h5'))
folder_path_ts = 'Add your 1D time series testing data'
file_list_ts = glob.glob(os.path.join(folder_path_ts, '*.h5'))