import dataset

class Spatial(nn.Module):
    def __init__(self):
        super(Spatial, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=7, padding=3)
        self.bn1  = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, padding=2)
        self.bn2  = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=2, padding=1)
        self.bn3  = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)
        self.d1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*10*12*10, 128)
        self.d2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.d3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool3(x)
        x = self.d1(x)
        #print('x_shape:',x.shape)
        x = x.view(-1, 64*10*12*10)
        x = F.relu(self.fc1(x))
        x = self.d2(x)
        x = F.relu(self.fc2(x))
        x = self.d3(x)
        x = self.fc3(x)
        return x
    
class Temporal(nn.Module):
    def __init__(self):
        super(Temporal, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.d1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*76,128)
        self.d2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.d3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool3(x)
        x = self.d1(x)
        #print('x_shape:',x.shape)
        x = x.view(-1, 64*76)
        x = F.relu(self.fc1(x))
        x = self.d2(x)
        x = F.relu(self.fc2(x))
        x = self.d3(x)
        x = self.fc3(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, spatial_model, temporal_model):
        super(Ensemble, self).__init__()

        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(4, 2)

    def forward(self, x, y):
        x = self.spatial_model(x)
        x = self.dropout(x)
        y = self.temporal_model(y)
        y = self.dropout(y)
        z = torch.cat((x, y), dim=1)
        z = self.fc(z)
        return z
# Initialize the two models
spatial_model = Spatial()
temporal_model = Temporal()
ensemble_model = Ensemble(spatial_model, temporal_model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ensemble_model.parameters(), lr = 0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
n_total_steps = len(dataloader_sp_tr)
num_epochs = 20
writer = SummaryWriter()
acc = []
f1 = BinaryF1Score()


a = BinaryAUROC(thresholds=None)

for epoch in range(num_epochs):
    print(f"traing number: {epoch+1}")
    correct_t = 0
    total_t = 0
    for i, ((data_spatial, labels), (data_temporal, lab)) in enumerate(zip(dataloader_sp_tr, dataloader_ts_tr)):

        data_spatial = data_spatial.permute(0, 4, 1, 2, 3)  # permute the tensor to [batch_size, channels, depth, height, width]
        data_spatial = data_spatial.float()
        labels = labels.long()
        data_temporal = data_temporal.view(-1, 1, 609)  # permute the tensor to [batch_size, channels, depth, height, width]
        data_temporal = data_temporal.float()
        outputs = ensemble_model(data_spatial, data_temporal)
        loss = criterion(outputs, labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 2 ==0:
            print(f"epoch {epoch+1} /{num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}")
        scheduler.step()
        _, predicted_t = torch.max(outputs.data, 1)
        total_t += labels.size(0)
        correct_t += (predicted_t == labels).sum().item()
    accuracy_t = 100 * correct_t / total_t
    print("Accuracy of the training: {:.2f}%".format(accuracy_t))
    writer.add_scalar('Loss/train', loss.item(), epoch+1)
    writer.add_scalar('Loss/train_', loss, epoch+1)
    writer.add_scalar('acc/train_', accuracy_t, epoch+1)

    ensemble_model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for  (data_spatial,labels), (data_temporal,lab)  in zip(dataloader_sp_val, dataloader_ts_val):
            data_spatial = data_spatial.permute(0, 4, 1, 2, 3)  # permute the tensor to [batch_size, channels, depth, height, width]

            data_spatial = data_spatial.float()
            labels = labels.long()
            batch_size = data_spatial.size(0)
            data_temporal = data_temporal.view(-1, 1, 609)  # permute the tensor to [batch_size, channels, depth, height, width]
            data_temporal = data_temporal.float()
            outputs = ensemble_model(data_spatial, data_temporal)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            validation_loss = criterion(outputs, labels).item()
            f1(predicted, labels)
            a(predicted, labels)




        accuracy = 100 * correct / total
        acc.append(accuracy)
        f1_score = f1.compute()
        auc = a.compute()

        print('Accuracy of the ensemble model on the validation set: {:.2f}%'.format(accuracy))
        print('loss of the ensemble model on the validation set: {:.4f}'.format(validation_loss))
        print('F1 of the ensemble model on the validation set: {:.2f}%'.format(f1_score))
        print('AUC of the ensemble model on the validation set: {:.2f}%'.format(auc))

        writer.add_scalar('Accuracy/val', accuracy, epoch+1)
        writer.add_scalar('loss/val', validation_loss, epoch+1)
        writer.add_scalar('f1/val', f1_score, epoch+1)
        writer.add_scalar('auc/val', auc, epoch+1)
    if accuracy > 75 and auc > 75:
        torch.save(ensemble_model.state_dict(), f'final_Ensamble_model_{epoch+1}_{accuracy}_{auc}.pt')

    predictions = []
    for file_path_sp, file_path_ts in zip(file_list_sp, file_list_ts):
        file_name_sp = os.path.basename(file_path_sp)
        file_name_ts = os.path.basename(file_path_ts)
        print(f"Prediction on {file_name_sp} and {file_name_ts}")
        dataset_sp_test = Hfdata_test(folder_path_sp, file_name_sp)
        dataloader_sp_test = DataLoader(dataset_sp_test, batch_size=batch_size, shuffle=False)
        dataset_ts_test = Hfdata_test(folder_path_ts, file_name_ts)
        dataloader_ts_test = DataLoader(dataset_ts_test, batch_size=batch_size, shuffle=False)
        for i, ((data_sp), (data_ts)) in enumerate(zip(dataloader_sp_test, dataloader_ts_test)):
        # prepare the data
        #print(f'sp tensor shape: {data_sp.shape}')
            data_sp = data_sp.permute(0, 4, 1,2, 3)
            data_sp = data_sp.float()
        #print(f'tm tensor shape: {data_tm.shape}')
            data_ts = data_ts.view(-1, 1, 609).float()

        # pass the data through the model
            with torch.no_grad():
                outputs = ensemble_model(data_sp, data_ts)

        # obtain the predicted labels from the predicted outputs
            _, predicted_labels = torch.max(outputs.data, 1)

            predicted_labels = predicted_labels.item()
            if predicted_labels ==0:
                print(f"signal or noise: noise")
            else:
                print(f"signal or noise: signal")
       # print(predicted_labels)

        # append the predicted labels to the list of predictions
            predictions.append(predicted_labels)

    print(predictions)

writer.flush()
writer.close()
torch.save(ensemble_model.state_dict(), 'final_Ensamble_model.pt')
avg_acc = sum(acc) / len(acc)
print('Average Accuracy of the ensemble model on the validation set: {:.2f}%'.format(avg_acc))
# print the predictions
print(predictions)