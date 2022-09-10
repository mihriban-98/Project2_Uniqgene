# 1) Model
# Linear model f = wx + b , sigmoid at the end

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 500)
        self.linear2 = nn.Linear(500,200)
        self.linear3 = nn.Linear(200,7)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        y_pred = torch.softmax(self.linear3(x),dim=1)

        return y_pred

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 400
learning_rate = 0.1
batch_size = 20
batch_no = len(X_train) // batch_size

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    for i in range(batch_no):

        start = i * batch_size
        end   = start + batch_size
        x_var = torch.FloatTensor(X_train[start:end])
        y_var = torch.LongTensor(y_train[start:end])

    # Forward pass and loss

        y_pred = model(x_var) 
        loss = criterion(y_pred, y_var.float())

    # Backward pass and update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch+1) % 1 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    
    acc_train, y_pred2 = accuracy(batch_size,y_pred, y_var)
    print(f'accuracy_training: {acc_train.item():.4f}')
       
    with torch.no_grad():
        y_predicted = model(X_test)
        acc, y_predicted2 = accuracy(test_shape,y_predicted, y_test)
        print(f'accuracy_test: {acc.item():.4f}')


        
