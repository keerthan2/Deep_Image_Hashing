
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

def train(net,train_loader,criterion,optimizer,epoch_num,device):
    print('\nEpoch: %d' % epoch_num)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(total=math.ceil(len(train_loader)), desc="Training") as pbar:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum()
            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss), 'accuracy': '{:.2%}'.format(correct.item() / total)})
            pbar.update(1)
    pbar.close()
    return net

def evaluate(net,test_loader,criterion,best_val_acc,save_name,device):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        with tqdm(total=math.ceil(len(test_loader)), desc="Testing") as pbar:
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).sum()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss), 'accuracy': '{:.2%}'.format(correct.item() / total)})
                pbar.update(1)
        pbar.close()
        acc = 100 * int(correct) / int(total)
        if acc > best_val_acc:
            torch.save(net.state_dict(),save_name)
            best_val_acc = acc
        return test_loss / (batch_idx + 1), best_val_acc