from going_modular.utils import compute_iou, calculate_precision_recall_f1
from going_modular import config

import torch


def one_step_train(model, 
                   train_dataloader,
                   loss_fn,
                   optimizer,
                   device):
    
    model = model.to(device)
    model.train()

    train_loss, train_iou = 0, 0
    t_precision, t_recall, t_f1 = 0, 0, 0

    for i, (inputs, targets) in enumerate(train_dataloader):

        inputs, targets = inputs.to(device), targets.to(device).to(dtype=torch.int64)

        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs shape : torch.Size([2, 150, 400, 400])

        # Reshape the target mask to (batch_size, height, width)
        targets = targets.squeeze(1).to(torch.float32)
        outputs = outputs.squeeze(1)
        outputs = torch.sigmoid(outputs)
        # print(f'targets : {targets.shape} {targets.dtype}')
        # print(f'outputs : {outputs.shape} {outputs.dtype}')
        # targets shape : torch.Size([2, 400, 400])

        # Calculate CrossEntropy loss
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()
        

        loss.backward()
        optimizer.step()

        num_classes = config.NUM_CLASSES
        
        # predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        import numpy as np
        predictions = outputs.detach().cpu().numpy()
        predictions = np.where(1, predictions>.5, 0)
        # iou : predictions shape : (batch, image_size, image_size)

        # Assuming targets are already numpy arrays
        targets = targets.squeeze(1).detach().cpu().numpy()
        # iou : targets shape : (batch, image_size, image_size)

        # Calculate mIoU for the current batch
        batch_iou = compute_iou(predictions, targets, num_classes)

        train_iou += batch_iou

        # print(f'targets : {targets.shape} {targets.dtype}')
        # print(f'predictions : {predictions.shape} {predictions.dtype}')

        precision, recall, f1 = calculate_precision_recall_f1(targets, predictions)

        t_precision += precision
        t_recall += precision
        t_f1 += precision



        

        # if i % 20 == 0:
        #     print(f'train_mode i is: {i}')

    train_loss = train_loss/len(train_dataloader)
    tain_iou = train_iou/len(train_dataloader)
    
    t_precision = t_precision/len(train_dataloader)
    t_recall = t_recall/len(train_dataloader)
    t_f1 = t_f1/len(train_dataloader)


    # print(f'pre: {t_precision}')
    # print(f'recal: {t_recall}')
    # print(f'f1: {t_f1}')

    return train_loss, tain_iou


def one_step_test(model, 
                  test_dataloader,
                  loss_fn,
                  device):
    
    model = model.to(device)
    model.eval()

    test_loss, test_iou = 0, 0
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(test_dataloader):

            inputs, targets = inputs.to(device), targets.to(device).to(torch.int64)

            outputs = model(inputs)
            # print(f'outputs shape : {outputs.shape}')

            # Reshape the target mask to (batch_size, height, width)
            targets = targets.squeeze(1).to(torch.float32)
            outputs = outputs.squeeze(1)
            outputs = torch.sigmoid(outputs)
            # print(f'targets : {targets.shape} {targets.dtype}')
            # print(f'outputs : {outputs.shape} {outputs.dtype}')
            # targets shape : torch.Size([2, 400, 400])

            # Calculate CrossEntropy loss
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()


            import numpy as np
            predictions = outputs.detach().cpu().numpy()
            predictions = np.where(1, predictions>.5, 0)
            # iou : predictions shape : (batch, image_size, image_size)

            # Assuming targets are already numpy arrays
            targets = targets.squeeze(1).detach().cpu().numpy()
            # iou : targets shape : (batch, image_size, image_size)

            num_classes = config.NUM_CLASSES

            # Calculate mIoU for the current batch
            batch_iou = compute_iou(predictions, targets, num_classes)

            test_iou += batch_iou

            # if i % 20 == 0:
            #     print(f'test_mode i is: {i}')

    test_iou = test_iou / len(test_dataloader)
    test_loss = test_loss/ len(test_dataloader)

    return test_loss, test_iou



def train(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          device,
          epochs):
    
    results = {
            'train_loss':[],
            'train_iou':[],
            'test_loss':[],
            'test_iou':[]
        }
    
    for epoch in range(epochs):

        train_loss, train_iou = one_step_train(model,
                                                train_dataloader,
                                                loss_fn, 
                                                optimizer,
                                                device)

        test_loss, test_iou = one_step_test(model,
                                            test_dataloader,
                                            loss_fn,
                                            device)

        results['train_loss'].append(train_loss)
        results['train_iou'].append(train_iou)
        results['test_loss'].append(test_loss)
        results['test_iou'].append(test_iou)


        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
        )



















































        if epoch == 5:
            f"train_loss: .89 | "
            f"test_iou: .91"

        
    return results



