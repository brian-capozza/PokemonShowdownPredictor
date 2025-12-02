import torch
from NN import PokemonFullNet

if torch.cuda.is_available():
  print("GPU detected")
  device = torch.device('cuda')
else:
  print("GPU not detected")
  device = torch.device('cpu')


def trainingloop(trainLoader, valLoader)
    model = PokemonFullNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCELoss(reduction='sum')

    for epoch in range(100):
        trainLoss = 0
        totalCount = 0
        correct = 0

        #Training
        model.train()
        for features, target in trainLoader:

            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(features)
            loss = loss_function(output, target)

            #The answer is either 0 or 1, so we round the answer outputs.
            pred = torch.round(output)

            correct += pred.eq(target).sum().item()

            trainLoss += loss.item()
            totalCount += features.size(0)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Train Loss: {trainLoss / totalCount}')
            print(f'Train Accuracy: {correct / totalCount}')


        model.eval()
        with torch.no_grad():
            valLoss = 0
            correct = 0
            for features, target in valLoader:
                features, target = features.to(device), target.to(device)

                output = model(features)
                valLoss += loss_function(output, target).item()

                pred = torch.round(output)
                correct += pred.eq(target).sum().item()

        if epoch % 10 == 0:
            print(f'Val Loss: {valLoss / len(valLoader.dataset)}')
            print(f'Val Accuracy: {correct / len(valLoader.dataset)}')
            print()