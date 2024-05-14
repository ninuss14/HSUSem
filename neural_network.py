import torch
import torch.nn as nn
import torch.nn.functional as F

categories_names = {
    0: "Povolená rýchlosť 20",
    1: "Povolená rýchlosť 30",
    2: "Povolená rýchlosť 50",
    3: "Povolená rýchlosť 60",
    4: "Povolená rýchlosť 70",
    5: "Povolená rýchlosť 80",
    6: "Koniec max. rýchlosti 80",
    7: "Povolená rýchlosť 100",
    8: "Povolená rýchlosť 120",
    9: "Zákaz predbiehania",
    10: "Zákaz predbiehania kamiónom",
    11: "Križovatka s vedľajšou cestou mimo obce",
    12: "Hlavná cesta",
    13: "Daj prednosť v jazde",
    14: "Stop",
    15: "Zákaz vjazdu všetkých vozidiel v oboch smeroch",
    16: "Zákaz vjazdu nákladných automobilov",
    17: "Zákaz vjazdu všetkých vozidiel",
    18: "Pozor iné nebezpečenstvo",
    19: "Zákruta vľavo",
    20: "Zákruta vpravo",
    21: "Dvojitá zákruta",
    22: "Nerovnosť vozovky",
    23: "Nebezpečenstvo šmyku",
    24: "Zúžená vozovka",
    25: "Výkopové práce",
    26: "Svetelná signalizácia",
    27: "Pozor chodci",
    28: "Pozor deti",
    29: "Pozor cyklisti",
    30: "Pozor poľadovica",
    31: "Pozor zver",
    32: "Koniec viacerých zákazov",
    33: "Prikázaný smer jazdy vpravo",
    34: "Prikázaný smer jazdy vľavo",
    35: "Prikázaný smer jazdy rovno",
    36: "Prikázaný smer jazdy rovno/vpravo",
    37: "Prikázaný smer jazdy rovno/vľavo",
    38: "Prikázaný smer obchádzania vpravo",
    39: "Prikázaný smer obchádzania vľavo",
    40: "Kruhový objazd",
    41: "Koniec zákazu predchádzania",
    42: "Koniec zákazu predchádzania pre nákladné vozidlá"
}


# categories_names[train_categories[i * 3000]] # da nam slovne kategoriu namiesto cisla

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.dropout = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2, 2)
        self.l1 = nn.Linear(20 * 6 * 6, 128)
        self.l2 = nn.Linear(128, 43)  # 43 zodpovedana poctu nasich kategorii 0-42

        self.apply(init_weights)  # inicializacia vah

    def forward(self, x):
        # Začíname s 32x32 shape [batch_size,1,32,32]
        x = F.relu(self.conv1(x))
        # [batch_size, 10, 32, 32]
        x = self.pool(x)
        # [batch_size, 10, 16, 16]
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        # [batch_size, 20, 16, 16]
        x = self.pool(x)
        # [batch_size, 20, 8, 8]
        x = x.flatten(1)
        # [batch_size]
        x = F.relu(self.l1(x))
        # [batch_size]
        x = self.l2(x)
        # [batch_size]
        return x


class FullyConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(FullyConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 5, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv4 = nn.Conv2d(48, 64, 5, padding=1)

        self.dropout = nn.Dropout2d(0.3)
        self.pool = nn.MaxPool2d(2, 2)  # Rozdelí každú dimenziu napoly

        # Plne konvolučná vrstva - transformuje príznaky na formu ktoru pouzijeme na klasifikáciu pomocou klasifikačnej vrstvy
        self.fc = nn.Conv2d(64, 128, 1)

        # Klasifikačná vrstva
        self.classifier = nn.Conv2d(128, 43, 1)  # 43 zodpovedá počtu kategórií (0-42)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # normalizacia dat
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)

        self.apply(init_weights)  # Inicializácia váh

    def forward(self, x):
        # Začíname s 32x32 shape [batch_size,1,32,32]
        x = F.relu(self.bn1(self.conv1(x)))
        # [batch_size, 12, 32, 32]
        x = self.pool(x)
        # [batch_size, 12, 16, 16]
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # [batch_size, 24, 14, 14]
        x = self.pool(x)
        # [batch_size, 12, 7, 7]
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        # [batch_size, 48, 6, 6]
        x = self.pool(x)
        # [batch_size, 48, 3, 3]
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        # [batch_size, 64, 1, 1]
        x = F.relu(self.fc(x))
        # [batch_size, 128, 1, 1]
        x = self.adaptive_pool(x)
        # [batch_size, 64, 1, 1]
        x = self.classifier(x)
        # [batch_size, 43, 1, 1]
        x = x.view(x.size(0), -1)
        # [batch_size, 43]
        return x
