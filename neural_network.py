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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)  # z tejto vyjde: [batch_size, 10, 30, 30]
        self.conv2 = nn.Conv2d(10, 20, 3, 1)  # z tejto vyjde: [batch_size, 20, 13, 13]

        self.dropout = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2, 2)  # rozdeli kazdu dimenziu napoly

        # zaciname s 32x32 shape
        # pouzivame dve konvolucne siete kde po prvej mame 30x30 feature map
        # ked aplikujeme pooling s 2x2 tak nam ostane 15x15 feature mapa
        # po druhej konvolucnej sieti nam vnikne 13x13 a po poolingu 6x6
        # mame 20 output channels - musi matchovat posledy output z konvolucnej siete
        # velkost vstupujuca do  linearnej vrstvy musi byt teda 20x6x6

        self.l1 = nn.Linear(20 * 6 * 6, 128)  # 128 cislo mozeme menit, je to len common practice
        self.l2 = nn.Linear(128, 43)  # 43 zodpovedana poctu nasich kategorii 0-42

        self.apply(init_weights)  # inicializacia vah

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.flatten(1)
        x = F.relu(self.l1(x))
        return self.l2(x)
